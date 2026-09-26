import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model
from transformers.cache_utils import DynamicCache

from plrm.backbone import (
    INTERNAL_TOKEN_COUNT,
    autocast_dtype,
    configure_pad_token,
    input_embedding,
    load_causal_lm,
    load_tokenizer,
    model_family_from_config,
    token_id_set,
)
from plrm.distributed import gather_batch, global_mean, shared_directions


IGNORE_INDEX = -100
LEADIN = "The reasoning is:"


def as_cache(past_key_values):
    if isinstance(past_key_values, DynamicCache):
        return past_key_values
    return DynamicCache.from_legacy_cache(past_key_values)


def as_legacy(past_key_values):
    if isinstance(past_key_values, DynamicCache):
        return past_key_values.to_legacy_cache()
    return past_key_values


def positions_from_mask(mask):
    return (mask.long().cumsum(-1) - 1).clamp(min=0)


def repeat_cache(past_key_values, times):
    return tuple(
        tuple(state.repeat(times, 1, 1, 1) for state in layer)
        for layer in as_legacy(past_key_values)
    )


class PLRM(nn.Module):
    accepts_loss_kwargs = False
    _quantile_cache = {}

    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.dtype = autocast_dtype(training_args.bf16)

        self.num_latent = training_args.num_latent
        self.posterior_mc_samples = training_args.posterior_mc_samples
        self.posterior_tokens_per_latent = training_args.posterior_tokens_per_latent
        self.num_prior_components = training_args.num_prior_components
        self.prior_film_gain = training_args.prior_film_gain
        self.kl_factor = training_args.kl_factor
        self.kl_balance = training_args.kl_balance
        self.kl_warmup_ratio = training_args.kl_warmup_ratio
        self.latent_reg_kl_lambda = training_args.latent_reg_kl_lambda
        self.exp_loss_weight = training_args.exp_loss_weight
        self.logvar_min = training_args.logvar_min
        self.logvar_max = training_args.logvar_max
        self.q_logvar_reg_weight = training_args.q_logvar_reg_weight
        self.q_var_ceiling = training_args.q_var_ceiling
        self.visreg_num_slices = training_args.visreg_num_slices
        self.visreg_scale_weight = training_args.visreg_scale_weight
        self.visreg_shape_weight = training_args.visreg_shape_weight
        self.visreg_center_weight = training_args.visreg_center_weight
        self.visreg_eps = training_args.visreg_eps
        self.mode_temperature = training_args.mode_temperature
        self.dem_dispersion_gamma = training_args.dem_dispersion_gamma
        self.mode_balance_weight = training_args.mode_balance_weight
        if self.posterior_mc_samples < 1:
            raise ValueError("posterior_mc_samples must be >= 1")
        if self.num_prior_components < 1:
            raise ValueError("num_prior_components must be >= 1")
        if self.q_var_ceiling <= 0.0:
            raise ValueError("q_var_ceiling must be > 0")
        if not 0.0 <= self.dem_dispersion_gamma <= 1.0:
            raise ValueError("dem_dispersion_gamma must be in [0, 1]")

        self.backbone = self._build_backbone(lora_config)
        self.pad_token_id = self.vocab_size
        self.bot_id = self.vocab_size + 1
        self.eot_id = self.vocab_size + 2
        self.soe_id = self.vocab_size + 3
        self.lr_id = self.vocab_size + 4
        self.generation_vocab_size = self.lr_id

        self.tokenizer = load_tokenizer(model_args, self.model_family)
        configure_pad_token(self.tokenizer, self.pad_token_id, self.model_family)
        self.eos_token_ids = token_id_set(self.tokenizer.eos_token_id)
        self.eos_token_ids.update(token_id_set(getattr(self.backbone.config, "eos_token_id", None)))

        self.latent_dim = training_args.latent_dim or self.hidden_size
        self.latent_head = self._build_latent_head()
        self.leadin_ids = self.tokenizer(LEADIN, add_special_tokens=False)["input_ids"]

        if model_args.train:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(
                f"trainable params: {trainable} || all params: {total} || "
                f"trainable%: {100 * trainable / total:.4f}",
                flush=True,
            )

    def _build_backbone(self, lora_config):
        backbone = load_causal_lm(self.model_args, self.dtype)
        self.model_family = model_family_from_config(backbone.config)
        self.vocab_size = backbone.config.vocab_size
        backbone.resize_token_embeddings(self.vocab_size + INTERNAL_TOKEN_COUNT)
        self.hidden_size = backbone.config.hidden_size
        return get_peft_model(backbone, lora_config)

    def _build_latent_head(self):
        multiple = 256
        hidden = int(2 * (4 * self.hidden_size) / 3)
        hidden = multiple * ((hidden + multiple - 1) // multiple)
        components = self.num_prior_components
        head = nn.ModuleDict(
            {
                "latent_norm": nn.RMSNorm(self.hidden_size),
                "latent_gate": nn.Linear(self.hidden_size, hidden, bias=False),
                "latent_value": nn.Linear(self.hidden_size, hidden, bias=False),
                "latent_mu_head": nn.Linear(hidden, self.latent_dim, bias=True),
                "latent_logvar_head": nn.Linear(hidden, self.latent_dim, bias=True),
                "thought_proj": nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_size), nn.LayerNorm(self.hidden_size)
                ),
                "prior_film_gamma": nn.Embedding(components, hidden),
                "prior_film_beta": nn.Embedding(components, hidden),
                "prior_mu_head": nn.Linear(hidden, self.latent_dim, bias=True),
                "prior_logvar_head": nn.Linear(hidden, self.latent_dim, bias=True),
                "prior_pi_head": nn.Linear(hidden, 1, bias=True),
            }
        )
        nn.init.orthogonal_(head["prior_film_gamma"].weight, gain=self.prior_film_gain)
        nn.init.orthogonal_(head["prior_film_beta"].weight, gain=self.prior_film_gain)
        head = head.to(torch.float32)
        target = -2.0
        p = (target - self.logvar_min) / (self.logvar_max - self.logvar_min)
        p = min(max(p, 1e-4), 1 - 1e-4)
        logvar_bias = math.log(p / (1 - p))
        for name in ("latent_logvar_head", "prior_logvar_head"):
            nn.init.zeros_(head[name].weight)
            nn.init.constant_(head[name].bias, logvar_bias)
        nn.init.zeros_(head["prior_pi_head"].weight)
        nn.init.zeros_(head["prior_pi_head"].bias)
        return head

    def embedding(self):
        return input_embedding(self.backbone)

    def _trunk(self, hidden):
        norm = self.latent_head["latent_norm"]
        x = norm(hidden.to(norm.weight.dtype))
        return F.silu(self.latent_head["latent_gate"](x)) * self.latent_head["latent_value"](x)

    def _bounded_logvar(self, raw):
        return self.logvar_min + (self.logvar_max - self.logvar_min) * torch.sigmoid(raw)

    def posterior_params(self, hidden):
        x = self._trunk(hidden)
        mu = self.latent_head["latent_mu_head"](x)
        logvar = self._bounded_logvar(self.latent_head["latent_logvar_head"](x))
        return mu, logvar

    def prior_params(self, hidden):
        head = self.latent_head
        x = self._trunk(hidden)
        h = (1.0 + head["prior_film_gamma"].weight) * x.unsqueeze(-2) + head["prior_film_beta"].weight
        mu = head["prior_mu_head"](h)
        logvar = self._bounded_logvar(head["prior_logvar_head"](h))
        log_pi = F.log_softmax(head["prior_pi_head"](h).squeeze(-1), dim=-1)
        return mu, logvar, log_pi

    def to_thought(self, z):
        projection = self.latent_head["thought_proj"]
        return projection(z.to(projection[0].weight.dtype)).to(self.dtype)

    @staticmethod
    def _component_kl(mu_q, lv_q, mu_p, lv_p):
        mu_q = mu_q.unsqueeze(-2)
        lv_q = lv_q.unsqueeze(-2)
        return 0.5 * (
            lv_p - lv_q + (torch.exp(lv_q) + (mu_q - mu_p).square()) / torch.exp(lv_p) - 1.0
        ).sum(dim=-1)

    @staticmethod
    def _dem_dispersion(mu_p, lv_p, log_rho):
        mu = mu_p.float()
        lv = lv_p.float()
        rho = log_rho.float().exp()
        precision = torch.exp(-lv)
        eta = mu * precision
        weights = rho.unsqueeze(-1)
        precision_bar = (weights * precision).sum(dim=-2)
        eta_bar = (weights * eta).sum(dim=-2)
        mu_bar = eta_bar / precision_bar
        lv_bar = -torch.log(precision_bar)
        a_component = 0.5 * (mu.square() * precision + lv).sum(dim=-1)
        a_bar = 0.5 * (mu_bar.square() * precision_bar + lv_bar).sum(dim=-1)
        return (rho * a_component).sum(dim=-1) - a_bar

    def _mode_balance(self, log_rho, valid):
        weights = valid.to(log_rho.dtype).view(-1, 1, 1)
        rho = log_rho.exp()
        rho_bar = global_mean((rho * weights).sum(dim=0), weights.sum())
        return (
            math.log(self.num_prior_components) * rho_bar.sum(dim=-1)
            + (rho_bar * rho_bar.clamp_min(1e-12).log()).sum(dim=-1)
        ).mean()

    def _visibility_limits(self, token_counts, latent_count):
        token_counts = token_counts.reshape(-1).long()
        step = torch.arange(1, latent_count + 1, device=token_counts.device).unsqueeze(0)
        limits = torch.minimum(step * self.posterior_tokens_per_latent, token_counts.unsqueeze(1))
        limits = limits.clone()
        limits[:, -1] = token_counts
        return limits

    def posterior_mask(self, ctx_mask, latent_count, posterior_step_idx, dtype):
        batch, ctx_len = ctx_mask.shape
        total_len = ctx_len + latent_count
        device = ctx_mask.device
        real = ctx_mask.bool()
        allow = torch.zeros((batch, total_len, total_len), dtype=torch.bool, device=device)
        causal = torch.tril(torch.ones((ctx_len, ctx_len), dtype=torch.bool, device=device))
        allow[:, :ctx_len, :ctx_len] = causal.unsqueeze(0) & real[:, None, :]

        is_cot = posterior_step_idx > 0
        token_count = posterior_step_idx.amax(dim=-1).clamp(min=1)
        limits = self._visibility_limits(token_count, latent_count)
        columns = torch.arange(ctx_len, device=device).unsqueeze(0)
        sentinel = torch.full_like(columns.expand(batch, -1), ctx_len)
        first_cot = torch.where(is_cot, columns.expand(batch, -1), sentinel).min(dim=-1).values
        prefix = columns < first_cot.unsqueeze(-1)
        allow[:, ctx_len:, :ctx_len] = real.unsqueeze(1) & (
            prefix.unsqueeze(1)
            | (is_cot.unsqueeze(1) & (posterior_step_idx.unsqueeze(1) <= limits.unsqueeze(2)))
        )
        allow[:, ctx_len:, ctx_len:] = torch.tril(
            torch.ones((latent_count, latent_count), dtype=torch.bool, device=device)
        ).unsqueeze(0)
        additive = torch.where(
            allow,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device),
        )
        return additive.unsqueeze(1)

    def encode_posterior(self, posterior_input_ids, posterior_attention_mask, posterior_step_idx):
        emb = self.embedding()
        batch = posterior_input_ids.size(0)
        latent_count = self.num_latent
        context = emb(posterior_input_ids)
        reads = emb(torch.tensor([self.lr_id], device=context.device))
        reads = reads.view(1, 1, -1).expand(batch, latent_count, -1)
        inputs = torch.cat([context, reads], dim=1)
        real_len = posterior_attention_mask.long().sum(dim=1, keepdim=True)
        read_positions = real_len + torch.arange(latent_count, device=context.device).unsqueeze(0)
        position_ids = torch.cat(
            [positions_from_mask(posterior_attention_mask), read_positions], dim=1
        )
        mask = self.posterior_mask(
            posterior_attention_mask, latent_count, posterior_step_idx, inputs.dtype
        )
        out = self.backbone(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        return self.posterior_params(out.hidden_states[-1][:, -latent_count:, :])

    def _normal_quantiles(self, n, device, dtype):
        key = (n, str(device), dtype)
        cached = PLRM._quantile_cache.get(key)
        if cached is None:
            p = torch.arange(1, n + 1, device=device, dtype=dtype) / (n + 1)
            cached = torch.erfinv(2.0 * p - 1.0) * math.sqrt(2.0)
            PLRM._quantile_cache[key] = cached
        return cached

    def _visreg_directions(self, dim, device):
        w = torch.randn(dim, self.visreg_num_slices, device=device, dtype=torch.float32)
        return shared_directions(w / w.norm(dim=0, keepdim=True).clamp(min=1e-8))

    def _visreg(self, mu, valid, directions):
        global_mu = gather_batch(mu.float())
        global_valid = gather_batch(valid, with_grad=False)
        x = global_mu[global_valid]
        n = x.size(0)
        if n < 2:
            zero = global_mu.sum().float() * 0.0
            return zero, zero, zero
        mean = x.mean(dim=0)
        centered = x - mean.unsqueeze(0)
        center_loss = mean.square().mean()
        std = torch.sqrt(centered.square().sum(dim=0) / (n - 1) + self.visreg_eps)
        scale_loss = (1.0 - std).square().mean()
        normalized = centered / (std.detach() + self.visreg_eps)
        projected, _ = torch.sort(torch.einsum("nld,dk->nlk", normalized, directions), dim=0)
        quantiles = self._normal_quantiles(n, x.device, x.dtype).view(n, 1, 1)
        shape_loss = (projected - quantiles).square().mean()
        return scale_loss, shape_loss, center_loss

    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        posterior_input_ids,
        posterior_attention_mask,
        posterior_step_idx,
        decoder_input_ids,
        labels,
        explain_input_ids,
        explain_attention_mask,
        explain_labels,
        step_ratio=None,
        **kwargs,
    ):
        emb = self.embedding()
        batch = encoder_input_ids.size(0)
        latent_count = self.num_latent
        device = encoder_input_ids.device

        mu_q, lv_q = self.encode_posterior(
            posterior_input_ids, posterior_attention_mask, posterior_step_idx
        )
        samples = self.posterior_mc_samples if self.training else 1
        mc_batch = samples * batch
        if samples > 1:
            eps = torch.randn((samples,) + tuple(mu_q.shape), device=mu_q.device, dtype=mu_q.dtype)
            z = mu_q.unsqueeze(0) + eps * torch.exp(0.5 * lv_q).unsqueeze(0)
            z = z.reshape(mc_batch, latent_count, z.size(-1))
        else:
            z = mu_q + torch.randn_like(mu_q) * torch.exp(0.5 * lv_q)
        thoughts = self.to_thought(z)

        query_out = self.backbone(
            inputs_embeds=emb(encoder_input_ids),
            attention_mask=encoder_attention_mask,
            position_ids=positions_from_mask(encoder_attention_mask),
            use_cache=True,
            output_hidden_states=True,
        )
        query_past = query_out.past_key_values
        if samples > 1:
            query_past_mc = repeat_cache(query_past, samples)
        token_index = torch.arange(encoder_input_ids.size(1), device=device)
        bot_index = token_index.expand_as(encoder_attention_mask).masked_fill(
            ~encoder_attention_mask.bool(), -1
        ).amax(dim=1)
        bot_hidden = query_out.hidden_states[-1][
            torch.arange(batch, device=device), bot_index
        ].unsqueeze(1)
        query_len = encoder_attention_mask.long().sum(dim=1, keepdim=True)
        latent_positions = torch.arange(latent_count, device=device).unsqueeze(0)
        z_pos = query_len + latent_positions
        ones = torch.ones(
            (batch, latent_count), dtype=encoder_attention_mask.dtype, device=device
        )
        prefix_mask = torch.cat([encoder_attention_mask, ones], dim=1)
        prefix_out = self.backbone(
            inputs_embeds=thoughts[:batch] if samples > 1 else thoughts,
            attention_mask=prefix_mask,
            position_ids=z_pos,
            past_key_values=as_cache(query_past),
            use_cache=True,
            output_hidden_states=True,
        )
        prefix_past = as_legacy(prefix_out.past_key_values)
        prefix_len = query_len + latent_count
        prior_h = torch.cat([bot_hidden, prefix_out.hidden_states[-1][:, :-1, :]], dim=1)
        if samples > 1:
            mask_mc = encoder_attention_mask.repeat(samples, 1)
            prefix_out_mc = self.backbone(
                inputs_embeds=thoughts,
                attention_mask=torch.cat(
                    [mask_mc, torch.ones((mc_batch, latent_count), dtype=mask_mc.dtype, device=device)],
                    dim=1,
                ),
                position_ids=query_len.repeat(samples, 1) + latent_positions,
                past_key_values=as_cache(query_past_mc),
                use_cache=True,
                output_hidden_states=True,
            )
            prior_h = torch.cat(
                [bot_hidden.repeat(samples, 1, 1), prefix_out_mc.hidden_states[-1][:, :-1, :]],
                dim=1,
            )
        mu_p, lv_p, log_pi = self.prior_params(prior_h)

        mu_q_first = mu_q.float()
        mu_qf = (mu_q.repeat(samples, 1, 1) if samples > 1 else mu_q).float()
        lv_qf = (lv_q.repeat(samples, 1, 1) if samples > 1 else lv_q).float()
        mu_pf, lv_pf, log_pi_f = mu_p.float(), lv_p.float(), log_pi.float()

        log_pi_first = log_pi_f[:, :1]
        kl_first = self._component_kl(mu_qf[:, :1], lv_qf[:, :1], mu_pf[:, :1], lv_pf[:, :1])
        temperature = torch.as_tensor(
            self.mode_temperature, device=kl_first.device, dtype=kl_first.dtype
        )
        log_rho_live = F.log_softmax(-kl_first / temperature, dim=-1)
        log_rho = log_rho_live.detach()
        rho = log_rho.exp()
        log_rho_steps = log_rho.expand(-1, latent_count, -1)
        rho_steps = log_rho_steps.exp()

        alpha = self.kl_balance
        kl_c_to_p = (rho * (log_rho - log_pi_first)).sum(dim=-1)
        kl_z_to_p = (
            rho_steps * self._component_kl(mu_qf.detach(), lv_qf.detach(), mu_pf, lv_pf)
        ).sum(dim=-1)
        dem_to_p = self._dem_dispersion(mu_pf, lv_pf, log_rho_steps)
        kl_c_to_q = (rho * (log_rho - log_pi_first.detach())).sum(dim=-1)
        kl_z_to_q = (
            rho_steps * self._component_kl(mu_qf, lv_qf, mu_pf.detach(), lv_pf.detach())
        ).sum(dim=-1)
        dem_to_q = self._dem_dispersion(mu_pf.detach(), lv_pf.detach(), log_rho_steps)
        kl_c = alpha * kl_c_to_p + (1.0 - alpha) * kl_c_to_q
        kl_z = alpha * kl_z_to_p + (1.0 - alpha) * kl_z_to_q
        dem = alpha * dem_to_p + (1.0 - alpha) * dem_to_q
        kl = kl_z - self.dem_dispersion_gamma * dem

        has_cot = (explain_labels != IGNORE_INDEX).any(dim=1)
        has_cot_mc = has_cot.repeat(samples) if samples > 1 else has_cot
        cot_mask = has_cot_mc.to(kl.dtype).view(mc_batch, 1)
        valid = has_cot_mc.sum().to(kl.dtype)
        mode_balance = self._mode_balance(log_rho_live, has_cot_mc)
        kl_by_step = global_mean((kl * cot_mask).sum(dim=0), valid)
        kl_c_mean = global_mean((kl_c.squeeze(-1) * cot_mask.squeeze(-1)).sum(), valid)
        kl_raw = kl_by_step.sum() + kl_c_mean
        kl_eff = kl_by_step.mean() + kl_c_mean

        directions = self._visreg_directions(mu_q_first.size(-1), mu_q_first.device)
        scale_loss, shape_loss, center_loss = self._visreg(mu_q_first, has_cot, directions)

        beta = self.kl_factor
        if step_ratio is not None:
            beta *= min(1.0, max(0.0, float(step_ratio)) / self.kl_warmup_ratio)
        kl_weight = beta * (1.0 - self.latent_reg_kl_lambda)
        reg_scale = beta * self.latent_reg_kl_lambda

        dec_mask = decoder_input_ids.ne(self.pad_token_id)
        ans_out = self.backbone(
            inputs_embeds=emb(decoder_input_ids),
            attention_mask=torch.cat([prefix_mask, dec_mask], dim=1),
            position_ids=prefix_len + torch.arange(decoder_input_ids.size(1), device=device).unsqueeze(0),
            past_key_values=as_cache(prefix_past),
            use_cache=False,
        )
        logits = ans_out.logits
        shift_labels = labels[:, 1:].reshape(-1)
        ans_sum = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            shift_labels,
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        ans_loss = global_mean(ans_sum, (shift_labels != IGNORE_INDEX).sum())

        soe = emb(torch.tensor([self.soe_id], device=device)).unsqueeze(0).expand(batch, -1, -1)
        lead = emb(torch.tensor(self.leadin_ids, device=device)).unsqueeze(0).expand(batch, -1, -1)
        continuation = torch.cat([soe, lead, emb(explain_input_ids)], dim=1)
        pre_len = 1 + len(self.leadin_ids)
        continuation_mask = torch.cat(
            [
                torch.ones((batch, pre_len), dtype=explain_attention_mask.dtype, device=device),
                explain_attention_mask,
            ],
            dim=1,
        )
        z_only_out = self.backbone(
            inputs_embeds=thoughts[:batch] if samples > 1 else thoughts,
            attention_mask=ones,
            position_ids=z_pos,
            use_cache=True,
        )
        exp_out = self.backbone(
            inputs_embeds=continuation,
            attention_mask=torch.cat([ones, continuation_mask], dim=1),
            position_ids=prefix_len + torch.arange(continuation.size(1), device=device).unsqueeze(0),
            past_key_values=as_cache(as_legacy(z_only_out.past_key_values)),
            use_cache=False,
        )
        exp_logits = exp_out.logits
        exp_labels = torch.cat(
            [
                torch.full((batch, pre_len), IGNORE_INDEX, dtype=explain_labels.dtype, device=device),
                explain_labels.masked_fill(~explain_attention_mask.bool(), IGNORE_INDEX),
            ],
            dim=1,
        )
        exp_target = exp_labels[:, 1:].reshape(-1)
        exp_sum = F.cross_entropy(
            exp_logits[:, :-1, :].reshape(-1, exp_logits.size(-1)),
            exp_target,
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        exp_loss = global_mean(exp_sum, (exp_target != IGNORE_INDEX).sum())

        var_q = torch.exp(lv_qf)
        ratio = var_q / self.q_var_ceiling
        q_penalty = 0.5 * (ratio - torch.log(ratio) - 1.0)
        q_penalty = q_penalty * (var_q > self.q_var_ceiling).to(q_penalty.dtype)
        q_penalty = global_mean(q_penalty.sum(), q_penalty.numel())

        ce_loss = ans_loss + self.exp_loss_weight * exp_loss
        balance_loss = beta * self.mode_balance_weight * mode_balance
        loss = (
            ce_loss
            + kl_weight * kl_eff
            + self.q_logvar_reg_weight * q_penalty
            + self.visreg_scale_weight * reg_scale * scale_loss
            + self.visreg_shape_weight * reg_scale * shape_loss
            + self.visreg_center_weight * reg_scale * center_loss
            + balance_loss
        )
        return {
            "loss": loss,
            "logits": logits,
            "ce_loss": ce_loss.detach(),
            "ans_loss": ans_loss.detach(),
            "exp_loss": exp_loss.detach(),
            "kl_loss": kl_raw.detach(),
            "kl_eff": kl_eff.detach(),
            "kl_c": kl_c.mean().detach(),
            "dem_dispersion": dem.mean().detach(),
            "visreg_scale_loss": scale_loss.detach(),
            "visreg_shape_loss": shape_loss.detach(),
            "visreg_center_loss": center_loss.detach(),
            "q_logvar_reg": q_penalty.detach(),
            "mode_balance_loss": mode_balance.detach(),
            "beta": torch.tensor(beta, device=loss.device),
        }
