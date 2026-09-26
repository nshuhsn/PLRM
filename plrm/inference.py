import json
import os

import torch

from plrm.backbone import build_lora_config
from plrm.config import ModelArguments, TrainingArguments
from plrm.data import normalize_question
from plrm.model import PLRM, positions_from_mask


CONFIG_NAME = "plrm_config.json"
DECODE_MODES = ("mode", "sample")
CONFIG_KEYS = (
    "num_latent", "latent_dim", "posterior_tokens_per_latent", "num_prior_components",
    "model_max_length", "max_new_tokens", "bf16",
)


def checkpoint_config(model, data_name):
    model_args = model.model_args
    return {
        "model_name_or_path": model_args.model_name_or_path,
        "model_revision": model_args.model_revision,
        "model_family": model.model_family,
        "lora_r": model_args.lora_r,
        "lora_alpha": model_args.lora_alpha,
        "lora_dropout": model_args.lora_dropout,
        "data_name": data_name,
        **{key: getattr(model.training_args, key) for key in CONFIG_KEYS},
        "latent_dim": model.latent_dim,
    }


def load_model(checkpoint, base_model=None, model_revision=None, device=None):
    with open(os.path.join(checkpoint, CONFIG_NAME), encoding="utf-8") as handle:
        config = json.load(handle)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = ModelArguments(
        model_name_or_path=base_model or config["model_name_or_path"],
        model_revision=model_revision or (config["model_revision"] if base_model is None else None),
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        train=False,
        ckpt_dir=checkpoint,
    )
    training_args = TrainingArguments(
        output_dir=os.path.join(checkpoint, "eval"),
        use_cpu=device.type == "cpu",
        report_to="none",
        **{key: config[key] for key in CONFIG_KEYS},
    )
    model = PLRM(model_args, training_args, build_lora_config(model_args))
    safetensors_path = os.path.join(checkpoint, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state = load_file(safetensors_path)
    else:
        state = torch.load(os.path.join(checkpoint, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _select(mu, logvar, log_pi, decode_mode, component=None):
    if component is None:
        if decode_mode == "mode":
            component = log_pi.argmax(dim=-1, keepdim=True)
        elif decode_mode == "sample":
            component = torch.distributions.Categorical(logits=log_pi).sample().unsqueeze(-1)
        else:
            raise ValueError(f"unknown decode_mode: {decode_mode}")
    index = component.unsqueeze(-1).expand(-1, -1, 1, mu.size(-1))
    z = mu.gather(2, index).squeeze(2)
    if decode_mode == "sample":
        z = z + torch.exp(0.5 * logvar.gather(2, index).squeeze(2)) * torch.randn_like(z)
    return z, component


@torch.no_grad()
def prior_thoughts(model, input_ids, attention_mask, decode_mode="mode"):
    out = model.backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=positions_from_mask(attention_mask),
        use_cache=True,
        output_hidden_states=True,
    )
    past = out.past_key_values
    hidden = out.hidden_states[-1][:, -1:, :]
    real_len = attention_mask.long().sum(dim=1, keepdim=True)
    running_mask = attention_mask
    component = None
    for index in range(model.num_latent):
        mu, logvar, log_pi = model.prior_params(hidden)
        z, component = _select(mu, logvar, log_pi, decode_mode, component)
        thought = model.to_thought(z)
        running_mask = torch.cat(
            [running_mask, torch.ones((input_ids.size(0), 1), dtype=running_mask.dtype, device=running_mask.device)],
            dim=1,
        )
        out = model.backbone(
            inputs_embeds=thought,
            attention_mask=running_mask,
            position_ids=real_len + index,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
        )
        past = out.past_key_values
        hidden = out.hidden_states[-1][:, -1:, :]
    return past, running_mask, real_len + model.num_latent


@torch.no_grad()
def posterior_prefix(model, input_ids, attention_mask, post_ids, post_mask, step_idx):
    mu, _ = model.encode_posterior(post_ids, post_mask, step_idx)
    thoughts = model.to_thought(mu)
    out = model.backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=positions_from_mask(attention_mask),
        use_cache=True,
    )
    real_len = attention_mask.long().sum(dim=1, keepdim=True)
    latent_count = thoughts.size(1)
    running_mask = torch.cat(
        [attention_mask, torch.ones((input_ids.size(0), latent_count), dtype=attention_mask.dtype, device=attention_mask.device)],
        dim=1,
    )
    out = model.backbone(
        inputs_embeds=thoughts,
        attention_mask=running_mask,
        position_ids=real_len + torch.arange(latent_count, device=attention_mask.device).unsqueeze(0),
        past_key_values=out.past_key_values,
        use_cache=True,
    )
    return out.past_key_values, running_mask, real_len + latent_count


@torch.no_grad()
def greedy_decode(model, past, attention_mask, position, max_new_tokens):
    batch = attention_mask.size(0)
    device = attention_mask.device
    emb = model.embedding()
    ones = torch.ones((batch, 1), dtype=attention_mask.dtype, device=device)
    attention_mask = torch.cat([attention_mask, ones], dim=1)
    out = model.backbone(
        inputs_embeds=emb(torch.tensor([model.eot_id], device=device)).unsqueeze(0).expand(batch, -1, -1),
        attention_mask=attention_mask,
        position_ids=position,
        past_key_values=past,
        use_cache=True,
    )
    past = out.past_key_values
    logits = out.logits[:, -1, : model.generation_vocab_size]
    position = position + 1
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    tokens = [[] for _ in range(batch)]
    for step in range(max_new_tokens):
        next_token = logits.argmax(dim=-1)
        for row in range(batch):
            if finished[row]:
                continue
            value = next_token[row].item()
            if value in model.eos_token_ids:
                finished[row] = True
            else:
                tokens[row].append(value)
        if finished.all() or step + 1 == max_new_tokens:
            break
        attention_mask = torch.cat([attention_mask, ones], dim=1)
        out = model.backbone(
            inputs_embeds=emb(next_token).unsqueeze(1),
            attention_mask=attention_mask,
            position_ids=position,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, : model.generation_vocab_size]
        position = position + 1
    return tokens


def _left_pad(rows, value):
    width = max(map(len, rows))
    result = torch.full((len(rows), width), value, dtype=torch.long)
    for index, row in enumerate(rows):
        result[index, width - len(row):] = torch.tensor(row, dtype=torch.long)
    return result


@torch.no_grad()
def iter_predictions(model, rows, batch_size, decode_mode="mode", teacher=False, max_new_tokens=None):
    max_new_tokens = max_new_tokens or model.training_args.max_new_tokens
    device = next(model.parameters()).device
    tokenizer = model.tokenizer
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            question_ids = [
                tokenizer(normalize_question(row["question"]), add_special_tokens=False)["input_ids"]
                for row in batch
            ]
            input_ids = _left_pad([ids + [model.bot_id] for ids in question_ids], model.pad_token_id).to(device)
            attention_mask = input_ids.ne(model.pad_token_id).long()
            if teacher:
                post_rows, step_rows = [], []
                for ids, row in zip(question_ids, batch):
                    cot = str(row.get("cot", "")).strip()
                    cot_ids = tokenizer(cot, add_special_tokens=False)["input_ids"] if cot else []
                    post_rows.append(ids + cot_ids)
                    step_rows.append([0] * len(ids) + list(range(1, len(cot_ids) + 1)))
                post_ids = _left_pad(post_rows, model.pad_token_id).to(device)
                step_idx = _left_pad(step_rows, 0).to(device)
                past, mask, position = posterior_prefix(
                    model, input_ids, attention_mask, post_ids, post_ids.ne(model.pad_token_id).long(), step_idx
                )
            else:
                past, mask, position = prior_thoughts(model, input_ids, attention_mask, decode_mode)
            for row, token_ids in zip(batch, greedy_decode(model, past, mask, position, max_new_tokens)):
                yield {
                    "example_id": row["example_id"],
                    "prediction": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                    "generated_token_count": len(token_ids),
                    "finish_reason": "length" if len(token_ids) == max_new_tokens else "eos",
                }
    finally:
        model.train(was_training)
