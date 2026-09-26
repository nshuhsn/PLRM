import os

import torch
from peft import LoraConfig, TaskType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


INTERNAL_TOKEN_COUNT = 5
LORA_TARGETS = {
    "gpt2": ("c_attn", "c_proj", "c_fc"),
    "llama": (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
}
PAD_TOKENS = {"gpt2": "[PAD]", "llama": "<|latent_pad|>"}


def _pretrained_kwargs(model_args):
    return {
        "cache_dir": model_args.cache_dir,
        "token": model_args.token,
        "local_files_only": model_args.local_files_only,
        "revision": model_args.model_revision,
    }


def model_family_from_config(config):
    model_type = getattr(config, "model_type", "").lower()
    if model_type not in LORA_TARGETS:
        raise ValueError(
            f"unsupported backbone model_type={model_type!r}; "
            f"supported: {', '.join(sorted(LORA_TARGETS))}"
        )
    return model_type


def build_lora_config(model_args, inference_mode=False):
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, **_pretrained_kwargs(model_args)
    )
    family = model_family_from_config(config)
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=inference_mode,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=list(LORA_TARGETS[family]),
        init_lora_weights=True,
    )


def load_causal_lm(model_args, dtype):
    return AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=dtype, **_pretrained_kwargs(model_args)
    )


def _has_tokenizer_files(directory):
    if not directory or not os.path.isdir(directory):
        return False
    config_exists = os.path.exists(os.path.join(directory, "tokenizer_config.json"))
    vocabulary_exists = any(
        os.path.exists(os.path.join(directory, name))
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    )
    return config_exists and vocabulary_exists


def load_tokenizer(model_args, model_family):
    kwargs = _pretrained_kwargs(model_args)
    source = model_args.model_name_or_path
    if _has_tokenizer_files(model_args.ckpt_dir):
        source = model_args.ckpt_dir
        kwargs["local_files_only"] = True
        kwargs.pop("revision")
    return AutoTokenizer.from_pretrained(source, use_fast=model_family != "gpt2", **kwargs)


def configure_pad_token(tokenizer, pad_token_id, model_family):
    if tokenizer.pad_token_id == pad_token_id:
        return
    tokenizer.add_special_tokens({"pad_token": PAD_TOKENS[model_family]})
    if tokenizer.pad_token_id != pad_token_id:
        raise ValueError(
            f"tokenizer/model vocabulary mismatch: expected the pad token at id "
            f"{pad_token_id}, got {tokenizer.pad_token_id}"
        )


def token_id_set(token_ids):
    if token_ids is None:
        return set()
    if isinstance(token_ids, int):
        return {token_ids}
    return {int(token_id) for token_id in token_ids}


def input_embedding(model):
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base_model.get_input_embeddings()


def autocast_dtype(bf16):
    return torch.bfloat16 if bf16 else torch.float16
