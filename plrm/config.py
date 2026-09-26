from dataclasses import dataclass, field
from typing import Optional

import transformers


DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TASKS = ("gsm8k", "gsm8k_nl", "commonsenseqa")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=DEFAULT_BASE_MODEL)
    model_revision: Optional[str] = field(default=None)
    lora_r: int = field(default=256)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.1)
    cache_dir: Optional[str] = field(default=None)
    token: Optional[str] = field(default=None, repr=False)
    local_files_only: bool = field(default=False)
    train: bool = field(default=True)
    ckpt_dir: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_name: str = field(default="gsm8k_nl")
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=512)
    max_token_num: int = field(default=256)
    seed: int = field(default=11)
    num_train_epochs: float = field(default=10.0)
    learning_rate: float = field(default=3e-4)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    weight_decay: float = field(default=0.1)
    max_grad_norm: float = field(default=2.0)
    bf16: bool = field(default=True)
    per_device_train_batch_size: int = field(default=64)
    per_device_eval_batch_size: int = field(default=4)
    test_batch_size: int = field(default=16)
    max_new_tokens: int = field(default=256)
    save_only_model: bool = field(default=True)

    num_latent: int = field(default=6)
    latent_dim: int = field(default=768)
    posterior_tokens_per_latent: int = field(default=14)
    posterior_mc_samples: int = field(default=2)
    num_prior_components: int = field(default=1)
    prior_film_gain: float = field(default=3.0)

    kl_factor: float = field(default=3.0)
    kl_balance: float = field(default=0.8)
    kl_warmup_ratio: float = field(default=0.8)
    latent_reg_kl_lambda: float = field(default=0.9)
    exp_loss_weight: float = field(default=1.0)
    logvar_min: float = field(default=-6.0)
    logvar_max: float = field(default=2.0)
    q_logvar_reg_weight: float = field(default=1.0)
    q_var_ceiling: float = field(default=1.0)

    visreg_num_slices: int = field(default=512)
    visreg_scale_weight: float = field(default=1.0)
    visreg_shape_weight: float = field(default=1.0)
    visreg_center_weight: float = field(default=1.0)
    visreg_eps: float = field(default=1e-4)

    mode_temperature: float = field(default=0.5)
    dem_dispersion_gamma: float = field(default=0.9)
    mode_balance_weight: float = field(default=0.0)
