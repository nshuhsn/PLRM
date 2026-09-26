import transformers

from plrm.backbone import build_lora_config
from plrm.config import TASKS, DataArguments, ModelArguments, TrainingArguments
from plrm.data import Collator, ReasoningDataset
from plrm.model import PLRM
from plrm.trainer import EpochAccuracyCallback, PLRMTrainer


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.n_gpu > 1:
        raise ValueError("use torchrun for multi-GPU training")
    if data_args.data_name not in TASKS:
        raise ValueError(f"data_name must be one of {', '.join(TASKS)}")
    if not data_args.train_file:
        raise ValueError("--train_file is required")

    transformers.set_seed(training_args.seed)
    model = PLRM(model_args, training_args, build_lora_config(model_args))
    tokenizer = model.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = training_args.model_max_length

    train_dataset = ReasoningDataset(
        data_args.train_file, tokenizer, model, training_args.max_token_num
    )
    eval_dataset = None
    if data_args.validation_file:
        eval_dataset = ReasoningDataset(
            data_args.validation_file, tokenizer, model, training_args.max_token_num
        )
    trainer = PLRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Collator(model.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.data_name = data_args.data_name
    if data_args.validation_file:
        trainer.add_callback(EpochAccuracyCallback(
            trainer, data_args.validation_file, data_args.data_name,
            training_args.test_batch_size, split="val", save_best=True,
        ))
    if data_args.test_file:
        trainer.add_callback(EpochAccuracyCallback(
            trainer, data_args.test_file, data_args.data_name,
            training_args.test_batch_size, split="test", save_best=False,
        ))
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
