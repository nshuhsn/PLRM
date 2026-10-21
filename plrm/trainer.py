import json
import math
import os
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback

from plrm.data import gold_answers, read_jsonl
from plrm.distributed import world_size
from plrm.inference import CONFIG_NAME, checkpoint_config, iter_predictions
from plrm.metrics import is_correct


LOGGED_METRICS = (
    "ce_loss", "ans_loss", "exp_loss", "kl_loss", "kl_eff", "kl_c", "dem_dispersion",
    "visreg_scale_loss", "visreg_shape_loss", "visreg_center_loss", "q_logvar_reg",
    "mode_balance_loss", "beta",
)


class PLRMTrainer(Trainer):
    data_name = None

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict=state_dict)
        model = self.accelerator.unwrap_model(self.model)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as handle:
            json.dump(checkpoint_config(model, self.data_name), handle, indent=2)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        samples, num_items = super().get_batch_samples(epoch_iterator, num_batches, device)
        self._actual_accumulation_steps = max(1, len(samples))
        return samples, num_items

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if world_size() > 1:
            self.accelerator.wait_for_everyone()
            if self.state.best_global_step:
                best_path = os.path.join(
                    self._get_output_dir(trial), f"checkpoint-{self.state.best_global_step}"
                )
                if os.path.isdir(best_path):
                    self.state.best_model_checkpoint = best_path

    def _step_ratio(self):
        total_steps = self.state.max_steps
        if total_steps <= 0:
            effective_batch = (
                self.args.per_device_train_batch_size
                * self.args.world_size
                * self.args.gradient_accumulation_steps
            )
            total_steps = max(
                1, math.ceil(len(self.train_dataset) / effective_batch) * self.args.num_train_epochs
            )
        return self.state.global_step / total_steps

    @staticmethod
    def _reduce(value):
        value = value.detach().float().mean().clone()
        if world_size() > 1:
            torch.distributed.all_reduce(value)
            value /= world_size()
        return value.item()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["step_ratio"] = self._step_ratio()
        outputs = model(**inputs)
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({name: self._reduce(outputs[name]) for name in LOGGED_METRICS})
        loss = outputs["loss"]
        if model.training and not return_outputs:
            actual_steps = getattr(
                self, "_actual_accumulation_steps", self.args.gradient_accumulation_steps
            )
            loss = loss * (self.args.gradient_accumulation_steps / actual_steps)
        return (loss, outputs) if return_outputs else loss


class EpochAccuracyCallback(TrainerCallback):
    def __init__(self, trainer, path, task, batch_size, split, save_best):
        self.trainer = trainer
        self.rows = read_jsonl(path)
        if not self.rows:
            raise ValueError(f"{path} contains no examples")
        if len({row["example_id"] for row in self.rows}) != len(self.rows):
            raise ValueError(f"{path} example_id values must be unique")
        self.task = task
        # Teacher (posterior) accuracy needs a gold CoT; grouped multi-answer rows have none.
        self.has_cot = all(str(row.get("cot", "")).strip() for row in self.rows)
        self.batch_size = batch_size
        self.split = split
        self.save_best = save_best
        self.best_correct = -1
        self.best_epoch = None

    def _count(self, model, rows, teacher):
        answers = {row["example_id"]: gold_answers(row) for row in rows}
        device = next(model.parameters()).device
        correct = total = limited = 0
        cuda_devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices), self.trainer.accelerator.autocast():
            for prediction in iter_predictions(model, rows, self.batch_size, self.task, teacher=teacher):
                correct += int(is_correct(prediction["prediction"], answers[prediction["example_id"]], self.task))
                total += 1
                limited += prediction["finish_reason"] == "length"
        counts = torch.tensor([correct, total, limited], device=device, dtype=torch.long)
        if world_size() > 1:
            torch.distributed.all_reduce(counts)
        return counts.tolist()

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
        rank = torch.distributed.get_rank() if world_size() > 1 else 0
        rows = self.rows[rank::world_size()]
        correct, total, limited = self._count(model, rows, teacher=False)
        teacher_correct = self._count(model, rows, teacher=True)[0] if self.has_cot else None
        metrics = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "n_correct": correct,
            "n_total": total,
            "accuracy": correct / total,
            "teacher_accuracy": None if teacher_correct is None else teacher_correct / total,
            "length_limit_rate": limited / total,
        }
        output_dir = Path(args.output_dir)
        if self.save_best and correct > self.best_correct:
            best_dir = output_dir / f"best_{self.split}_accuracy"
            trainer.save_model(str(best_dir), _internal_call=True)
            if trainer.is_world_process_zero():
                with (best_dir / "selection.json").open("w", encoding="utf-8") as handle:
                    json.dump({**metrics, "criterion": f"maximum {self.split} accuracy"}, handle, indent=2)
            if world_size() > 1:
                trainer.accelerator.wait_for_everyone()
            self.best_correct = correct
            self.best_epoch = state.epoch
        if trainer.is_world_process_zero():
            with (output_dir / f"epoch_{self.split}_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            teacher = metrics["teacher_accuracy"]
            teacher_text = "" if teacher is None else f" teacher_accuracy={100 * teacher:.2f}%"
            print(
                f"[{self.split}] epoch={state.epoch:.2f} accuracy={100 * metrics['accuracy']:.2f}%"
                f"{teacher_text} ({correct}/{total})",
                flush=True,
            )
        logs = {f"{self.split}_accuracy": metrics["accuracy"]}
        if metrics["teacher_accuracy"] is not None:
            logs[f"{self.split}_teacher_accuracy"] = metrics["teacher_accuracy"]
        trainer.log(logs)
        return control
