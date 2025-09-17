from __future__ import annotations

from typing import Optional

import torch
import wandb


class TrainerLoggingMixin:
    def _aggregate_epoch_event_counts(self) -> tuple[int, int, int, int]:
        counts_local = torch.tensor(
            [
                self._epoch_loss_nan_inf_count,
                self._epoch_grad_nan_inf_count,
                self._epoch_grad_clip_norm_count,
                self._epoch_grad_clip_value_count,
            ],
            device=self.accelerator.device,
            dtype=torch.long,
        )
        if self.accelerator.num_processes > 1:
            gathered = self.accelerator.gather(counts_local)
            counts_all = gathered.view(-1, 4).sum(dim=0)
        else:
            counts_all = counts_local
        return tuple(int(x.item()) for x in counts_all)

    def _log_epoch_event_counts(self, counts: tuple[int, int, int, int]) -> None:
        if not self.accelerator.is_main_process:
            return
        loss_nan, grad_nan, clip_norm, clip_value = counts
        if wandb.run is not None:
            wandb.log(
                {
                    "train/events/loss_nan_or_inf_count": loss_nan,
                    "train/events/grad_nan_or_inf_count": grad_nan,
                    "train/events/grad_clip_norm_count": clip_norm,
                    "train/events/grad_clip_value_count": clip_value,
                },
                step=self.state.step,
            )
        self.logger.info(
            "Epoch %d | events: loss_nan_or_inf=%d, grad_nan_or_inf=%d, clip_norm=%d, clip_value=%d",
            self.state.epoch + 1,
            loss_nan,
            grad_nan,
            clip_norm,
            clip_value,
        )

    def _log_epoch_train_summary(self, mean_loss: float, avg_step_time: float, epoch_time: float) -> None:
        if not self.accelerator.is_main_process:
            return
        if wandb.run is not None:
            wandb.log(
                {
                    f"train/avg_{self.criterion_name}_loss": mean_loss,
                    "train/avg_step_time": avg_step_time,
                    "train/epoch_time": epoch_time,
                },
                step=self.state.step,
            )
        self.logger.info(
            "Epoch %d | train/avg_%s_loss=%.6f | avg_step_time=%.4fs | epoch_time=%.2fs | step=%d",
            self.state.epoch + 1,
            self.criterion_name,
            mean_loss,
            avg_step_time,
            epoch_time,
            self.state.step,
        )
        self.logger.info("End train_epoch (epoch=%d)", self.state.epoch + 1)

    def _log_train_step_metrics(
        self,
        loss: torch.Tensor,
        coord_loss: Optional[torch.Tensor] = None,
        heatmap_loss: Optional[torch.Tensor] = None,
    ) -> float:
        loss_value = loss.detach()
        coord_value = coord_loss.detach() if coord_loss is not None else None
        heatmap_value = heatmap_loss.detach() if heatmap_loss is not None else None

        if self.accelerator.num_processes > 1:
            loss_value = self.accelerator.gather(loss_value).mean()
            if coord_value is not None:
                coord_value = self.accelerator.gather(coord_value).mean()
            if heatmap_value is not None:
                heatmap_value = self.accelerator.gather(heatmap_value).mean()

        loss_scalar = float(loss_value.item())
        coord_scalar = float(coord_value.item()) if coord_value is not None else None
        heatmap_scalar = float(heatmap_value.item()) if heatmap_value is not None else None

        if self.accelerator.sync_gradients:
            if wandb.run is not None and self.accelerator.is_main_process:
                lr = self.optimizer.param_groups[0]["lr"]
                primary_scalar = coord_scalar if coord_scalar is not None else loss_scalar
                log_payload: dict[str, float] = {f"train/{self.criterion_name}_loss": primary_scalar}
                heatmap_metric_name = getattr(self, "heatmap_criterion_name", None)
                if heatmap_scalar is not None and heatmap_metric_name is not None:
                    log_payload[f"train/{heatmap_metric_name}_loss_heatmap"] = heatmap_scalar
                    log_payload["train/total_loss"] = loss_scalar
                elif coord_scalar is not None and coord_scalar != loss_scalar:
                    log_payload["train/total_loss"] = loss_scalar
                log_payload["train/lr"] = lr
                wandb.log(log_payload, step=self.state.step)
            self.state.increment_step()

        self.state.update_wall_time()

        if self.accelerator.is_main_process:
            parts = [f"loss={loss_scalar:.6f}"]
            if coord_scalar is not None:
                parts.append(f"coord_loss={coord_scalar:.6f}")
            if heatmap_scalar is not None:
                parts.append(f"heatmap_loss={heatmap_scalar:.6f}")
            metrics_str = ", ".join(parts)
            self.logger.debug(
                "End train_step (epoch=%d, step=%d, micro_step=%d, %s)",
                self.state.epoch,
                self.state.step,
                self.state.micro_step,
                metrics_str,
            )

        return loss_scalar

    def _log_train_step_begin(self) -> None:
        if self.accelerator.is_main_process:
            self.logger.debug(
                "Begin train_step (epoch=%d, step=%d, micro_step=%d)",
                self.state.epoch,
                self.state.step,
                self.state.micro_step + 1,
            )

    def _log_train_epoch_begin(self, start_step: int) -> None:
        if self.accelerator.is_main_process:
            self.logger.info(
                "Begin train_epoch (epoch=%d, start_step=%d)",
                self.state.epoch + 1,
                start_step,
            )

    def _log_run_evaluation_skip(self, eval_every: int) -> None:
        if self.accelerator.is_main_process:
            self.logger.info(
                "Skip run_evaluation (epoch=%d): eval_every=%d",
                self.state.epoch,
                eval_every,
            )

    def _log_run_evaluation_begin(self) -> None:
        if self.accelerator.is_main_process:
            self.logger.info(
                "Begin run_evaluation (epoch=%d, step=%d)",
                self.state.epoch,
                self.state.step,
            )

    def _log_run_evaluation_step_begin(self) -> None:
        if self.accelerator.is_main_process:
            self.logger.debug(
                "Begin run_evaluation_step (epoch=%d)",
                self.state.epoch,
            )

    def _log_run_evaluation_step_end(self, metrics: Optional[dict[str, float]] = None) -> None:
        if not self.accelerator.is_main_process:
            return
        metrics = metrics or {}
        if metrics:
            metrics_str = ", ".join(f"{name}=%.6f" % value for name, value in metrics.items())
            message = f"End run_evaluation_step (epoch={self.state.epoch}, {metrics_str})"
        else:
            message = f"End run_evaluation_step (epoch={self.state.epoch})"
        self.logger.debug(message)

    def _log_visualisation_begin(self) -> None:
        if self.accelerator.is_main_process:
            self.logger.info(
                "Begin run_visualisation (epoch=%d, step=%d)",
                self.state.epoch,
                self.state.step,
            )

    def _log_visualisation_end(self, num_examples: int) -> None:
        if self.accelerator.is_main_process:
            self.logger.info(
                "End run_visualisation (epoch=%d, files=%d)",
                self.state.epoch,
                num_examples,
            )

    def _aggregate_validation_mean(self, total_loss_local: float, num_batches_local: int) -> float:
        sum_tensor = torch.tensor([total_loss_local], device=self.accelerator.device, dtype=torch.float32)
        cnt_tensor = torch.tensor([num_batches_local], device=self.accelerator.device, dtype=torch.long)
        if self.accelerator.num_processes > 1:
            sum_all = self.accelerator.gather(sum_tensor).sum()
            cnt_all = self.accelerator.gather(cnt_tensor).sum()
        else:
            sum_all = sum_tensor[0]
            cnt_all = cnt_tensor[0]
        total = float(sum_all.item())
        count = max(int(cnt_all.item()), 1)
        return total / count

    def _log_epoch_val_summary(self, mean_loss: float) -> None:
        if not self.accelerator.is_main_process:
            return
        if wandb.run is not None:
            wandb.log({f"val/avg_{self.criterion_name}_loss": mean_loss}, step=self.state.step)
        self.logger.info(
            "Epoch %d | val/avg_%s_loss=%.6f | step=%d",
            self.state.epoch,
            self.criterion_name,
            mean_loss,
            self.state.step,
        )
        self.logger.info("End run_evaluation (epoch=%d)", self.state.epoch)

    def _log_locator_val_summary(
        self,
        mean_normalized_loss: float,
        mean_pixel_loss: float,
        mean_heatmap_loss: float,
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        metric_norm = f"val/avg_{self.criterion_name}_loss_normalized_v1"
        metric_pixel = f"val/avg_{self.criterion_name}_loss_pixel_v1"
        metric_heatmap = f"val/avg_{self.heatmap_criterion_name}_loss_patch_v1"
        if wandb.run is not None:
            wandb.log(
                {
                    metric_norm: mean_normalized_loss,
                    metric_pixel: mean_pixel_loss,
                    metric_heatmap: mean_heatmap_loss,
                },
                step=self.state.step,
            )
        self.logger.info(
            "Epoch %d | %s=%.6f | %s=%.6f | %s=%.6f | step=%d",
            self.state.epoch,
            metric_norm,
            mean_normalized_loss,
            metric_pixel,
            mean_pixel_loss,
            metric_heatmap,
            mean_heatmap_loss,
            self.state.step,
        )
        self.logger.info("End run_evaluation (epoch=%d)", self.state.epoch)


__all__ = ["TrainerLoggingMixin"]
