"""Strict checkpoint loading and inference-only serving for real evaluation."""

import torch

from rlinf.utils.realworld_eval import evaluation_checkpoint
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class RealworldEvalRolloutWorker(MultiStepRolloutWorker):
    def init_worker(self):
        """Load a complete policy or the explicitly requested initial residual."""
        checkpoint = evaluation_checkpoint(self.cfg)
        self.cfg.runner.ckpt_path = checkpoint
        if self.cfg.actor.get("initial_checkpoint_exclude_keys", []):
            raise ValueError(
                "Evaluation must load all checkpoint weights, including logstd"
            )
        torch.manual_seed(int(self.cfg.actor.seed))
        super().init_worker()
        self.hf_model.requires_grad_(False)
        for name, value in self.hf_model.state_dict().items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite checkpoint weight: {name}")
        if self.cfg.rollout.get("residual_base_inference", False):
            from rlinf.workers.rollout.hf.residual_inference import (
                RolloutResidualInference,
            )

            self.residual_inference = RolloutResidualInference(self.cfg, self.device)
        return {
            "checkpoint": checkpoint,
            "loaded": checkpoint is not None,
            "policy_source": "checkpoint" if checkpoint else "initial_residual",
        }

    async def serve(self, observations, actions):
        """Serve actions until STOP, never compute gradients or update weights."""
        mode = "eval" if self.cfg.evaluation.policy_mode == "deterministic" else "train"
        while True:
            request = await observations.get(key="obs", async_op=True).async_wait()
            if request.get("stop", False):
                break
            try:
                action, _ = self.predict(request["obs"], mode=mode)
                actions.put({"actions": action.detach().cpu().numpy()}, key="actions")
            except Exception as exc:
                actions.put({"error": f"{type(exc).__name__}: {exc}"}, key="actions")
                break
