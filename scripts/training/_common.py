from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize


class VecNormSyncEvalCallback(EvalCallback):
    """EvalCallback that keeps eval obs normalization in sync with training env."""

    def _on_step(self) -> bool:
        if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = self.training_env.obs_rms
        return super()._on_step()
