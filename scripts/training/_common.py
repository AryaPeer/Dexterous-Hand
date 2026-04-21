from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize


class VecNormSyncEvalCallback(EvalCallback):
    """EvalCallback that keeps eval obs normalization in sync with training env."""

    def _on_step(self) -> bool:
        if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = self.training_env.obs_rms
        return super()._on_step()


class RewardInfoLoggerCallback(BaseCallback):
    """Aggregates per-component reward/metric info dicts from vec env step and
    records their mean onto SB3's logger each rollout.

    Picks up any key starting with ``reward/`` or ``metrics/`` that appears in
    ``infos`` and emits it as ``train/<key>`` at rollout end, alongside SB3's
    built-in ``rollout/`` and ``time/`` sections.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._buf: dict[str, list[float]] = defaultdict(list)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            for k, v in info.items():
                if not (k.startswith("reward/") or k.startswith("metrics/")):
                    continue
                try:
                    self._buf[k].append(float(v))
                except (TypeError, ValueError):
                    continue
        return True

    def _on_rollout_end(self) -> None:
        for k, vals in self._buf.items():
            if not vals:
                continue
            self.logger.record(f"train/{k}", float(np.mean(vals)))
        self._buf.clear()


def setup_sb3_logger(model, run_dir: Path) -> None:
    """Point SB3's logger at ``run_dir/logs`` so stdout stays pretty and a CSV
    is written next to the model. CSV is easy to paste into chat or grep later.
    """
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    new_logger = configure(str(log_dir), ["stdout", "csv"])
    model.set_logger(new_logger)
