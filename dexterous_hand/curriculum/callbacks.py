from stable_baselines3.common.callbacks import BaseCallback


class ReorientCurriculumCallback(BaseCallback):
    """Gradually increases the target rotation difficulty during training."""

    def __init__(self, stages: list[tuple[int, float]], verbose: int = 0) -> None:
        """Set up curriculum stages.

        Args:
            stages: list of (timestep, max_angle_rad) — when to increase difficulty
            verbose: print transitions if > 0
        """
        super().__init__(verbose)
        self.stages = stages
        self._current_stage = 0

    def _on_step(self) -> bool:
        """Check if we should advance to a harder stage."""
        while (
            self._current_stage < len(self.stages) - 1
            and self.num_timesteps >= self.stages[self._current_stage + 1][0]
        ):
            self._current_stage += 1
            max_angle = self.stages[self._current_stage][1]
            self.training_env.env_method("set_curriculum_stage", max_angle)
            if self.verbose:
                print(
                    f"[Curriculum] Stage {self._current_stage}: "
                    f"max_angle={max_angle:.2f} rad at step {self.num_timesteps}"
                )
        return True


class AssemblyCurriculumCallback(BaseCallback):
    """Gradually makes the peg-in-hole task harder (tighter clearance, no pre-grasp)."""

    def __init__(self, stages: list[tuple[int, float, bool]], verbose: int = 0) -> None:
        """Set up curriculum stages.

        Args:
            stages: list of (timestep, clearance, peg_pre_grasped) tuples
            verbose: print transitions if > 0
        """
        super().__init__(verbose)
        self.stages = stages
        self._current_stage = 0

    def _on_step(self) -> bool:
        """Check if we should advance to a harder stage."""
        while (
            self._current_stage < len(self.stages) - 1
            and self.num_timesteps >= self.stages[self._current_stage + 1][0]
        ):
            self._current_stage += 1
            clearance = self.stages[self._current_stage][1]
            pre_grasped = self.stages[self._current_stage][2]
            self.training_env.env_method("set_curriculum_params", clearance, pre_grasped)
            if self.verbose:
                print(
                    f"[Curriculum] Stage {self._current_stage}: "
                    f"clearance={clearance * 1000:.1f}mm, "
                    f"pre_grasped={pre_grasped} at step {self.num_timesteps}"
                )
        return True
