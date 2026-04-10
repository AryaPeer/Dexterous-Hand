from stable_baselines3.common.callbacks import BaseCallback


def scale_stage_starts(
    stages: list[tuple],
    total_timesteps: int,
    reference_total_timesteps: int,
) -> list[tuple]:
    """Scale curriculum stage start steps to match a total timestep budget."""

    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be > 0")
    if reference_total_timesteps <= 0:
        raise ValueError("reference_total_timesteps must be > 0")
    if not stages:
        return []

    scaled_stages: list[tuple] = []
    prev_start = 0
    for stage in stages:
        if len(stage) == 0:
            raise ValueError("stages cannot contain empty tuples")

        base_start = int(stage[0])
        scaled_start = int(round((base_start / reference_total_timesteps) * total_timesteps))
        scaled_start = min(max(scaled_start, 0), total_timesteps)
        scaled_start = max(scaled_start, prev_start)
        prev_start = scaled_start
        scaled_stages.append((scaled_start, *stage[1:]))

    scaled_stages[0] = (0, *scaled_stages[0][1:])
    return scaled_stages


class ReorientCurriculumCallback(BaseCallback):
    """Gradually increases the target rotation difficulty during training."""

    def __init__(self, stages: list[tuple[int, float]], verbose: int = 0) -> None:
        """Set up curriculum stages.

        @param stages: list of (timestep, max_angle_rad) — when to increase difficulty
        @type stages: list[tuple[int, float]]
        @param verbose: print transitions if > 0
        @type verbose: int
        """

        super().__init__(verbose)
        self.stages = stages
        self._current_stage = 0

    def _on_training_start(self) -> None:
        """Apply stage-0 settings at the beginning of training."""

        if not self.stages:
            return

        max_angle = self.stages[0][1]
        self.training_env.env_method("set_curriculum_stage", max_angle)

        if self.verbose:
            print(f"[Curriculum] Stage 0: max_angle={max_angle:.2f} rad at step 0")

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

        @param stages: list of (timestep, clearance, peg_pre_grasped) tuples
        @type stages: list[tuple[int, float, bool]]
        @param verbose: print transitions if > 0
        @type verbose: int
        """

        super().__init__(verbose)
        self.stages = stages
        self._current_stage = 0

    def _on_training_start(self) -> None:
        """Apply stage-0 settings at the beginning of training."""

        if not self.stages:
            return

        clearance = self.stages[0][1]
        pre_grasped = self.stages[0][2]
        self.training_env.env_method("set_curriculum_params", clearance, pre_grasped)

        if self.verbose:
            print(
                f"[Curriculum] Stage 0: clearance={clearance * 1000:.1f}mm, "
                f"pre_grasped={pre_grasped} at step 0"
            )

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
