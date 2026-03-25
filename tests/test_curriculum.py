from unittest.mock import MagicMock

from dexterous_hand.curriculum.callbacks import (
    AssemblyCurriculumCallback,
    ReorientCurriculumCallback,
)


def _setup_callback(cb: ReorientCurriculumCallback | AssemblyCurriculumCallback) -> MagicMock:
    """Hook up a mock training env so we can test the callback logic."""
    mock_env = MagicMock()
    cb.locals = {}
    cb.globals = {}
    cb.model = MagicMock()
    cb.model.get_env.return_value = mock_env
    return mock_env


class TestReorientCurriculumCallback:
    def test_no_transition_at_start(self) -> None:
        stages = [(0, 0.5), (20_000_000, 1.57), (60_000_000, 3.14)]
        cb = ReorientCurriculumCallback(stages)
        mock_env = _setup_callback(cb)
        cb.num_timesteps = 0
        cb._on_step()
        mock_env.env_method.assert_not_called()

    def test_transition_at_threshold(self) -> None:
        stages = [(0, 0.5), (100, 1.57)]
        cb = ReorientCurriculumCallback(stages)
        mock_env = _setup_callback(cb)
        cb.num_timesteps = 100
        cb._on_step()
        mock_env.env_method.assert_called_once_with("set_curriculum_stage", 1.57)

    def test_multiple_transitions(self) -> None:
        stages = [(0, 0.5), (100, 1.57), (200, 3.14)]
        cb = ReorientCurriculumCallback(stages)
        _setup_callback(cb)
        cb.num_timesteps = 250
        cb._on_step()
        assert cb._current_stage == 2


class TestAssemblyCurriculumCallback:
    def test_no_transition_at_start(self) -> None:
        stages = [(0, 0.004, True), (25_000_000, 0.004, False)]
        cb = AssemblyCurriculumCallback(stages)
        mock_env = _setup_callback(cb)
        cb.num_timesteps = 0
        cb._on_step()
        mock_env.env_method.assert_not_called()

    def test_transition(self) -> None:
        stages = [(0, 0.004, True), (100, 0.002, False)]
        cb = AssemblyCurriculumCallback(stages)
        mock_env = _setup_callback(cb)
        cb.num_timesteps = 100
        cb._on_step()
        mock_env.env_method.assert_called_once_with("set_curriculum_params", 0.002, False)

    def test_returns_true(self) -> None:
        stages = [(0, 0.004, True)]
        cb = AssemblyCurriculumCallback(stages)
        _setup_callback(cb)
        cb.num_timesteps = 0
        assert cb._on_step() is True
