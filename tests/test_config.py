from dexterous_hand.config import (
    PegRewardConfig,
    PegRewardWeights,
    PegSceneConfig,
    PegTrainConfig,
    ReorientRewardConfig,
    ReorientRewardWeights,
    ReorientSceneConfig,
    ReorientTrainConfig,
    RewardConfig,
    RewardWeights,
    SceneConfig,
    TactileConfig,
    TactileTrainConfig,
    TrainConfig,
)


class TestConfigDefaults:
    def test_scene_config(self):
        c = SceneConfig()
        assert c.mount_x == -0.10
        assert c.mount_y == 0.0
        assert c.mount_height == 0.82
        assert c.sim_timestep == 0.002
        assert c.frame_skip == 20

    def test_reward_weights(self):
        w = RewardWeights()
        assert w.reaching == 0.4
        assert w.grasping == 2.5

    def test_reward_config(self):
        c = RewardConfig()
        assert isinstance(c.weights, RewardWeights)
        assert c.lift_target == 0.1
        assert c.no_contact_idle_penalty == -0.08

    def test_train_config(self):
        c = TrainConfig()
        assert c.n_envs == 256
        assert c.seed == 42
        assert len(c.net_arch) == 3

    def test_reorient_scene_config(self):
        c = ReorientSceneConfig()
        assert c.mount_height == 0.4
        assert c.cube_size == 0.02
        assert c.action_smoothing_alpha == 0.4
        assert c.target_min_angle == 0.15

    def test_reorient_reward_config(self):
        c = ReorientRewardConfig()
        assert c.success_threshold == 0.1
        assert c.success_hold_steps == 25
        assert c.no_contact_penalty == -0.25

    def test_reorient_train_config(self):
        c = ReorientTrainConfig()
        assert c.gamma == 0.998
        assert c.ent_coef == 0.002

    def test_peg_scene_config(self):
        c = PegSceneConfig()
        assert c.mount_x == -0.10
        assert c.mount_y == 0.0
        assert c.mount_height == 0.82
        assert c.action_smoothing_alpha == 0.2
        assert c.spawn_min_radius == 0.04
        assert c.clearance == 0.004
        assert c.hole_depth == 0.05
        assert len(c.hole_offset) == 2

    def test_peg_reward_config(self):
        c = PegRewardConfig()
        assert c.complete_bonus == 50.0
        assert c.force_threshold == 5.0
        assert c.idle_stage0_penalty == -0.1

    def test_peg_train_config(self):
        c = PegTrainConfig()
        assert c.ent_coef == "auto"
        assert len(c.curriculum_stages) == 4

    def test_tactile_config(self):
        c = TactileConfig()
        assert c.grid_size == 4
        assert c.n_fingers == 5
        assert c.n_fingers * c.grid_size**2 == 80

    def test_tactile_train_config(self):
        c = TactileTrainConfig()
        assert c.use_tactile is True
        assert isinstance(c.tactile_config, TactileConfig)

    def test_all_configs_instantiate(self):
        """Every config class can be created with just defaults."""

        configs = [
            SceneConfig,
            RewardWeights,
            RewardConfig,
            TrainConfig,
            ReorientSceneConfig,
            ReorientRewardWeights,
            ReorientRewardConfig,
            ReorientTrainConfig,
            PegSceneConfig,
            PegRewardWeights,
            PegRewardConfig,
            PegTrainConfig,
            TactileConfig,
            TactileTrainConfig,
        ]
        for cls in configs:
            obj = cls()
            assert obj is not None
