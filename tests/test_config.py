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
        assert w.reaching == 1.0
        assert w.grasping == 3.0
                                                                            
                                                   
        assert w.upward == 0.0
        assert w.opposition == 1.0
        assert w.action == 0.0

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
        assert isinstance(c.scene_config, SceneConfig)
        assert isinstance(c.reward_config, RewardConfig)

    def test_reorient_scene_config(self):
        c = ReorientSceneConfig()
        assert c.mount_height == 0.4
        assert c.cube_size == 0.02
        assert c.action_smoothing_alpha == 0.4
        assert c.target_min_angle == 0.15

    def test_reorient_reward_config(self):
        c = ReorientRewardConfig()
        assert c.success_threshold == 0.2
        assert c.success_hold_steps == 25
        assert c.no_contact_penalty == -0.25
        assert c.angular_progress_clip == 0.2

    def test_reorient_train_config(self):
        c = ReorientTrainConfig()
        assert c.gamma == 0.998
        assert c.ent_coef == 0.01
        assert isinstance(c.scene_config, ReorientSceneConfig)
        assert isinstance(c.reward_config, ReorientRewardConfig)
        assert c.curriculum_reference_timesteps == 400_000_000
        assert len(c.curriculum_stages) == 4

    def test_peg_scene_config(self):
        c = PegSceneConfig()
        assert c.mount_x == -0.10
        assert c.mount_y == 0.0
        assert c.mount_height == 0.82
        assert c.action_smoothing_alpha == 0.2
        assert c.spawn_min_radius == 0.04
        assert c.clearance == 0.004
        assert c.hole_depth == 0.06
        assert len(c.hole_offset) == 2
        assert c.peg_radius == 0.008
        assert c.peg_half_length == 0.03
        assert c.peg_mass == 0.02

    def test_peg_reward_config(self):
        c = PegRewardConfig()
                                                                              
                                                                            
                                                                             
                                                     
        assert c.complete_bonus == 600.0
        assert c.force_threshold == 5.0
        assert c.idle_stage0_penalty == -0.3
                                                                          
                                                                               
        assert c.weights.upward == 0.0
        assert c.weights.opposition == 1.0
        assert c.lateral_gate_k == 10.0
        assert c.peg_hold_steps == 10
        assert c.success_threshold == 0.7

    def test_peg_train_config(self):
        c = PegTrainConfig()
        assert c.ent_coef == "auto"
        assert c.n_envs == 32
        assert c.gradient_steps == 8
        assert c.total_timesteps == 40_000_000
        assert isinstance(c.scene_config, PegSceneConfig)
        assert isinstance(c.reward_config, PegRewardConfig)
        assert c.curriculum_reference_timesteps == 40_000_000
        assert len(c.curriculum_stages) == 5
                                                                              
        for stage in c.curriculum_stages:
            assert len(stage) == 3
            step, clearance, p = stage
            assert 0.0 <= p <= 1.0

    def test_tactile_config(self):
        c = TactileConfig()
        assert c.grid_size == 4
        assert c.n_fingers == 5
        assert c.n_fingers * c.grid_size**2 == 80

    def test_tactile_train_config(self):
        c = TactileTrainConfig()
        assert isinstance(c.scene_config, PegSceneConfig)
        assert isinstance(c.reward_config, PegRewardConfig)
        assert c.curriculum_reference_timesteps == 40_000_000
        assert len(c.curriculum_stages) == 5
        assert isinstance(c.tactile_config, TactileConfig)
        for stage in c.curriculum_stages:
            assert len(stage) == 3

    def test_all_configs_instantiate(self):

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
