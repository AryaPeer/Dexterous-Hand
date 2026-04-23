
from dataclasses import dataclass, field
import math

@dataclass
class SceneConfig:

    mount_x: float = -0.10
    mount_y: float = 0.0
    mount_height: float = 0.82
    table_height: float = 0.4
    table_half_size: float = 0.25
    object_mass: float = 0.1
    object_friction: tuple[float, float, float] = (1.0, 0.005, 0.001)
    action_smoothing_alpha: float = 0.2
    sim_timestep: float = 0.002
    frame_skip: int = 20

@dataclass
class RewardWeights:

    reaching: float = 1.0
    grasping: float = 3.0
    lifting: float = 3.0
    holding: float = 4.0
    drop: float = 1.0
    action: float = 0.0
    action_rate: float = 0.3
    idle: float = 1.0
                                                                           
                                                                            
                                                               
    upward: float = 0.0
    opposition: float = 1.0

@dataclass
class RewardConfig:

    weights: RewardWeights = field(default_factory=RewardWeights)
                           
    reach_tanh_k: float = 5.0
    lift_target: float = 0.1
    hold_velocity_threshold: float = 0.05
    hold_height_smoothness_k: float = 50.0
    hold_velocity_smoothness_k: float = 20.0
                                                                      
    fingertip_weights: tuple[float, float, float, float, float] = (2.5, 1.0, 1.0, 1.0, 1.0)
    drop_penalty: float = -10.0
    no_contact_idle_penalty: float = -0.08
    hold_bonus: float = 500.0
                                                                               
                                                                          
                                                                           
                                               
    idle_grace_steps: int = 3

@dataclass
class TrainConfig:

    n_envs: int = 256
    total_timesteps: int = 30_000_000
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)

@dataclass
class ReorientRewardWeights:

    angular_progress: float = 2.5
    orientation_tracking: float = 3.0
    orientation_success: float = 4.0
    cube_drop: float = 5.0
    velocity_penalty: float = 0.0
    fingertip_distance: float = 0.0
    action_penalty: float = 0.5
    action_rate_penalty: float = 0.5
    contact_bonus: float = 0.3
    no_contact: float = 0.2
    position_stability: float = 0.0

@dataclass
class ReorientRewardConfig:

    weights: ReorientRewardWeights = field(default_factory=ReorientRewardWeights)
    success_threshold: float = 0.2
    success_hold_steps: int = 25
    drop_penalty: float = -20.0
    drop_height_offset: float = 0.05
    contact_bonus: float = 0.5
    no_contact_penalty: float = -0.25
    min_contacts_for_rotation: int = 2
                                                                           
                                                                              
    angular_progress_clip: float = 0.2
    # k=5 makes exp(-k·ang_dist) collapse past ~1 rad (exp(-5)=0.007), so the
    # policy can't feel any orientation gradient once the curriculum opens up
    # beyond 30°. k=2 keeps the signal meaningful out to ~2 rad (exp(-4)=0.018
    # is still differentiable) so the 90° / 180° / π stages aren't reward-flat.
    orientation_success_k: float = 2.0
    tracking_k: float = 2.0

@dataclass
class ReorientSceneConfig:

    mount_height: float = 0.4
    cube_size: float = 0.02                                   
    cube_mass: float = 0.1
    cube_friction: tuple[float, float, float] = (1.0, 0.005, 0.001)
    action_smoothing_alpha: float = 0.4
    target_min_angle: float = 0.15
    sim_timestep: float = 0.002
    frame_skip: int = 20

@dataclass
class ReorientTrainConfig:

    n_envs: int = 256
    total_timesteps: int = 400_000_000
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 5
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # 2M sanity on main showed PPO converging to a stable passive-contact
    # optimum (explained_variance=0.967, angular_distance drifting up) —
    # policy needs more exploration to escape. 0.01 matches IsaacGymEnvs ShadowHand.
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    scene_config: ReorientSceneConfig = field(default_factory=ReorientSceneConfig)
    reward_config: ReorientRewardConfig = field(default_factory=ReorientRewardConfig)
    curriculum_reference_timesteps: int = 400_000_000
    curriculum_stages: list[tuple[int, float]] = field(
        default_factory=lambda: [
            (0, math.radians(30)),
            (20_000_000, math.radians(90)),
            (60_000_000, math.radians(180)),
            (120_000_000, math.pi),
        ]
    )

@dataclass
class PegRewardWeights:

    reach: float = 0.5
    grasp: float = 5.0
    lift: float = 6.0
                                                                           
                                                                          
                                                                      
    upward: float = 0.0
    opposition: float = 1.0
    align: float = 2.0
    depth: float = 3.0
    complete: float = 1.0
    force: float = 1.0
    drop: float = 1.0
    smoothness: float = 0.3
    action_magnitude: float = 0.0
    idle_stage0: float = 1.0

@dataclass
class PegRewardConfig:

    weights: PegRewardWeights = field(default_factory=PegRewardWeights)
    drop_penalty: float = -10.0
                                                                                      
                                                                                 
                                                                               
                                                                         
                                                                 
    complete_bonus: float = 600.0
    depth_reward_scale: float = 10.0
    force_threshold: float = 5.0                                                
    idle_stage0_penalty: float = -0.3
    min_contacts_for_align: int = 2
    lift_target: float = 0.1                                          
                                                                        
                                                                         
                             
    lateral_gate_k: float = 10.0
                                                                           
                                                                       
    idle_stage_cutoff: int = 3
                                                                          
                                                                   
    idle_grace_steps: int = 3
                                                                         
                                                                     
    success_threshold: float = 0.7
    peg_hold_steps: int = 10
    reach_tanh_k: float = 5.0
    fingertip_weights: tuple[float, float, float, float, float] = (2.5, 1.0, 1.0, 1.0, 1.0)

@dataclass
class PegSceneConfig:

    mount_x: float = -0.10
    mount_y: float = 0.0
    mount_height: float = 0.82
    table_height: float = 0.4
    table_half_size: float = 0.25
    clearance: float = 0.004                                    
    hole_depth: float = 0.06
    hole_offset: list[float] = field(default_factory=lambda: [0.0, 0.0])                         
    spawn_min_radius: float = 0.04
    peg_radius: float = 0.008
    peg_half_length: float = 0.03
    peg_mass: float = 0.02
    peg_friction: tuple[float, float, float] = (1.0, 0.005, 0.001)
    action_smoothing_alpha: float = 0.2
    sim_timestep: float = 0.002
    frame_skip: int = 20

@dataclass
class PegTrainConfig:

    n_envs: int = 32
    total_timesteps: int = 40_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000                      
    learning_starts: int = 10_000                                                     
    tau: float = 0.005                                  
    gamma: float = 0.99
    train_freq: int = 1                            
    gradient_steps: int = 8                                                                       
    ent_coef: str = "auto"                                             
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    scene_config: PegSceneConfig = field(default_factory=PegSceneConfig)
    reward_config: PegRewardConfig = field(default_factory=PegRewardConfig)
    curriculum_reference_timesteps: int = 40_000_000
    curriculum_stages: list[tuple[int, float, float]] = field(
        default_factory=lambda: [
            (0, 0.004, 1.0),                                                 
            (8_000_000, 0.004, 0.7),                                                    
            (16_000_000, 0.003, 0.5),                                             
            (24_000_000, 0.002, 0.3),                                    
            (32_000_000, 0.001, 0.2),                    
        ]
    )

@dataclass
class MjxGraspTrainConfig:

    num_envs: int = 2048
    total_timesteps: int = 30_000_000
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    max_episode_steps: int = 200
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)

@dataclass
class MjxReorientTrainConfig:

    num_envs: int = 2048
    total_timesteps: int = 400_000_000
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 5
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    max_episode_steps: int = 400
    scene_config: ReorientSceneConfig = field(default_factory=ReorientSceneConfig)
    reward_config: ReorientRewardConfig = field(default_factory=ReorientRewardConfig)
    curriculum_reference_timesteps: int = 400_000_000
    curriculum_stages: list[tuple[int, float]] = field(
        default_factory=lambda: [
            (0, math.radians(30)),
            (20_000_000, math.radians(90)),
            (60_000_000, math.radians(180)),
            (120_000_000, math.pi),
        ]
    )

def _mjx_peg_reward_config() -> PegRewardConfig:

    return PegRewardConfig()

@dataclass
class MjxPegTrainConfig:

    num_envs: int = 2048
    total_timesteps: int = 40_000_000
    learning_rate: float = 3e-4
    batch_size: int = 1024
    buffer_size: int = 1_000_000
    learning_starts: int = 50_000
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1




    gradient_steps: int = 32
    ent_coef: str = "auto"
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    max_episode_steps: int = 500
    scene_config: PegSceneConfig = field(default_factory=PegSceneConfig)
    reward_config: PegRewardConfig = field(default_factory=_mjx_peg_reward_config)
    curriculum_reference_timesteps: int = 40_000_000
    curriculum_stages: list[tuple[int, float, float]] = field(
        default_factory=lambda: [
            (0, 0.004, 1.0),
            (8_000_000, 0.004, 0.7),
            (16_000_000, 0.003, 0.5),
            (24_000_000, 0.002, 0.3),
            (32_000_000, 0.001, 0.2),
        ]
    )

