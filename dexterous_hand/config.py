from dataclasses import dataclass, field


@dataclass
class SceneConfig:
    mount_height: float = 0.65
    table_height: float = 0.4
    table_half_size: float = 0.25
    object_mass: float = 0.1
    object_friction: tuple[float, float, float] = (1.0, 0.005, 0.001)
    sim_timestep: float = 0.002
    frame_skip: int = 20  # 500Hz sim / 20 = 25Hz control loop


@dataclass
class RewardWeights:
    reaching: float = 1.0
    grasping: float = 2.0
    lifting: float = 5.0
    holding: float = 3.0
    drop: float = 1.0
    action: float = 1.0
    action_rate: float = 1.0


@dataclass
class RewardConfig:
    weights: RewardWeights = field(default_factory=RewardWeights)
    lift_target: float = 0.1  # target height above table (10cm)
    hold_velocity_threshold: float = 0.05
    drop_penalty: float = -10.0


@dataclass
class TrainConfig:
    n_envs: int = 2048
    total_timesteps: int = 50_000_000
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


@dataclass
class ReorientRewardWeights:
    orientation_tracking: float = 5.0
    orientation_success: float = 1.0
    cube_drop: float = 1.0
    velocity_penalty: float = 1.0
    fingertip_distance: float = 1.0
    position_penalty: float = 1.0
    action_penalty: float = 1.0
    action_rate_penalty: float = 1.0


@dataclass
class ReorientRewardConfig:
    weights: ReorientRewardWeights = field(default_factory=ReorientRewardWeights)
    success_threshold: float = 0.1  # radians
    success_hold_steps: int = 25
    drop_penalty: float = -20.0
    drop_height_offset: float = 0.05  # how far below palm counts as "dropped"


@dataclass
class ReorientSceneConfig:
    mount_height: float = 0.4
    cube_size: float = 0.02  # half-size (so full cube is 4cm)
    cube_mass: float = 0.1
    cube_friction: tuple[float, float, float] = (1.0, 0.005, 0.001)
    sim_timestep: float = 0.002
    frame_skip: int = 20


@dataclass
class ReorientTrainConfig:
    n_envs: int = 4096
    total_timesteps: int = 200_000_000
    learning_rate: float = 3e-4
    batch_size: int = 8192
    n_steps_per_env: int = 64
    n_epochs: int = 5
    gamma: float = 0.998  # higher gamma bc reorientation needs longer horizon
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True


@dataclass
class PegRewardWeights:
    reach: float = 1.0
    grasp: float = 2.0
    lift: float = 2.0
    align: float = 5.0
    depth: float = 1.0
    complete: float = 1.0
    force: float = 1.0
    drop: float = 1.0
    smoothness: float = 1.0


@dataclass
class PegRewardConfig:
    weights: PegRewardWeights = field(default_factory=PegRewardWeights)
    peg_radius: float = 0.008
    peg_half_length: float = 0.03
    peg_mass: float = 0.02
    drop_penalty: float = -10.0
    complete_bonus: float = 50.0
    force_threshold: float = 5.0  # penalize contact forces above this (Newtons)


@dataclass
class PegSceneConfig:
    mount_height: float = 0.65
    table_height: float = 0.4
    table_half_size: float = 0.25
    clearance: float = 0.004  # initial hole clearance in meters
    hole_depth: float = 0.05
    hole_offset: list[float] = field(default_factory=lambda: [0.1, 0.0])  # XY offset from center
    peg_friction: tuple[float, float, float] = (1.0, 0.005, 0.001)
    sim_timestep: float = 0.002
    frame_skip: int = 20


@dataclass
class PegTrainConfig:
    """SAC config for peg-in-hole training."""

    n_envs: int = 1024
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000  # replay buffer size
    learning_starts: int = 10_000  # random exploration steps before we start learning
    tau: float = 0.005  # soft target update coefficient
    gamma: float = 0.99
    train_freq: int = 1  # update policy every step
    gradient_steps: int = 1
    ent_coef: str = "auto"  # let SAC auto-tune the entropy coefficient
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    curriculum_stages: list[tuple[int, float, bool]] = field(
        default_factory=lambda: [
            (0, 0.004, True),  # start easy: 4mm clearance, peg already in hand
            (25_000_000, 0.004, False),  # now it has to pick up the peg itself
            (50_000_000, 0.002, False),  # tighter hole (2mm)
            (75_000_000, 0.001, False),  # final difficulty (1mm clearance)
        ]
    )


@dataclass
class TactileConfig:
    grid_size: int = 4  # 4x4 taxels per finger
    n_fingers: int = 5
    noise_std: float = 0.1  # gaussian noise in Newtons (simulates real sensor noise)
    max_force: float = 10.0  # max reading before clipping
    grid_spacing: float = 0.002  # 2mm spacing between taxels


@dataclass
class TactileTrainConfig:
    """SAC config for tactile ablation study (can toggle tactile on/off)."""

    n_envs: int = 1024
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    use_tactile: bool = True  # flip to False for the no-tactile baseline
    tactile_config: TactileConfig = field(default_factory=TactileConfig)
