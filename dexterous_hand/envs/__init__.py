from gymnasium.envs.registration import register

register(
    id="ShadowHandGrasp-v0",
    entry_point="dexterous_hand.envs.cpu.grasp_env:ShadowHandGraspEnv",
    max_episode_steps=200,
)

register(
    id="ShadowHandReorient-v0",
    entry_point="dexterous_hand.envs.cpu.reorient_env:ShadowHandReorientEnv",
    max_episode_steps=400,
)

register(
    id="ShadowHandPeg-v0",
    entry_point="dexterous_hand.envs.cpu.peg_env:ShadowHandPegEnv",
    max_episode_steps=500,
)

register(
    id="ShadowHandPegTactile-v0",
    entry_point="dexterous_hand.envs.cpu.peg_tactile_env:ShadowHandPegTactileEnv",
    max_episode_steps=500,
)
