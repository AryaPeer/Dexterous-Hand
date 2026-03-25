from dataclasses import dataclass, field
import math

import mujoco
import numpy as np

from dexterous_hand.config import PegSceneConfig
from dexterous_hand.envs.scene_builder import (
    ASSETS_DIR,
    FINGERTIP_BODIES,
    FINGERTIP_OFFSETS,
    FINGERTIP_SITE_NAMES,
)
from dexterous_hand.utils.mujoco_helpers import get_joint_qpos_qvel_range


@dataclass
class PegNameMap:
    """MuJoCo IDs for the peg-in-hole scene."""

    hand_joint_ids: list[int]
    hand_actuator_ids: list[int]
    hand_qpos_start: int
    hand_qpos_end: int
    hand_qvel_start: int
    hand_qvel_end: int
    n_actuators: int
    ctrl_ranges: np.ndarray  # (n_actuators, 2)

    palm_body_id: int
    fingertip_site_ids: list[int]
    fingertip_geom_ids: set[int]
    table_geom_id: int

    peg_body_id: int
    peg_geom_id: int
    peg_qpos_start: int
    peg_qvel_start: int
    hole_body_id: int
    hole_wall_geom_ids: list[int] = field(default_factory=list)
    hole_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hole_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))


def build_peg_scene(
    config: PegSceneConfig | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, PegNameMap]:
    """Build peg-in-hole scene: hand, table, peg, and hole with walls.

    @param config: scene settings (defaults if None)
    @type config: PegSceneConfig | None
    @return: (model, data, name_map)
    @rtype: tuple[mujoco.MjModel, mujoco.MjData, PegNameMap]
    """

    if config is None:
        config = PegSceneConfig()

    spec = mujoco.MjSpec()
    spec.option.timestep = config.sim_timestep
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.stat.extent = 1.0
    spec.stat.center = [0.0, 0.0, config.table_height]

    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.01],
        rgba=[0.3, 0.3, 0.3, 1.0],
        conaffinity=1,
        condim=3,
    )

    table_half_h = 0.02
    table_body = spec.worldbody.add_body(
        name="table",
        pos=[0.0, 0.0, config.table_height - table_half_h],
    )
    table_body.add_geom(
        name="table_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[config.table_half_size, config.table_half_size, table_half_h],
        rgba=[0.5, 0.35, 0.2, 1.0],
        conaffinity=1,
        condim=3,
    )

    spec.worldbody.add_light(
        name="top_light",
        pos=[0.0, 0.0, 1.5],
        dir=[0.0, 0.0, -1.0],
        diffuse=[0.8, 0.8, 0.8],
        specular=[0.3, 0.3, 0.3],
    )
    spec.worldbody.add_camera(
        name="track_cam",
        pos=[0.8, -0.8, 0.8],
        xyaxes=[0.707, 0.707, 0.0, -0.354, 0.354, 0.866],
    )

    mount = spec.worldbody.add_body(
        name="hand_mount",
        pos=[0.0, 0.0, config.mount_height],
        euler=[math.pi, 0.0, 0.0],
    )
    mount_site = mount.add_site(name="hand_attach", pos=[0.0, 0.0, 0.0])

    hand_xml = str(ASSETS_DIR / "right_hand.xml")
    child_spec = mujoco.MjSpec.from_file(hand_xml)
    spec.attach(child_spec, site=mount_site, prefix="")

    for body_name, site_name in zip(FINGERTIP_BODIES, FINGERTIP_SITE_NAMES, strict=True):
        body = spec.body(body_name)
        offset = FINGERTIP_OFFSETS[body_name]
        body.add_site(
            name=site_name,
            pos=offset,
            size=[0.005],
            rgba=[1.0, 0.0, 0.0, 1.0],
        )

    peg_radius = 0.008
    peg_half_length = 0.03
    contact_kwargs = dict(
        contype=1,
        conaffinity=1,
        condim=4,
        margin=0.001,
        solref=[0.02, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
    )

    peg_z = config.table_height + peg_half_length + 0.001
    peg_body = spec.worldbody.add_body(
        name="peg",
        pos=[0.0, 0.0, peg_z],
    )
    peg_body.add_freejoint(name="peg_freejoint")
    peg_body.add_geom(
        name="peg_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[peg_radius, peg_half_length, 0.0],
        mass=0.02,
        friction=list(config.peg_friction),
        rgba=[0.8, 0.2, 0.2, 1.0],
        **contact_kwargs,
    )

    hole_x = config.hole_offset[0]
    hole_y = config.hole_offset[1]
    hole_z = config.table_height
    hole_body = spec.worldbody.add_body(
        name="hole",
        pos=[hole_x, hole_y, hole_z],
    )

    cr = peg_radius + config.clearance
    wt = 0.005
    wh = config.hole_depth / 2

    wall_rgba = [0.4, 0.4, 0.5, 1.0]

    hole_body.add_geom(
        name="hole_wall_px",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[wt / 2, cr + wt, wh],
        pos=[cr + wt / 2, 0.0, -wh],
        rgba=wall_rgba,
        **contact_kwargs,
    )

    hole_body.add_geom(
        name="hole_wall_nx",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[wt / 2, cr + wt, wh],
        pos=[-(cr + wt / 2), 0.0, -wh],
        rgba=wall_rgba,
        **contact_kwargs,
    )

    hole_body.add_geom(
        name="hole_wall_py",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cr + wt, wt / 2, wh],
        pos=[0.0, cr + wt / 2, -wh],
        rgba=wall_rgba,
        **contact_kwargs,
    )

    hole_body.add_geom(
        name="hole_wall_ny",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cr + wt, wt / 2, wh],
        pos=[0.0, -(cr + wt / 2), -wh],
        rgba=wall_rgba,
        **contact_kwargs,
    )

    hole_body.add_geom(
        name="hole_bottom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cr + wt, cr + wt, wt / 2],
        pos=[0.0, 0.0, -config.hole_depth],
        rgba=wall_rgba,
        **contact_kwargs,
    )

    model = spec.compile()
    data = mujoco.MjData(model)
    name_map = _resolve_peg_names(model, config)

    return model, data, name_map


def _resolve_peg_names(model: mujoco.MjModel, config: PegSceneConfig) -> PegNameMap:
    """Resolve MuJoCo IDs for the peg scene."""

    # peg joint
    peg_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
    peg_qpos_start = model.jnt_qposadr[peg_jnt_id]
    peg_qvel_start = model.jnt_dofadr[peg_jnt_id]

    # hand joints
    hand_joint_ids = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name != "peg_freejoint":
            hand_joint_ids.append(jid)

    if hand_joint_ids:
        hand_qpos_start, hand_qpos_end, hand_qvel_start, hand_qvel_end = (
            get_joint_qpos_qvel_range(model, hand_joint_ids)
        )
    else:
        hand_qpos_start = hand_qpos_end = 0
        hand_qvel_start = hand_qvel_end = 0

    # actuators
    hand_actuator_ids = list(range(model.nu))
    ctrl_ranges = model.actuator_ctrlrange[: model.nu].copy()
    n_actuators = model.nu

    palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")

    # fingertips
    fingertip_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FINGERTIP_SITE_NAMES
    ]

    fingertip_geom_ids: set[int] = set()
    fingertip_body_ids = set()

    for body_name in FINGERTIP_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            fingertip_body_ids.add(bid)

    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] in fingertip_body_ids:
            fingertip_geom_ids.add(gid)

    # peg + hole
    table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
    peg_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    peg_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
    hole_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hole")

    wall_names = ["hole_wall_px", "hole_wall_nx", "hole_wall_py", "hole_wall_ny", "hole_bottom"]
    hole_wall_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in wall_names]

    hole_pos = np.array([config.hole_offset[0], config.hole_offset[1], config.table_height])
    hole_quat = np.array([1.0, 0.0, 0.0, 0.0])

    return PegNameMap(
        hand_joint_ids=hand_joint_ids,
        hand_actuator_ids=hand_actuator_ids,
        hand_qpos_start=hand_qpos_start,
        hand_qpos_end=hand_qpos_end,
        hand_qvel_start=hand_qvel_start,
        hand_qvel_end=hand_qvel_end,
        n_actuators=n_actuators,
        ctrl_ranges=ctrl_ranges,
        palm_body_id=palm_body_id,
        fingertip_site_ids=fingertip_site_ids,
        fingertip_geom_ids=fingertip_geom_ids,
        table_geom_id=table_geom_id,
        peg_body_id=peg_body_id,
        peg_geom_id=peg_geom_id,
        peg_qpos_start=peg_qpos_start,
        peg_qvel_start=peg_qvel_start,
        hole_body_id=hole_body_id,
        hole_wall_geom_ids=hole_wall_geom_ids,
        hole_pos=hole_pos,
        hole_quat=hole_quat,
    )
