from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np

from dexterous_hand.config import SceneConfig
from dexterous_hand.utils.mujoco_helpers import get_joint_qpos_qvel_range

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "shadow_hand"

OBJECT_TYPES: dict[str, tuple[int, list[float]]] = {
    "large_cube": (mujoco.mjtGeom.mjGEOM_BOX, [0.035, 0.035, 0.035]),
    "cylinder": (mujoco.mjtGeom.mjGEOM_CYLINDER, [0.02, 0.04, 0.0]),
    "sphere": (mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0.0, 0.0]),
}

FINGERTIP_BODIES = [
    "rh_ffdistal",  # index
    "rh_mfdistal",  # middle
    "rh_rfdistal",  # ring
    "rh_lfdistal",  # little
    "rh_thdistal",  # thumb
]

FINGERTIP_SITE_NAMES = ["fftip", "mftip", "rftip", "lftip", "thtip"]

FINGERTIP_OFFSETS: dict[str, list[float]] = {
    "rh_ffdistal": [0.0, 0.0, 0.026],
    "rh_mfdistal": [0.0, 0.0, 0.026],
    "rh_rfdistal": [0.0, 0.0, 0.026],
    "rh_lfdistal": [0.0, 0.0, 0.026],
    "rh_thdistal": [0.0, 0.0, 0.032],
}

FINGER_BODY_PREFIXES = ["rh_ff", "rh_mf", "rh_rf", "rh_lf", "rh_th"]

TABLE_TASK_FLEXION_BIAS: dict[str, float] = {
    "rh_FFJ3": 1.2, "rh_MFJ3": 1.2, "rh_RFJ3": 1.2, "rh_LFJ3": 1.2,
    "rh_FFJ2": 1.0, "rh_MFJ2": 1.0, "rh_RFJ2": 1.0, "rh_LFJ2": 1.0,
    "rh_THJ4": 1.2, "rh_THJ1": 1.0,
}


@dataclass
class NameMap:
    # MuJoCo IDs resolved once at startup
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
    finger_geom_ids_per_finger: list[set[int]]
    object_body_id: int
    object_geom_id: int
    obj_qpos_start: int
    obj_qvel_start: int
    table_geom_id: int


def build_scene(
    config: SceneConfig | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, NameMap]:
    """Build the grasping scene: hand pointing down, table with object.

    @param config: scene settings (defaults if None)
    @type config: SceneConfig | None
    @return: (model, data, name_map)
    @rtype: tuple[mujoco.MjModel, mujoco.MjData, NameMap]
    """

    if config is None:
        config = SceneConfig()

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

    slider = spec.worldbody.add_body(
        name="hand_slider",
        pos=[config.mount_x, config.mount_y, config.mount_height],
    )
    slider.add_joint(
        name="slide_x", type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1, 0, 0], range=[-0.15, 0.15],
    )
    slider.add_joint(
        name="slide_y", type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 1, 0], range=[-0.15, 0.15],
    )

    mount = slider.add_body(
        name="hand_mount",
        euler=[math.pi, 0.0, 0.0],
    )
    mount_site = mount.add_site(
        name="hand_attach",
        pos=[0.0, 0.0, 0.0],
    )

    spec.add_actuator(
        name="slide_x_act", target="slide_x",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0, -100, -10, 0, 0, 0, 0, 0, 0, 0],
        ctrlrange=[-0.15, 0.15],
    )
    spec.add_actuator(
        name="slide_y_act", target="slide_y",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0, -100, -10, 0, 0, 0, 0, 0, 0, 0],
        ctrlrange=[-0.15, 0.15],
    )

    hand_xml = str(ASSETS_DIR / "right_hand.xml")
    child_spec = mujoco.MjSpec.from_file(hand_xml)
    spec.attach(child_spec, site=mount_site, prefix="")
    spec.body("rh_forearm").quat = [0.0, 1.0, 0.0, 0.0]

    for body_name, site_name in zip(FINGERTIP_BODIES, FINGERTIP_SITE_NAMES, strict=True):
        body = spec.body(body_name)
        offset = FINGERTIP_OFFSETS[body_name]
        body.add_site(
            name=site_name,
            pos=offset,
            size=[0.005],
            rgba=[1.0, 0.0, 0.0, 1.0],
        )

    default_type, default_size = OBJECT_TYPES["large_cube"]
    obj_body = spec.worldbody.add_body(
        name="object",
        pos=[0.0, 0.0, config.table_height + default_size[2]],
    )
    obj_body.add_freejoint(name="object_freejoint")
    obj_body.add_geom(
        name="object_geom",
        type=default_type,
        size=default_size,
        mass=config.object_mass,
        friction=list(config.object_friction),
        rgba=[0.2, 0.6, 0.9, 1.0],
        contype=1,
        conaffinity=1,
        condim=4,
    )

    model = spec.compile()
    data = mujoco.MjData(model)
    name_map = _resolve_names(model, spec)

    return model, data, name_map


def _resolve_names(model: mujoco.MjModel, spec: mujoco.MjSpec) -> NameMap:
    """Resolve MuJoCo IDs into a NameMap.

    @param model: compiled MuJoCo model
    @type model: mujoco.MjModel
    @param spec: spec used to build the model
    @type spec: mujoco.MjSpec
    """

    # object joint
    obj_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    obj_qpos_start = model.jnt_qposadr[obj_jnt_id]
    obj_qvel_start = model.jnt_dofadr[obj_jnt_id]

    # hand joints
    hand_joint_ids = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name != "object_freejoint":
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

    finger_geom_ids_per_finger: list[set[int]] = [set() for _ in FINGER_BODY_PREFIXES]
    for gid in range(model.ngeom):
        body_id = model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue

        for finger_idx, prefix in enumerate(FINGER_BODY_PREFIXES):
            if body_name.startswith(prefix):
                finger_geom_ids_per_finger[finger_idx].add(gid)
                break

    # object + table
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")

    return NameMap(
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
        finger_geom_ids_per_finger=finger_geom_ids_per_finger,
        object_body_id=object_body_id,
        object_geom_id=object_geom_id,
        obj_qpos_start=obj_qpos_start,
        obj_qvel_start=obj_qvel_start,
        table_geom_id=table_geom_id,
    )


def get_object_half_height(geom_type: int, geom_size: list[float]) -> float:
    """Half-height of a geom for table placement.

    @param geom_type: MuJoCo geom type (box, sphere, cylinder, etc.)
    @type geom_type: int
    @param geom_size: size params [x, y, z]
    @type geom_size: list[float]
    @return: half-height value
    @rtype: float
    """

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return geom_size[2]
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return geom_size[0]
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return geom_size[1]  # half-length
    else:
        return 0.03  # safe default if we don't recognize the type
