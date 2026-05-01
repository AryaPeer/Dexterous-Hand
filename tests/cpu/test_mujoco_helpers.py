import mujoco
import numpy as np
import pytest

from dexterous_hand.envs.scene_builder import build_scene
from dexterous_hand.utils.cpu.mujoco_helpers import (
    get_body_axis,
    get_fingertip_positions,
    get_object_state,
    get_palm_position,
)


@pytest.fixture(scope="module")
def grasp_scene():
    model, data, nm = build_scene()
    mujoco.mj_forward(model, data)
    return model, data, nm

@pytest.mark.slow
class TestGetFingertipPositions:
    def test_shape(self, grasp_scene):
        _, data, nm = grasp_scene
        pos = get_fingertip_positions(data, nm.fingertip_site_ids)
        assert pos.shape == (5, 3)

    def test_finite(self, grasp_scene):
        _, data, nm = grasp_scene
        pos = get_fingertip_positions(data, nm.fingertip_site_ids)
        assert np.all(np.isfinite(pos))

@pytest.mark.slow
class TestGetObjectState:
    def test_shapes(self, grasp_scene):
        _, data, nm = grasp_scene
        pos, quat, linvel, angvel = get_object_state(
            data, nm.object_body_id, nm.obj_qpos_start, nm.obj_qvel_start
        )
        assert pos.shape == (3,)
        assert quat.shape == (4,)
        assert linvel.shape == (3,)
        assert angvel.shape == (3,)

    def test_quat_unit(self, grasp_scene):
        _, data, nm = grasp_scene
        _, quat, _, _ = get_object_state(
            data, nm.object_body_id, nm.obj_qpos_start, nm.obj_qvel_start
        )
        np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-6)

@pytest.mark.slow
class TestGetPalmPosition:
    def test_shape(self, grasp_scene):
        _, data, nm = grasp_scene
        pos = get_palm_position(data, nm.palm_body_id)
        assert pos.shape == (3,)

    def test_finite(self, grasp_scene):
        _, data, nm = grasp_scene
        pos = get_palm_position(data, nm.palm_body_id)
        assert np.all(np.isfinite(pos))

@pytest.mark.slow
class TestGetBodyAxis:
    def test_unit_vector(self, grasp_scene):
        _, data, nm = grasp_scene
        axis = get_body_axis(data, nm.palm_body_id, axis=2)
        np.testing.assert_allclose(np.linalg.norm(axis), 1.0, atol=1e-6)

    def test_shape(self, grasp_scene):
        _, data, nm = grasp_scene
        axis = get_body_axis(data, nm.palm_body_id)
        assert axis.shape == (3,)
