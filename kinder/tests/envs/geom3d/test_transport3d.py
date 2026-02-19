"""Tests for transport3d.py."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, SE2Pose
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder.envs.geom3d.transport3d import (
    ObjectCentricTransport3DEnv,
    Transport3DEnv,
    Transport3DObjectCentricState,
)
from kinder.envs.geom3d.utils import extend_joints_to_include_fingers
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = Transport3DEnv(
        num_cubes=2,
        num_boxes=1,
        use_gui=False,
        render_mode="rgb_array",
        realistic_bg=False,
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def test_base_transport3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in base transport3d env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p

    # while True:
    #     # p.getMouseEvents(env.unwrapped._object_centric_env.physics_client_id)
    #     p.stepSimulation(env.unwrapped._object_centric_env.physics_client_id)


def _execute_joint_plan(environment, joint_plan, obs):
    """Execute a joint space plan and return the final observation."""
    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = Transport3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    return obs