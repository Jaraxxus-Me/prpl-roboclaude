"""Tests for obstruction3d.py."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics

from kinder.envs.geom3d.obstruction3d import (
    ObjectCentricObstruction3DEnv,
    Obstruction3DEnv,
    Obstruction3DEnvConfig,
)
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    config = Obstruction3DEnvConfig(target_block_height=0.01)
    environment = Obstruction3DEnv(
        num_obstructions=0,
        config=config,
        use_gui=False,
        render_mode="rgb_array",
        realistic_bg=False,
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def test_obstruction3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in obstruction3d env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(obs, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.unwrapped._object_centric_env.physics_client_id)


def test_grasp_fails_when_fingers_collide_with_table():
    """Test that grasping fails when fingers collide with table during grasp."""
    # Create environment with no obstructions.
    config = Obstruction3DEnvConfig(
        target_block_height=0.015, target_block_size_scale=0.5
    )
    oc_env = ObjectCentricObstruction3DEnv(
        num_obstructions=0, config=config, realistic_bg=False
    )

    obs, _ = oc_env.reset(seed=456)

    # Position the gripper very low (close to the table surface) around the block.
    x, y, _ = obs.target_block_pose.position
    # Position gripper very close to table surface - when fingers close, they'll
    # collide with the table.
    grasp_z = 0.11  # Just barely above table surface
    low_grasp_pose = Pose.from_rpy((x, y, grasp_z), (np.pi, 0, np.pi / 2))

    # Use IK to get joint positions for this pose, then directly set the state.
    target_joints = inverse_kinematics(
        oc_env._robot_arm,  # pylint: disable=protected-access
        low_grasp_pose,
        validate=False,
    )
    assert target_joints is not None

    # Directly set robot state to this configuration.
    oc_env.robot.arm.set_joints(target_joints)
    oc_env._robot_arm.open_fingers()  # pylint: disable=protected-access

    # Attempt to grasp. This should fail because the fingers will collide with
    # the table when they close.
    close_action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
    obs, _, _, _, _ = oc_env.step(close_action)

    # The grasp should have failed - grasped_object should be None.
    assert obs.grasped_object is None, "Grasp should have failed due to table collision"
