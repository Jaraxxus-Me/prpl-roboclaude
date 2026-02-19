"""Tests for base_motion3d.py."""

from unittest.mock import patch

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo

import kinder
from kinder.envs.geom3d.base_motion3d import (
    BaseMotion3DEnv,
    BaseMotion3DObjectCentricState,
)
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = BaseMotion3DEnv(
        render_mode="rgb_array", use_gui=False, realistic_bg=False
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def test_base_motion3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in base motion3D env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.unwrapped._object_centric_env.physics_client_id)


def test_check_mobile_base_collisions_is_called(
    env,
):  # pylint: disable=redefined-outer-name
    """Test that check_mobile_base_collisions is called when there is a collision."""
    env.reset(seed=123)

    # Patch the check_mobile_base_collisions function
    with patch(
        "kinder.envs.geom3d.base_env.check_mobile_base_collisions"
    ) as mock_check_base:
        # Set return value to False (no collision)
        mock_check_base.return_value = False

        # Take an action that moves the base (first 3 elements are base actions)
        action = np.array([0.01, 0.01, 0.0] + [0.0] * 7 + [0.0], dtype=np.float32)
        env.step(action)

        # Verify that check_mobile_base_collisions was called
        assert mock_check_base.called, "check_mobile_base_collisions should be called"

        # Verify it was called with the correct arguments
        assert mock_check_base.call_count >= 1
        call_args = mock_check_base.call_args
        assert call_args is not None

        # Verify the robot base was passed as the first argument
        assert (
            call_args[0][0]
            == env.unwrapped._object_centric_env.robot.base  # pylint: disable=protected-access
        )


def test_reset_with_init_state():
    """Test that reset accepts init_state via options and restores state correctly."""
    kinder.register_all_environments()
    env = kinder.make(  # pylint: disable=redefined-outer-name
        "kinder/BaseMotion3D-v0", render_mode="rgb_array", realistic_bg=False
    )

    # Get initial observation after reset.
    vec_obs, _ = env.reset(seed=123)
    initial_oc_obs = env.observation_space.devectorize(vec_obs)
    initial_state = BaseMotion3DObjectCentricState(
        initial_oc_obs.data, initial_oc_obs.type_features
    )

    # Take some actions to change state.
    for _ in range(5):
        action = np.array([0.05, 0.02, 0.01] + [0.0] * 7 + [0.0], dtype=np.float32)
        env.step(action)

    # Verify state has changed.
    changed_vec_obs = (
        env.unwrapped._object_centric_env._get_obs()  # pylint: disable=protected-access
    )
    changed_state = BaseMotion3DObjectCentricState(
        changed_vec_obs.data, changed_vec_obs.type_features
    )
    assert not np.allclose(
        initial_state.base_pose.x, changed_state.base_pose.x, atol=1e-3
    ), "State should have changed after taking actions"

    # Reset with init_state to restore the original state.
    restored_vec_obs, _ = env.reset(options={"init_state": initial_state})
    restored_oc_obs = env.observation_space.devectorize(restored_vec_obs)
    restored_state = BaseMotion3DObjectCentricState(
        restored_oc_obs.data, restored_oc_obs.type_features
    )

    # Verify the state matches the initial state.
    assert np.allclose(
        initial_state.base_pose.x, restored_state.base_pose.x, atol=1e-6
    ), "Base pose x should match after reset with init_state"
    assert np.allclose(
        initial_state.base_pose.y, restored_state.base_pose.y, atol=1e-6
    ), "Base pose y should match after reset with init_state"
    assert np.allclose(
        initial_state.target_base_pose.x,
        restored_state.target_base_pose.x,
        atol=1e-6,
    ), "Target pose x should match after reset with init_state"
    assert np.allclose(
        initial_state.target_base_pose.y,
        restored_state.target_base_pose.y,
        atol=1e-6,
    ), "Target pose y should match after reset with init_state"

    env.close()
