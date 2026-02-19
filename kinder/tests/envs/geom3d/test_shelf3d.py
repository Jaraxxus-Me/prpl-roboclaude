"""Tests for shelf3d.py."""

# pylint: disable=protected-access

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle

from kinder.envs.geom3d.shelf3d import (
    Shelf3DEnv,
    Shelf3DObjectCentricState,
)
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = Shelf3DEnv(
        num_cubes=2,
        use_gui=False,
        render_mode="rgb_array",
        realistic_bg=False,
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def _execute_base_plan(environment, base_plan, obs):
    """Execute a base motion plan and return the final observation."""
    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    return obs


def _execute_joint_plan(environment, joint_plan, obs):
    """Execute a joint space plan and return the final observation."""
    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    return obs


def test_shelf3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in shelf env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)


def test_camera_rendering(env):  # pylint: disable=redefined-outer-name
    """Test rendering from overview, base, and end-effector cameras."""
    env.reset(seed=123)

    # Get the object-centric env for direct camera access
    oc_env = env.unwrapped._object_centric_env
    config = oc_env.config

    # Test overview camera (default render)
    overview_image = oc_env.render()
    assert overview_image is not None
    assert overview_image.shape == (
        config.render_image_height,
        config.render_image_width,
        3,
    )
    assert overview_image.dtype == np.uint8

    # Test base camera
    base_image = oc_env.render_base_camera()
    assert base_image is not None
    assert base_image.shape == (
        config.base_camera_image_height,
        config.base_camera_image_width,
        3,
    )
    assert base_image.dtype == np.uint8

    # Test end-effector camera
    ee_image = oc_env.render_ee_camera()
    assert ee_image is not None
    assert ee_image.shape == (
        config.ee_camera_image_height,
        config.ee_camera_image_width,
        3,
    )
    assert ee_image.dtype == np.uint8

    # Test render_all_cameras
    all_images = oc_env.render_all_cameras()
    assert isinstance(all_images, dict)
    assert set(all_images.keys()) == {"overview", "base", "wrist"}
    assert all_images["overview"].shape == overview_image.shape
    assert all_images["base"].shape == base_image.shape
    assert all_images["wrist"].shape == ee_image.shape

    # Take a few steps and verify cameras still work (poses change)
    for _ in range(5):
        act = env.action_space.sample()
        env.step(act)

    # Verify cameras work after robot has moved
    all_images_after = oc_env.render_all_cameras()
    assert all_images_after["overview"].shape == overview_image.shape
    assert all_images_after["base"].shape == base_image.shape
    assert all_images_after["wrist"].shape == ee_image.shape
