"""Tests for motion3d.py."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo

from kinder.envs.geom3d.motion3d import (
    Motion3DEnv,
)
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = Motion3DEnv(
        render_mode="rgb_array", use_gui=False, realistic_bg=False
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def test_motion3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in motion3D env."""
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
