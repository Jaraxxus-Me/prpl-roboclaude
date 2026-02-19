"""Tests for table3d.py."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo

from kinder.envs.geom3d.table3d import (
    Table3DEnv,
)
from tests.conftest import MAKE_VIDEOS

# Flag to enable trajectory saving (can be controlled like MAKE_VIDEOS)
SAVE_TRAJECTORIES = MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = Table3DEnv(
        num_cubes=2, use_gui=False, render_mode="rgb_array", realistic_bg=False
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def test_base_table3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in base table3D env."""
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
