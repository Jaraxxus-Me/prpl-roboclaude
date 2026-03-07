"""Tests for pushpullhook2d_grasp_vertical.py."""

import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

import kinder
from kinder.envs.geom2d.pushpullhook2d_grasp_vertical import (
    ObjectCentricPushPullHook2DGraspVerticalEnv,
)
from tests.conftest import MAKE_VIDEOS


def test_object_centric_pushpullhook2d_grasp_vertical_env():
    """Tests for ObjectCentricPushPullHook2DGraspVerticalEnv()."""
    env = ObjectCentricPushPullHook2DGraspVerticalEnv()
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")
    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


def test_pushpullhook2d_grasp_vertical_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/PushPullHook2D-GraspVertical-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_pushpullhook2d_grasp_vertical_action_space():
    """Tests that actions are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/PushPullHook2D-GraspVertical-v0")
    assert isinstance(env.action_space, Box)
    for _ in range(5):
        action = env.action_space.sample()
        assert env.action_space.contains(action)


def test_hook_always_vertical():
    """Verify hook theta = pi/2 across multiple seeds."""
    env = ObjectCentricPushPullHook2DGraspVerticalEnv()
    for seed in range(20):
        state, _ = env.reset(seed=seed)
        hook = next(o for o in state if o.name == "hook")
        assert np.isclose(state.get(hook, "theta"), np.pi / 2, atol=1e-6), (
            f"seed={seed}: hook theta={state.get(hook, 'theta')}"
        )
    env.close()
