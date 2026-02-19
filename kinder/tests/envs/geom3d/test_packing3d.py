"""Tests for packing3d.py."""

import numpy as np
from relational_structs import Object

from kinder.envs.geom3d.packing3d import (
    Packing3DEnv,
    Packing3DObjectCentricState,
)
from tests.conftest import MAKE_VIDEOS

# Flag to enable trajectory saving (can be controlled like MAKE_VIDEOS)
SAVE_TRAJECTORIES = MAKE_VIDEOS


def test_packing3d_env_basic():
    """Basic smoke test for the packing3d environment."""

    for num_parts in [1, 2, 3]:
        env = Packing3DEnv(
            num_parts=num_parts, use_gui=False, realistic_bg=False
        )  # set use_gui=False to debug
        obs, _ = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)

        for _ in range(10):
            act = env.action_space.sample()
            assert isinstance(obs, np.ndarray)
            obs, _, _, _, _ = env.step(act)

        env.close()


def get_target_object_from_obs(
    obs: Packing3DObjectCentricState,
) -> Object | None:
    """Get the target object from the observation."""
    available_parts = obs.available_parts
    if not available_parts:
        return None
    # For simplicity, just choose the first available part.
    target_part_name = available_parts[0]
    return obs.get_object_from_name(target_part_name)