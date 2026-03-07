"""Tests for pushpullhook2d_grasp_random.py."""

import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

import kinder
from kinder.envs.geom2d.pushpullhook2d_grasp_random import (
    ObjectCentricPushPullHook2DGraspRandomEnv,
)
from tests.conftest import MAKE_VIDEOS


def test_object_centric_pushpullhook2d_grasp_random_env():
    """Tests for ObjectCentricPushPullHook2DGraspRandomEnv()."""
    env = ObjectCentricPushPullHook2DGraspRandomEnv()
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")
    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


def test_pushpullhook2d_grasp_random_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/PushPullHook2D-GraspRandom-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_pushpullhook2d_grasp_random_action_space():
    """Tests that actions are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/PushPullHook2D-GraspRandom-v0")
    assert isinstance(env.action_space, Box)
    for _ in range(5):
        action = env.action_space.sample()
        assert env.action_space.contains(action)


def test_hook_theta_varies():
    """Verify hook theta varies across seeds and stays in [pi/4, 3*pi/4]."""
    env = ObjectCentricPushPullHook2DGraspRandomEnv()
    thetas = []
    for seed in range(20):
        state, _ = env.reset(seed=seed)
        hook = next(o for o in state if o.name == "hook")
        theta = state.get(hook, "theta")
        assert np.pi / 4 - 1e-6 <= theta <= 3 * np.pi / 4 + 1e-6, (
            f"seed={seed}: hook theta={theta} out of bounds"
        )
        thetas.append(theta)
    env.close()
    # Check that not all thetas are the same (i.e. orientation is random).
    assert len(set(round(t, 4) for t in thetas)) > 1, (
        "Hook theta should vary across seeds"
    )


def _goal_config_is_collision_free(env, state, robot, gx, gy, gtheta, arm):
    """Check if robot at (gx, gy, gtheta, arm) is collision-free."""
    from kinder.envs.utils import state_2d_has_collision

    test = state.copy()
    test.set(robot, "x", gx)
    test.set(robot, "y", gy)
    test.set(robot, "theta", gtheta)
    test.set(robot, "arm_joint", arm)
    test.set(robot, "vacuum", 0.0)
    full = test.copy()
    full.data.update(env.initial_constant_state.data)
    return not state_2d_has_collision(
        full, {robot}, set(full) - {robot}, env._static_object_body_cache
    )


def _solve_grasp(env, state, max_steps=500, step_env=None):
    """Solver: compute a collision-free grasp pose, navigate there with arm
    retracted, extend arm, then vacuum.

    1. Compute goal config where suction zone overlaps the hook bar,
       trying both perpendicular approach directions and picking the one
       that is collision-free with arm fully extended.
    2. Navigate to goal (x, y, theta) with arm retracted.
    3. Extend arm at goal.
    4. Turn on vacuum.
    """
    if step_env is None:
        step_env = env
    obj_map = {o.name: o for o in state}
    robot = obj_map["robot"]
    hook = obj_map["hook"]

    hx = state.get(hook, "x")
    hy = state.get(hook, "y")
    ht = state.get(hook, "theta")
    hl1 = state.get(hook, "length_side1")
    arm_length = state.get(robot, "arm_length")
    gripper_w = state.get(robot, "gripper_width")
    base_r = state.get(robot, "base_radius")

    # Bar direction and target point (75% along bar).
    bar_dir = np.array([-np.cos(ht), -np.sin(ht)])
    bar_pt = np.array([hx, hy]) + bar_dir * hl1 * 0.75

    # Suction zone offset from robot base center.
    suction_offset = arm_length + gripper_w + gripper_w / 2

    # Try both perpendicular approach directions; pick collision-free one.
    perp1 = np.array([-bar_dir[1], bar_dir[0]])
    perp2 = -perp1

    goal_x, goal_y, face_theta = None, None, None
    for bar_t in [0.75, 0.7, 0.6, 0.8, 0.5]:
        bp = np.array([hx, hy]) + bar_dir * hl1 * bar_t
        for perp in [perp1, perp2]:
            ft = np.arctan2(-perp[1], -perp[0])
            gx = np.clip(
                bp[0] + perp[0] * suction_offset,
                env.config.world_min_x + base_r * 3,
                env.config.world_max_x - base_r * 3,
            )
            gy = np.clip(
                bp[1] + perp[1] * suction_offset,
                env.config.world_min_y + base_r * 3,
                env.config.world_max_y - base_r * 3,
            )
            # Must be collision-free with arm extended.
            if _goal_config_is_collision_free(
                env, state, robot, gx, gy, ft, arm_length
            ):
                goal_x, goal_y, face_theta = gx, gy, ft
                break
        if goal_x is not None:
            break

    if goal_x is None:
        return False, max_steps

    # Phase 1: Navigate to goal (x, y, theta) with arm retracted.
    step_count = 0
    phase = "navigate"
    for step_i in range(max_steps):
        rx = state.get(robot, "x")
        ry = state.get(robot, "y")
        rt = state.get(robot, "theta")
        arm_joint = state.get(robot, "arm_joint")

        dx_t = goal_x - rx
        dy_t = goal_y - ry
        dist = np.sqrt(dx_t**2 + dy_t**2)
        angle_err = (face_theta - rt + np.pi) % (2 * np.pi) - np.pi
        dtheta = np.clip(
            angle_err, env.config.min_dtheta, env.config.max_dtheta
        )

        if phase == "navigate":
            if dist > 0.02:
                speed = 1.0 if abs(angle_err) < 0.4 else 0.3
                move_dx = np.clip(
                    dx_t * speed, env.config.min_dx, env.config.max_dx
                )
                move_dy = np.clip(
                    dy_t * speed, env.config.min_dy, env.config.max_dy
                )
            else:
                move_dx, move_dy = 0.0, 0.0
            if dist < 0.03 and abs(angle_err) < 0.2:
                phase = "extend"
            action = np.array(
                [move_dx, move_dy, dtheta, 0.0, 0.0], dtype=np.float32
            )
        elif phase == "extend":
            if arm_joint < arm_length - 0.01:
                action = np.array(
                    [0.0, 0.0, 0.0, env.config.max_darm, 0.0],
                    dtype=np.float32,
                )
            else:
                phase = "vacuum"
                action = np.array(
                    [0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
                )
        else:  # vacuum
            action = np.array(
                [0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
            )

        state, _, terminated, _, _ = step_env.step(action)
        step_count += 1
        if terminated:
            return True, step_count

    return False, step_count


def test_grasp_random_solvable_seed0():
    """Test that the scripted solver solves the environment with seed=0."""
    env = ObjectCentricPushPullHook2DGraspRandomEnv()
    step_env = env
    if MAKE_VIDEOS:
        step_env = RecordVideo(env, "unit_test_videos")
    state, _ = step_env.reset(seed=0)
    solved, steps = _solve_grasp(env, state, step_env=step_env)
    step_env.close()
    assert solved, f"Scripted solver failed on seed=0 after {steps} steps"
