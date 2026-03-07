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


def _solve_grasp(env, state, max_steps=500, step_env=None):
    """Scripted solver: navigate beside the hook bar, face it, extend arm,
    suction.

    The hook has a random orientation in [pi/4, 3*pi/4]. The L-shape's
    long bar (length_side1) extends from the hook origin along a direction
    determined by theta. The solver targets a point 75% along the bar
    (well below the table), approaches perpendicular to the bar, extends
    the arm so the suction zone overlaps, then turns on vacuum.

    Three phases:
      1. Navigate to standoff position and face the bar (arm retracted).
      2. Extend arm fully (no movement).
      3. Turn on vacuum.
    """
    if step_env is None:
        step_env = env
    obj_map = {o.name: o for o in state}
    robot = obj_map["robot"]
    hook = obj_map["hook"]

    phase = "navigate"

    for step_i in range(max_steps):
        rx = state.get(robot, "x")
        ry = state.get(robot, "y")
        rt = state.get(robot, "theta")
        hx = state.get(hook, "x")
        hy = state.get(hook, "y")
        ht = state.get(hook, "theta")
        hw = state.get(hook, "width")
        hl1 = state.get(hook, "length_side1")
        arm_length = state.get(robot, "arm_length")
        gripper_w = state.get(robot, "gripper_width")
        base_r = state.get(robot, "base_radius")
        arm_joint = state.get(robot, "arm_joint")

        # Compute a target point along the hook's long bar.
        # The bar extends from (hx, hy) in direction (-cos(ht), -sin(ht))
        # with length hl1. Target 75% along the bar (well below the table).
        bar_dir_x = -np.cos(ht)
        bar_dir_y = -np.sin(ht)
        bar_pt_x = hx + bar_dir_x * hl1 * 0.75
        bar_pt_y = hy + bar_dir_y * hl1 * 0.75

        # Approach perpendicular to the bar from the robot's current side.
        perp1_x, perp1_y = -bar_dir_y, bar_dir_x
        perp2_x, perp2_y = bar_dir_y, -bar_dir_x

        to_robot_x = rx - bar_pt_x
        to_robot_y = ry - bar_pt_y
        dot1 = to_robot_x * perp1_x + to_robot_y * perp1_y
        if dot1 >= 0:
            perp_x, perp_y = perp1_x, perp1_y
        else:
            perp_x, perp_y = perp2_x, perp2_y

        # Standoff distance from the bar centerline: must clear the bar
        # half-width (hw/2) plus the arm + gripper.
        standoff_dist = arm_length + gripper_w / 2 + 0.005 + hw / 2

        target_x = bar_pt_x + perp_x * standoff_dist
        target_y = bar_pt_y + perp_y * standoff_dist

        # Clamp within world bounds (robot must stay inside walls).
        target_x = np.clip(
            target_x,
            env.config.world_min_x + base_r * 3,
            env.config.world_max_x - base_r * 3,
        )
        target_y = np.clip(
            target_y,
            env.config.world_min_y + base_r * 3,
            env.config.world_max_y - base_r * 3,
        )

        # Face angle: point from standoff toward the bar target point.
        face_theta = np.arctan2(
            bar_pt_y - target_y, bar_pt_x - target_x
        )

        dx_t = target_x - rx
        dy_t = target_y - ry
        dist_to_target = np.sqrt(dx_t**2 + dy_t**2)

        # Angle control: face the bar.
        angle_err = (face_theta - rt + np.pi) % (2 * np.pi) - np.pi
        dtheta = np.clip(angle_err, -env.config.max_dtheta, env.config.max_dtheta)

        if phase == "navigate":
            # Move toward standoff position with arm retracted.
            if dist_to_target > 0.02:
                speed = 1.0 if abs(angle_err) < 0.4 else 0.3
                move_dx = np.clip(
                    dx_t * speed, -env.config.max_dx, env.config.max_dx
                )
                move_dy = np.clip(
                    dy_t * speed, -env.config.max_dy, env.config.max_dy
                )
            else:
                move_dx = 0.0
                move_dy = 0.0
            # Transition when at position and facing correctly.
            if dist_to_target < 0.03 and abs(angle_err) < 0.2:
                phase = "extend"
            action = np.array(
                [move_dx, move_dy, dtheta, 0.0, 0.0], dtype=np.float32
            )

        elif phase == "extend":
            # Extend arm without moving.
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
        if terminated:
            return True, step_i + 1

    return False, max_steps


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
