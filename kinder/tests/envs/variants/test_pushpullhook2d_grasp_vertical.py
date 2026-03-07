"""Tests for pushpullhook2d_grasp_vertical.py.

Verifies the environment keeps all original objects and geometry,
the hook is always vertical, and the env can be solved by a simple
scripted policy that navigates to the hook and suctions it.
"""

import numpy as np

from kinder.envs.geom2d.pushpullhook2d_grasp_vertical import (
    ObjectCentricPushPullHook2DGraspVerticalEnv,
)


def _solve_grasp(env, state, max_steps=500):
    """Scripted solver: navigate toward the hook's long bar, extend arm,
    turn on vacuum.

    At theta=pi/2 the L-shape origin is at the top and the long bar
    (length_side1) extends downward. The bar's bottom tip reaches below
    the table edge into the robot's reachable area. The robot approaches
    from below, points its arm upward toward the bar, extends, and
    suctions.

    Returns (terminated, num_steps).
    """
    obj_map = {o.name: o for o in state}
    robot = obj_map["robot"]
    hook = obj_map["hook"]

    for step_i in range(max_steps):
        rx = state.get(robot, "x")
        ry = state.get(robot, "y")
        rt = state.get(robot, "theta")
        hx = state.get(hook, "x")
        hy = state.get(hook, "y")
        hl1 = state.get(hook, "length_side1")
        hw = state.get(hook, "width")

        # Target: bottom tip of the hook's long bar, offset slightly to
        # approach the thin edge from the side where the robot can reach
        # without its base colliding with the bar.
        # At theta=pi/2 the long bar runs from (hx, hy) downward to
        # (hx, hy - l1). The bar center x is hx + hw/2.
        # We target just outside the bar edge so the suction zone can
        # overlap the bar.
        bar_bottom_y = hy - hl1
        bar_center_x = hx + hw / 2

        # Approach from whichever side the robot is already on.
        arm_length = state.get(robot, "arm_length")
        gripper_w = state.get(robot, "gripper_width")
        standoff = arm_length * 0.8  # don't collide base into bar
        if rx < bar_center_x:
            target_x = bar_center_x - standoff
        else:
            target_x = bar_center_x + standoff
        # Target a point near the bottom of the bar (which hangs below
        # the table edge and is more accessible).
        target_y = bar_bottom_y + hl1 * 0.15

        dx_t = target_x - rx
        dy_t = target_y - ry
        dist_to_target = np.sqrt(dx_t**2 + dy_t**2)

        # Desired angle: point arm toward the bar center at target_y.
        aim_dx = bar_center_x - rx
        aim_dy = target_y - ry
        desired_theta = np.arctan2(aim_dy, aim_dx)
        angle_err = (desired_theta - rt + np.pi) % (2 * np.pi) - np.pi

        max_dx = env.config.max_dx
        max_dy = env.config.max_dy
        max_dtheta = env.config.max_dtheta
        max_darm = env.config.max_darm

        dtheta = np.clip(angle_err, -max_dtheta, max_dtheta)

        # Move toward standoff position.
        if dist_to_target > 0.05:
            speed = 1.0 if abs(angle_err) < 0.4 else 0.3
            move_dx = np.clip(dx_t * speed, -max_dx, max_dx)
            move_dy = np.clip(dy_t * speed, -max_dy, max_dy)
        else:
            move_dx = 0.0
            move_dy = 0.0

        # Extend arm and vacuum when near the standoff.
        if dist_to_target < 0.15:
            darm = max_darm
            vacuum = 1.0
        else:
            darm = 0.0
            vacuum = 0.0

        action = np.array(
            [move_dx, move_dy, dtheta, darm, vacuum], dtype=np.float32
        )
        state, reward, terminated, truncated, info = env.step(action)

        if terminated:
            return True, step_i + 1

    return False, max_steps


def test_grasp_vertical_env_basic():
    """Test that the environment can reset and step without errors."""
    env = ObjectCentricPushPullHook2DGraspVerticalEnv()
    state, info = env.reset(seed=42)

    # All original objects must be present.
    obj_names = {o.name for o in state}
    assert "robot" in obj_names
    assert "hook" in obj_names
    assert "movable_button" in obj_names
    assert "target_button" in obj_names

    # Hook theta is always pi/2.
    hook = next(o for o in state if o.name == "hook")
    assert np.isclose(state.get(hook, "theta"), np.pi / 2, atol=1e-6)

    # Step with a zero action.
    action = np.zeros(5, dtype=np.float32)
    next_state, reward, terminated, truncated, info = env.step(action)
    assert reward == -1.0
    assert not terminated
    env.close()


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


def test_grasp_vertical_env_solvable():
    """Test that the scripted solver can reliably solve the environment."""
    env = ObjectCentricPushPullHook2DGraspVerticalEnv()
    num_successes = 0
    num_trials = 10
    for seed in range(num_trials):
        state, _ = env.reset(seed=seed)
        solved, steps = _solve_grasp(env, state)
        if solved:
            num_successes += 1
    env.close()
    # The scripted policy should solve at least 80% of random seeds.
    assert num_successes >= 8, (
        f"Scripted solver only solved {num_successes}/{num_trials} episodes"
    )


def test_grasp_vertical_reward_structure():
    """Test that reward is -1 until success (0)."""
    env = ObjectCentricPushPullHook2DGraspVerticalEnv()
    env.reset(seed=0)

    # Taking a no-op should give -1.
    action = np.zeros(5, dtype=np.float32)
    _, reward, terminated, _, _ = env.step(action)
    assert reward == -1.0
    assert not terminated
    env.close()
