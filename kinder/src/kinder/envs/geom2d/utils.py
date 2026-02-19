"""Utilities."""

from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from relational_structs import (
    Array,
    Object,
    ObjectCentricState,
)
from tomsgeoms2d.structs import Rectangle
from tomsgeoms2d.utils import find_closest_points, geom2ds_intersect

from kinder.core import RobotActionSpace
from kinder.envs.geom2d.object_types import (
    DoubleRectType,
    RectangleType,
)
from kinder.envs.geom2d.structs import (
    MultiBody2D,
    SE2Pose,
    ZOrder,
)
from kinder.envs.utils import (
    BLACK,
    crv_robot_to_multibody2d,
    double_rectangle_object_to_part_geom,
    get_se2_pose,
    object_to_multibody2d,
    rectangle_object_to_geom
)


class CRVRobotActionSpace(RobotActionSpace):
    """An action space for a CRV robot.

    Actions are bounded relative movements of the base and the arm, as well as an
    absolute setting for the vacuum.
    """

    def __init__(
        self,
        min_dx: float = -5e-1,
        max_dx: float = 5e-1,
        min_dy: float = -5e-1,
        max_dy: float = 5e-1,
        min_dtheta: float = -np.pi / 16,
        max_dtheta: float = np.pi / 16,
        min_darm: float = -1e-1,
        max_darm: float = 1e-1,
        min_vac: float = 0.0,
        max_vac: float = 1.0,
    ) -> None:
        low = np.array([min_dx, min_dy, min_dtheta, min_darm, min_vac])
        high = np.array([max_dx, max_dy, max_dtheta, max_darm, max_vac])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        # pylint: disable=line-too-long
        features = [
            ("dx", "Change in robot x position (positive is right)"),
            ("dy", "Change in robot y position (positive is up)"),
            ("dtheta", "Change in robot angle in radians (positive is ccw)"),
            ("darm", "Change in robot arm length (positive is out)"),
            ("vac", "Directly sets the vacuum (0.0 is off, 1.0 is on)"),
        ]
        md_table_str = (
            "| **Index** | **Feature** | **Description** | **Min** | **Max** |"
        )
        md_table_str += "\n| --- | --- | --- | --- | --- |"
        for idx, (feature, description) in enumerate(features):
            lb = self.low[idx]
            ub = self.high[idx]
            md_table_str += (
                f"\n| {idx} | {feature} | {description} | {lb:.3f} | {ub:.3f} |"
            )
        return f"The entries of an array in this Box space correspond to the following action features:\n{md_table_str}\n"


def create_walls_from_world_boundaries(
    world_min_x: float,
    world_max_x: float,
    world_min_y: float,
    world_max_y: float,
    min_dx: float,
    max_dx: float,
    min_dy: float,
    max_dy: float,
) -> dict[Object, dict[str, float]]:
    """Create wall objects and feature dicts based on world boundaries.

    Velocities are used to determine how large the walls need to be to avoid the
    possibility that the robot will transport over the wall.
    """
    state_dict: dict[Object, dict[str, float]] = {}
    # Right wall.
    right_wall = Object("right_wall", RectangleType)
    side_wall_height = world_max_y - world_min_y
    state_dict[right_wall] = {
        "x": world_max_x,
        "y": world_min_y,
        "width": 2 * max_dx,  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Left wall.
    left_wall = Object("left_wall", RectangleType)
    state_dict[left_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_min_y,
        "width": 2 * abs(min_dx),  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Top wall.
    top_wall = Object("top_wall", RectangleType)
    horiz_wall_width = 2 * 2 * abs(min_dx) + world_max_x - world_min_x
    state_dict[top_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_max_y,
        "width": horiz_wall_width,
        "height": 2 * max_dy,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Bottom wall.
    bottom_wall = Object("bottom_wall", RectangleType)
    state_dict[bottom_wall] = {
        "x": world_min_x + 2 * min_dx,
        "y": world_min_y + 2 * min_dy,
        "width": horiz_wall_width,
        "height": 2 * max_dy,
        "theta": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    return state_dict


def get_tool_tip_position(
    state: ObjectCentricState, robot: Object
) -> tuple[float, float]:
    """Get the tip of the tool for the robot, which is defined as the center of the
    bottom edge of the gripper."""
    multibody = crv_robot_to_multibody2d(robot, state)
    gripper_geom = multibody.get_body("gripper").geom
    assert isinstance(gripper_geom, Rectangle)
    # Transform the x, y point.
    tool_tip = np.array([1.0, 0.5])
    scale_matrix = np.array(
        [
            [gripper_geom.width, 0],
            [0, gripper_geom.height],
        ]
    )
    translate_vector = np.array([gripper_geom.x, gripper_geom.y])
    tool_tip = tool_tip @ scale_matrix.T
    tool_tip = tool_tip @ gripper_geom.rotation_matrix.T
    tool_tip = translate_vector + tool_tip
    return (tool_tip[0], tool_tip[1])


def get_suctioned_objects(
    state: ObjectCentricState, robot: Object
) -> list[tuple[Object, SE2Pose]]:
    """Find objects that are in the suction zone of a CRVRobot and return the associated
    transform from gripper tool tip to suctioned object."""
    # If the robot's vacuum is not on, there are no suctioned objects.
    if state.get(robot, "vacuum") <= 0.5:
        return []
    robot_multibody = crv_robot_to_multibody2d(robot, state)
    suction_body = robot_multibody.get_body("suction")
    gripper_x, gripper_y = get_tool_tip_position(state, robot)
    gripper_theta = state.get(robot, "theta")
    world_to_gripper = SE2Pose(gripper_x, gripper_y, gripper_theta)
    # Find MOVABLE objects in collision with the suction geom.
    movable_objects = [o for o in state if o != robot and state.get(o, "static") < 0.5]
    suctioned_objects: list[tuple[Object, SE2Pose]] = []
    for obj in movable_objects:
        # No point in using a static object cache because these objects are
        # not static by definition.
        obj_multibody = object_to_multibody2d(obj, state, {})
        for obj_body in obj_multibody.bodies:
            if geom2ds_intersect(suction_body.geom, obj_body.geom):
                world_to_obj = get_se2_pose(state, obj)
                gripper_to_obj = world_to_gripper.inverse * world_to_obj
                suctioned_objects.append((obj, gripper_to_obj))
    return suctioned_objects


def snap_suctioned_objects(
    state: ObjectCentricState,
    robot: Object,
    suctioned_objs: list[tuple[Object, SE2Pose]],
) -> None:
    """Updates the state in-place."""
    gripper_x, gripper_y = get_tool_tip_position(state, robot)
    gripper_theta = state.get(robot, "theta")
    world_to_gripper = SE2Pose(gripper_x, gripper_y, gripper_theta)
    for obj, gripper_to_obj in suctioned_objs:
        world_to_obj = world_to_gripper * gripper_to_obj
        state.set(obj, "x", world_to_obj.x)
        state.set(obj, "y", world_to_obj.y)
        state.set(obj, "theta", world_to_obj.theta)


def move_objects_in_contact(
    state: ObjectCentricState,
    robot: Object,
    suctioned_objs: list[tuple[Object, SE2Pose]],
) -> tuple[ObjectCentricState, set[tuple[Object, SE2Pose]]]:
    """Move objects that are in contact with the robot's suctioned objects."""
    moved_objects = []
    moving_objects = {robot} | {o for o, _ in suctioned_objs}
    nonstatic_objects = {
        o for o in state if (o not in moving_objects) and (not state.get(o, "static"))
    }

    for contact_obj in nonstatic_objects:
        for suctioned_obj, _ in suctioned_objs:
            suctioned_body = object_to_multibody2d(suctioned_obj, state, {})
            contact_body = object_to_multibody2d(contact_obj, state, {})
            for b1 in suctioned_body.bodies:
                for b2 in contact_body.bodies:
                    if geom2ds_intersect(b1.geom, b2.geom):
                        closest_points_b1, closest_points_b2, _ = find_closest_points(
                            b1.geom, b2.geom
                        )
                        contact_vec = np.array(closest_points_b2) - np.array(
                            closest_points_b1
                        )

                        current_x = state.get(contact_obj, "x")
                        current_y = state.get(contact_obj, "y")

                        new_x = current_x + contact_vec[0]
                        new_y = current_y + contact_vec[1]

                        state.set(contact_obj, "x", new_x)
                        state.set(contact_obj, "y", new_y)

                        moved_objects.append(
                            (contact_obj, get_se2_pose(state, contact_obj))
                        )

                        return state, set(moved_objects)  # Stop checking other objects

    return state, set(moved_objects)


def crv_pose_plan_to_action_plan(
    pose_plan: list[SE2Pose],
    action_space: CRVRobotActionSpace,
    vacuum_while_moving: bool = False,
) -> list[Array]:
    """Convert a CRV robot pose plan into corresponding actions."""
    action_plan: list[Array] = []
    for pt1, pt2 in zip(pose_plan[:-1], pose_plan[1:]):
        action = np.zeros_like(action_space.high)
        action[0] = pt2.x - pt1.x
        action[1] = pt2.y - pt1.y
        action[2] = get_signed_angle_distance(pt2.theta, pt1.theta)
        action[4] = 1.0 if vacuum_while_moving else 0.0
        action_plan.append(action)
    return action_plan


def is_inside(
    state: ObjectCentricState,
    inner: Object,
    outer: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> bool:
    """Checks if the inner object is completely inside the outer one.

    Only rectangles are currently supported.
    """
    inner_geom = rectangle_object_to_geom(state, inner, static_object_cache)
    outer_geom = rectangle_object_to_geom(state, outer, static_object_cache)
    for x, y in inner_geom.vertices:
        if not outer_geom.contains_point(x, y):
            return False
    return True


def is_inside_shelf(
    state: ObjectCentricState,
    inner: Object,
    outer: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> bool:
    """Checks if the inner object is completely inside the outer shelf.

    The outer object is assumed to be a double rectangle type. (shelf)
    """
    assert outer.is_instance(
        DoubleRectType
    ), "Outer object must be a shelf (DoubleRectType)."
    inner_geom = rectangle_object_to_geom(state, inner, static_object_cache)
    outer_geom = double_rectangle_object_to_part_geom(state, outer, static_object_cache)
    for x, y in inner_geom.vertices:
        if not outer_geom.contains_point(x, y):
            return False
    return True


def is_on(
    state: ObjectCentricState,
    top: Object,
    bottom: Object,
    static_object_cache: dict[Object, MultiBody2D],
    tol: float = 0.025,
) -> bool:
    """Checks top object is completely on the bottom one.

    Only rectangles are currently supported.

    Assumes that "up" is positive y.
    """
    top_geom = rectangle_object_to_geom(state, top, static_object_cache)
    bottom_geom = rectangle_object_to_geom(state, bottom, static_object_cache)
    # The bottom-most vertices of top_geom should be contained within the bottom
    # geom when those vertices are offset by tol.
    sorted_vertices = sorted(top_geom.vertices, key=lambda v: v[1])
    for x, y in sorted_vertices[:2]:
        offset_y = y - tol
        if not bottom_geom.contains_point(x, offset_y):
            return False
    return True


def is_movable_rectangle(state: ObjectCentricState, obj: Object) -> bool:
    """Checks if an object is a movable rectangle."""
    return obj.is_instance(RectangleType) and state.get(obj, "static") < 0.5


def get_geom2d_crv_robot_action_from_gui_input(
    action_space: CRVRobotActionSpace, gui_input: dict[str, Any]
) -> NDArray[np.float32]:
    """Get the mapping from human inputs to actions, derived from action space."""
    # Unpack the input.
    keys_pressed = gui_input["keys"]
    right_x, right_y = gui_input["right_stick"]
    left_x, _ = gui_input["left_stick"]

    # Initialize the action.
    low = action_space.low
    high = action_space.high
    action = np.zeros(action_space.shape, action_space.dtype)

    def _rescale(x: float, lb: float, ub: float) -> float:
        """Rescale from [-1, 1] to [lb, ub]."""
        return lb + (x + 1) * (ub - lb) / 2

    # The right stick controls the x, y movement of the base.
    action[0] = _rescale(right_x, low[0], high[0])
    action[1] = _rescale(right_y, low[1], high[1])

    # The left stick controls the rotation of the base. Only the x axis
    # is used right now.
    action[2] = _rescale(left_x, low[2], high[2])

    # The w/s mouse keys are used to adjust the robot arm.
    if "w" in keys_pressed:
        action[3] = low[3]
    if "s" in keys_pressed:
        action[3] = high[3]

    # The space bar is used to turn on the vacuum.
    if "space" in keys_pressed:
        action[4] = 1.0

    return action
