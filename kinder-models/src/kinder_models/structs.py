"""Common data structures."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar

import numpy as np
from gymnasium.spaces import Box
from relational_structs import (
    Object,
    Type,
    Variable,
)

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action

class ParameterizedController(abc.ABC, Generic[_X, _U]):
    """A parameterized policy, a parameter sampler, and a termination check."""

    @abc.abstractmethod
    def sample_parameters(self, x: _X, rng: np.random.Generator) -> Any:
        """Sample parameters."""

    @abc.abstractmethod
    def reset(self, x: _X, params: Any) -> None:
        """Reset the internal state and current parameters."""

    @abc.abstractmethod
    def terminated(self) -> bool:
        """Check if the controller has terminated."""

    @abc.abstractmethod
    def step(self) -> _U:
        """Return the next action to execute."""

    @abc.abstractmethod
    def observe(self, x: _X) -> None:
        """Observe the current state."""


@dataclass(frozen=True)
class LiftedParameterizedController(Generic[_X, _U]):
    """A parameterized controller factory with placeholders for objects."""

    variables: Sequence[Variable]
    controller_cls: type[GroundParameterizedController]
    params_space: Box | None = None

    def ground(
        self, objects: Sequence[Object]
    ) -> GroundParameterizedController[_X, _U]:
        """Create a ground parameterized controller."""
        assert all(
            o.is_instance(v.type) for o, v in zip(objects, self.variables, strict=True)
        )
        return self.controller_cls(objects)

    @property
    def name(self) -> str:
        """Get the name of the controller class."""
        return self.controller_cls.__name__

    @property
    def types(self) -> Sequence[Type]:
        """Get the types of the variables."""
        return [v.type for v in self.variables]

    @property
    def name_vars_str(self) -> str:
        """Get a string representation of the variable names."""
        return f"{self.controller_cls.__name__}{self.var_str}"

    @property
    def var_str(self) -> str:
        """Get a string representation of the variable types."""
        result = "(types=[" + ", ".join(v.type.name for v in self.variables) + "])"
        if self.params_space is not None:
            result += ", params_space=" + str(self.params_space)
        return result


class GroundParameterizedController(ParameterizedController[_X, _U], abc.ABC):
    """A parameterized controller that is object-parameterized.

    Subclasses determine how the objects should be used.
    """

    def __init__(self, objects: Sequence[Object]) -> None:
        self.objects = objects
