"""Partially observable stochastic dungeon environment.

The environment uses only the Python standard library.  Its API mirrors common
reinforcement-learning environments:

    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(Action.RIGHT)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import Dict, Iterable, Mapping, Optional, Tuple

Position = Tuple[int, int]


class Action(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


ACTION_DELTAS: Mapping[Action, Position] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True)
class DungeonConfig:
    """Everything that defines one dungeon and its reward function."""

    rows: int = 8
    cols: int = 8
    start: Position = (0, 0)
    treasure: Position = (7, 7)
    enemy_start: Position = (4, 4)
    walls: frozenset[Position] = field(
        default_factory=lambda: frozenset(
            {
                (1, 2), (1, 3), (2, 3), (3, 1), (3, 2),
                (4, 5), (5, 5), (6, 1), (6, 2),
            }
        )
    )
    traps: frozenset[Position] = field(
        default_factory=lambda: frozenset({(0, 5), (2, 6), (5, 2), (6, 6)})
    )
    vision_radius: int = 2
    max_steps: int = 100
    step_reward: float = -0.1
    wall_reward: float = -1.0
    explore_reward: float = 0.2
    treasure_reward: float = 20.0
    trap_reward: float = -15.0
    enemy_reward: float = -20.0


@dataclass(frozen=True)
class Observation:
    """Only information currently visible to the agent.

    ``tiles`` maps relative coordinates (dr, dc) to symbols.  Coordinates
    outside vision are deliberately absent, so the complete map is never
    leaked to the learning agent.
    """

    tiles: Mapping[Position, str]
    enemy_visible: bool

    def q_key(self) -> Tuple[Tuple[Position, str], ...]:
        """A hashable local-view representation for a future tabular Q-agent."""
        return tuple(sorted(self.tiles.items())) + (((99, 99), "E") if self.enemy_visible else ())


class DungeonEnv:
    """Grid world with fog of war, hidden traps, and a stochastic enemy."""

    def __init__(self, config: Optional[DungeonConfig] = None) -> None:
        self.config = config or DungeonConfig()
        self._validate_config()
        self.rng = Random()
        self.agent: Position = self.config.start
        self.enemy: Position = self.config.enemy_start
        self.steps = 0
        self.visited: set[Position] = set()
        self.terminated = False

    def reset(self, seed: Optional[int] = None) -> tuple[Observation, dict]:
        if seed is not None:
            self.rng.seed(seed)
        self.agent = self.config.start
        self.enemy = self.config.enemy_start
        self.steps = 0
        self.visited = {self.agent}
        self.terminated = False
        return self.observe(), {"event": "reset"}

    def step(self, action: Action | str) -> tuple[Observation, float, bool, bool, dict]:
        """Move once, then let the enemy make one random legal move."""
        if self.terminated:
            raise RuntimeError("Episode has ended. Call reset() before step().")
        action = Action(action)
        self.steps += 1
        next_agent = self._move(self.agent, action)
        reward = self.config.step_reward
        event = "moved"

        if next_agent == self.agent:
            reward = self.config.wall_reward
            event = "blocked"
        else:
            self.agent = next_agent
            if self.agent not in self.visited:
                self.visited.add(self.agent)
                reward += self.config.explore_reward
                event = "explored"

        if self.agent == self.config.treasure:
            self.terminated = True
            return self.observe(), self.config.treasure_reward, True, False, {"event": "treasure"}
        if self.agent in self.config.traps:
            self.terminated = True
            return self.observe(), self.config.trap_reward, True, False, {"event": "trap"}

        self._move_enemy()
        if self.enemy == self.agent:
            self.terminated = True
            return self.observe(), self.config.enemy_reward, True, False, {"event": "enemy"}

        truncated = self.steps >= self.config.max_steps
        return self.observe(), reward, False, truncated, {"event": event}

    def observe(self) -> Observation:
        tiles: Dict[Position, str] = {}
        radius = self.config.vision_radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                absolute = (self.agent[0] + dr, self.agent[1] + dc)
                if not self._in_bounds(absolute):
                    tiles[(dr, dc)] = "X"  # outside the board
                else:
                    tiles[(dr, dc)] = self._tile_at(absolute)
        return Observation(tiles=tiles, enemy_visible=self._can_see(self.enemy))

    def render(self, reveal_all: bool = False) -> str:
        """Return an ASCII board. Hidden tiles are shown as '?' unless revealed."""
        lines = []
        for row in range(self.config.rows):
            chars = []
            for col in range(self.config.cols):
                pos = (row, col)
                visible = reveal_all or self._can_see(pos)
                chars.append(self._tile_at(pos) if visible else "?")
            lines.append(" ".join(chars))
        return "\n".join(lines)

    def _move_enemy(self) -> None:
        legal = [pos for pos in self._neighbors(self.enemy) if pos not in self.config.traps]
        if legal:
            self.enemy = self.rng.choice(legal)

    def _move(self, position: Position, action: Action) -> Position:
        dr, dc = ACTION_DELTAS[action]
        candidate = (position[0] + dr, position[1] + dc)
        return candidate if self._is_walkable(candidate) else position

    def _neighbors(self, position: Position) -> Iterable[Position]:
        for action in Action:
            candidate = self._move(position, action)
            if candidate != position:
                yield candidate

    def _tile_at(self, position: Position) -> str:
        if position == self.agent:
            return "A"
        if position == self.enemy:
            return "E"
        if position in self.config.walls:
            return "#"
        if position == self.config.treasure:
            return "G"
        if position in self.config.traps:
            return "T"
        return "."

    def _can_see(self, position: Position) -> bool:
        return max(abs(position[0] - self.agent[0]), abs(position[1] - self.agent[1])) <= self.config.vision_radius

    def _is_walkable(self, position: Position) -> bool:
        return self._in_bounds(position) and position not in self.config.walls

    def _in_bounds(self, position: Position) -> bool:
        return 0 <= position[0] < self.config.rows and 0 <= position[1] < self.config.cols

    def _validate_config(self) -> None:
        important = {self.config.start, self.config.treasure, self.config.enemy_start}
        if len(important) != 3:
            raise ValueError("Start, treasure, and enemy must be distinct positions.")
        for pos in important | set(self.config.walls) | set(self.config.traps):
            if not self._in_bounds(pos):
                raise ValueError(f"Position {pos} is outside the board.")
        if important & set(self.config.walls | self.config.traps):
            raise ValueError("Start, treasure, and enemy cannot be walls or traps.")
