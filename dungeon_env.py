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
    revisit_penalty: float = -0.05
    treasure_spotted_reward: float = 0.5
    visible_goal_progress_reward: float = 0.4
    enemy_proximity_penalty: float = 0.4
    treasure_reward: float = 20.0
    trap_reward: float = -15.0
    enemy_reward: float = -20.0


@dataclass(frozen=True)
class Observation:
    """Everything the agent currently knows, expressed relative to itself.

    ``tiles`` is the 5×5 visible window. ``visited_offsets`` is a compact
    memory: it marks squares in that window the agent has stood on before.
    The global map and all squares outside the window remain hidden.
    """

    tiles: Mapping[Position, str]
    visited_offsets: frozenset[Position]
    enemy_visible: bool

    def q_key(self) -> tuple:
        """Return an immutable state key for tabular Q-learning.

        Two views are different states if either their visible tiles differ or
        their nearby visit histories differ. This helps distinguish exploring a
        new corridor from walking back into a known loop.
        """
        return (tuple(sorted(self.tiles.items())), tuple(sorted(self.visited_offsets)))


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
        self.visit_counts: dict[Position, int] = {}
        self.terminated = False

    def reset(self, seed: Optional[int] = None) -> tuple[Observation, dict]:
        if seed is not None:
            self.rng.seed(seed)
        self.agent = self.config.start
        self.enemy = self.config.enemy_start
        self.steps = 0
        self.visited = {self.agent}
        self.visit_counts = {self.agent: 1}
        self.terminated = False
        return self.observe(), {"event": "reset"}

    def step(self, action: Action | str) -> tuple[Observation, float, bool, bool, dict]:
        """Move once, then let the enemy make one random legal move.

        Besides terminal rewards, the agent receives small, observation-safe
        shaping rewards: new ground is useful, loops are mildly discouraged,
        moving toward a *visible* treasure is useful, and a visible nearby
        enemy is risky. No hidden tile is used to calculate these terms.
        """
        if self.terminated:
            raise RuntimeError("Episode has ended. Call reset() before step().")
        action = Action(action)
        self.steps += 1
        goal_distance_before = self._visible_treasure_distance()
        next_agent = self._move(self.agent, action)
        reward = self.config.step_reward
        reward_terms = {"step": self.config.step_reward}
        event = "moved"

        if next_agent == self.agent:
            reward = self.config.wall_reward
            reward_terms = {"blocked": self.config.wall_reward}
            event = "blocked"
        else:
            self.agent = next_agent
            count = self.visit_counts.get(self.agent, 0) + 1
            self.visit_counts[self.agent] = count
            self.visited.add(self.agent)
            if count == 1:
                reward += self.config.explore_reward
                reward_terms["new_tile"] = self.config.explore_reward
                event = "explored"
            else:
                penalty = self.config.revisit_penalty * min(count - 1, 3)
                reward += penalty
                reward_terms["revisit"] = penalty
                event = "revisited"

        if self.agent == self.config.treasure:
            self.terminated = True
            return self.observe(), self.config.treasure_reward, True, False, {"event": "treasure", "reward_terms": {"treasure": self.config.treasure_reward}}
        if self.agent in self.config.traps:
            self.terminated = True
            return self.observe(), self.config.trap_reward, True, False, {"event": "trap", "reward_terms": {"trap": self.config.trap_reward}}

        self._move_enemy()
        if self.enemy == self.agent:
            self.terminated = True
            return self.observe(), self.config.enemy_reward, True, False, {"event": "enemy", "reward_terms": {"enemy": self.config.enemy_reward}}

        goal_distance_after = self._visible_treasure_distance()
        if goal_distance_before is None and goal_distance_after is not None:
            reward += self.config.treasure_spotted_reward
            reward_terms["treasure_spotted"] = self.config.treasure_spotted_reward
        elif goal_distance_before is not None and goal_distance_after is not None:
            progress = self.config.visible_goal_progress_reward * (goal_distance_before - goal_distance_after)
            reward += progress
            reward_terms["visible_goal_progress"] = progress

        enemy_distance = self._visible_enemy_distance()
        if enemy_distance is not None:
            danger = -self.config.enemy_proximity_penalty / (enemy_distance + 1)
            reward += danger
            reward_terms["enemy_proximity"] = danger

        truncated = self.steps >= self.config.max_steps
        return self.observe(), reward, False, truncated, {"event": event, "reward_terms": reward_terms}

    def observe(self) -> Observation:
        tiles: Dict[Position, str] = {}
        visited_offsets: set[Position] = set()
        radius = self.config.vision_radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                relative = (dr, dc)
                absolute = (self.agent[0] + dr, self.agent[1] + dc)
                if not self._in_bounds(absolute):
                    tiles[relative] = "X"  # outside the board
                else:
                    tiles[relative] = self._tile_at(absolute)
                    if absolute in self.visited:
                        visited_offsets.add(relative)
        return Observation(
            tiles=tiles,
            visited_offsets=frozenset(visited_offsets),
            enemy_visible=self._can_see(self.enemy),
        )

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

    def _visible_treasure_distance(self) -> Optional[int]:
        if not self._can_see(self.config.treasure):
            return None
        return self._manhattan_distance(self.agent, self.config.treasure)

    def _visible_enemy_distance(self) -> Optional[int]:
        if not self._can_see(self.enemy):
            return None
        return self._manhattan_distance(self.agent, self.enemy)

    @staticmethod
    def _manhattan_distance(first: Position, second: Position) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

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
