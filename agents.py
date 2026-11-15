"""Baselines and common interface for dungeon agents."""

from __future__ import annotations

from random import Random
from typing import Protocol

from dungeon_env import Action, Observation


class Agent(Protocol):
    def choose_action(self, observation: Observation) -> Action:
        """Choose one of the four actions from the partial observation."""


class RandomAgent:
    """A baseline that ignores the observation and chooses a random direction."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = Random(seed)

    def choose_action(self, observation: Observation) -> Action:
        del observation
        return self.rng.choice(list(Action))


class TabularQAgent:
    """Epsilon-greedy Q-learning agent using only the current local view.

    This agent never reads the environment's hidden map or absolute position.
    A state is the 5×5 observation encoded by ``Observation.q_key()``.
    """

    def __init__(
        self,
        learning_rate: float = 0.15,
        discount: float = 0.95,
        epsilon: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.rng = Random(seed)
        self.q_values: dict[tuple, dict[Action, float]] = {}

    def choose_action(self, observation: Observation) -> Action:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(Action))
        return self.greedy_action(observation)

    def greedy_action(self, observation: Observation) -> Action:
        values = self._values(observation)
        best_value = max(values.values())
        best_actions = [action for action, value in values.items() if value == best_value]
        return self.rng.choice(best_actions)

    def learn(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        next_observation: Observation,
        episode_finished: bool,
    ) -> None:
        """Apply one Bellman Q-learning update.

        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
        The future-value term is omitted when the episode has ended.
        """
        current = self._values(observation)
        old_value = current[action]
        future_value = 0.0 if episode_finished else max(self._values(next_observation).values())
        target = reward + self.discount * future_value
        current[action] = old_value + self.learning_rate * (target - old_value)

    def _values(self, observation: Observation) -> dict[Action, float]:
        key = observation.q_key()
        if key not in self.q_values:
            self.q_values[key] = {action: 0.0 for action in Action}
        return self.q_values[key]
