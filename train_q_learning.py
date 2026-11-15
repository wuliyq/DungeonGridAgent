"""Train a tabular Q-learning agent and compare it with the random baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol

from agents import RandomAgent, TabularQAgent
from dungeon_env import Action, DungeonEnv, Observation


class ActionAgent(Protocol):
    def choose_action(self, observation: Observation) -> Action: ...


@dataclass
class Evaluation:
    average_reward: float
    treasure_rate: float
    average_steps: float


def train(agent: TabularQAgent, env: DungeonEnv, episodes: int, seed: int) -> list[float]:
    """Train with epsilon decay and return one total reward per episode."""
    rewards: list[float] = []
    for episode in range(episodes):
        # Exploration starts high and finishes at 5% random actions.
        progress = episode / max(1, episodes - 1)
        agent.epsilon = max(0.05, 1.0 - 0.95 * progress)
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        while True:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.learn(observation, action, reward, next_observation, terminated or truncated)
            observation = next_observation
            total_reward += reward
            if terminated or truncated:
                rewards.append(total_reward)
                break
    return rewards


def evaluate(agent: ActionAgent, env: DungeonEnv, episodes: int, seed: int) -> Evaluation:
    rewards: list[float] = []
    wins = 0
    steps: list[int] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        while True:
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                rewards.append(total_reward)
                steps.append(env.steps)
                wins += info["event"] == "treasure"
                break
    return Evaluation(sum(rewards) / episodes, wins / episodes, sum(steps) / episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=15_000)
    parser.add_argument("--evaluate-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    learner = TabularQAgent(seed=args.seed)
    training_rewards = train(learner, DungeonEnv(), args.episodes, args.seed)
    learner.epsilon = 0.0  # Evaluation asks: what has it learned, without exploration?

    q_result = evaluate(learner, DungeonEnv(), args.evaluate_episodes, args.seed + 100_000)
    random_result = evaluate(RandomAgent(seed=args.seed), DungeonEnv(), args.evaluate_episodes, args.seed + 100_000)

    print(f"trained for {args.episodes} episodes; Q-table has {len(learner.q_values)} observed states")
    print(f"last 500 training reward average: {sum(training_rewards[-500:]) / min(500, len(training_rewards)):.2f}")
    print("\nagent            avg reward   treasure rate   avg steps")
    print(f"Q-learning       {q_result.average_reward:10.2f}   {q_result.treasure_rate:11.1%}   {q_result.average_steps:9.1f}")
    print(f"random baseline  {random_result.average_reward:10.2f}   {random_result.treasure_rate:11.1%}   {random_result.average_steps:9.1f}")


if __name__ == "__main__":
    main()
