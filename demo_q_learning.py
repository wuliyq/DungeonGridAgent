"""Short presentation demo for the trained Q-learning agent.

Run from the repository root:
    python3 demo_q_learning.py
"""

from __future__ import annotations

import argparse

from agents import TabularQAgent
from dungeon_env import DungeonEnv
from train_q_learning import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode-seed", type=int, default=100_007)
    args = parser.parse_args()

    print(f"Training Q-learning for {args.train_episodes:,} episodes...")
    agent = TabularQAgent(seed=args.seed)
    train(agent, DungeonEnv(), args.train_episodes, args.seed)
    agent.epsilon = 0.0

    env = DungeonEnv()
    observation, _ = env.reset(seed=args.episode_seed)
    print(f"Learned local states: {len(agent.q_values)}")
    print("\nInitial fog-of-war view:")
    print(env.render())
    print("\nGreedy action trace:")

    total_reward = 0.0
    while True:
        action = agent.choose_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"step {env.steps:2d}: {action.value:5s}  reward {reward:6.2f}  {info['event']}")
        if terminated or truncated:
            outcome = info["event"] if terminated else "time_limit"
            break

    print(f"\nOutcome: {outcome}; steps: {env.steps}; total reward: {total_reward:.2f}")
    print("\nFinal fog-of-war view:")
    print(env.render())


if __name__ == "__main__":
    main()
