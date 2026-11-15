"""Run and inspect the random baseline."""

from __future__ import annotations

import argparse

from agents import RandomAgent
from dungeon_env import DungeonEnv


def play_episode(env: DungeonEnv, agent: RandomAgent, seed: int, show: bool) -> tuple[float, str]:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    while True:
        action = agent.choose_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if show:
            print(f"\naction={action.value:5s} reward={reward:5.1f} event={info['event']}")
            print(env.render())
        if terminated or truncated:
            return total_reward, info["event"] if terminated else "time_limit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show", action="store_true", help="Print one episode under fog of war")
    args = parser.parse_args()

    env = DungeonEnv()
    agent = RandomAgent(seed=args.seed)
    rewards = []
    outcomes: dict[str, int] = {}
    for episode in range(args.episodes):
        reward, outcome = play_episode(env, agent, args.seed + episode, args.show and episode == 0)
        rewards.append(reward)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    print(f"episodes={args.episodes} average_reward={sum(rewards) / len(rewards):.2f}")
    print("outcomes:", outcomes)
    print("\nFull map (for debugging only):")
    print(env.render(reveal_all=True))


if __name__ == "__main__":
    main()
