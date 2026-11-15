import unittest

from dungeon_env import Action, DungeonConfig, DungeonEnv


class DungeonEnvironmentTests(unittest.TestCase):
    def test_reset_has_a_local_five_by_five_observation(self) -> None:
        env = DungeonEnv()
        observation, _ = env.reset(seed=1)
        self.assertEqual(len(observation.tiles), 25)
        self.assertEqual(observation.tiles[(0, 0)], "A")
        self.assertFalse(observation.enemy_visible)

    def test_wall_move_stays_put_and_is_penalized(self) -> None:
        env = DungeonEnv()
        env.reset(seed=1)
        _, reward, terminated, truncated, info = env.step(Action.UP)
        self.assertEqual(env.agent, (0, 0))
        self.assertEqual(reward, -1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["event"], "blocked")

    def test_treasure_ends_episode(self) -> None:
        config = DungeonConfig(rows=2, cols=2, start=(0, 0), treasure=(0, 1), enemy_start=(1, 1), walls=frozenset(), traps=frozenset())
        env = DungeonEnv(config)
        env.reset(seed=1)
        _, reward, terminated, _, info = env.step(Action.RIGHT)
        self.assertTrue(terminated)
        self.assertEqual(reward, 20.0)
        self.assertEqual(info["event"], "treasure")

    def test_trap_ends_episode(self) -> None:
        config = DungeonConfig(rows=2, cols=3, start=(0, 0), treasure=(1, 2), enemy_start=(1, 1), walls=frozenset(), traps=frozenset({(0, 1)}))
        env = DungeonEnv(config)
        env.reset(seed=1)
        _, reward, terminated, _, info = env.step(Action.RIGHT)
        self.assertTrue(terminated)
        self.assertEqual(reward, -15.0)
        self.assertEqual(info["event"], "trap")


if __name__ == "__main__":
    unittest.main()


class QLearningTests(unittest.TestCase):
    def test_terminal_update_moves_value_toward_reward(self) -> None:
        from agents import TabularQAgent

        env = DungeonEnv()
        observation, _ = env.reset(seed=3)
        agent = TabularQAgent(learning_rate=0.5, epsilon=0.0, seed=3)
        agent.learn(observation, Action.RIGHT, 10.0, observation, episode_finished=True)
        self.assertEqual(agent.q_values[observation.q_key()][Action.RIGHT], 5.0)

    def test_greedy_agent_prefers_highest_q_value(self) -> None:
        from agents import TabularQAgent

        env = DungeonEnv()
        observation, _ = env.reset(seed=3)
        agent = TabularQAgent(epsilon=0.0, seed=3)
        values = agent._values(observation)
        values[Action.DOWN] = 4.0
        self.assertEqual(agent.choose_action(observation), Action.DOWN)
