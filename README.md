# Fog-of-War Dungeon RL

A small, dependency-free base for a final project about reinforcement learning under incomplete and stochastic information.

The agent sees only a 5×5 window around itself. It can move up, down, left, or right. The map contains hidden traps and a randomly moving enemy. Reaching treasure wins; a trap or enemy ends the episode.

## Reproduce the current result

This project has no third-party Python dependencies. Clone the repository, then run the commands below from its root directory.


```bash
python3 -m unittest discover -s tests -v
python3 run_demo.py --episodes 100
python3 run_demo.py --show --episodes 1
python3 train_q_learning.py --episodes 15000 --evaluate-episodes 1000
```

`run_demo.py` uses `RandomAgent`, an intentionally weak baseline. Your first result table can compare it with a future Q-learning agent.

## Environment contract

```python
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(Action.RIGHT)
```

`Observation.tiles` is a dictionary of relative tile coordinates to visible symbols:

- `A`: agent, `E`: enemy, `G`: treasure, `T`: trap
- `#`: wall, `.`: safe floor, `X`: outside the board

It deliberately excludes every tile outside the vision radius. `Observation.q_key()` is a hashable local representation for a future tabular Q-learning agent.

## Suggested next milestones

1. Keep the random agent as a baseline and record reward, win rate, and episode length.
2. Add a tabular Q-learning agent that uses `observation.q_key()` as its state.
3. Compare both agents across at least 100 seeded episodes.
4. Add a second dungeon layout or adjustable enemy speed.
5. Only then consider DQN, where a neural network estimates Q-values from the 5×5 observation.

## Important limitation

The local observation is intentionally ambiguous: two distinct locations can look identical. That makes this a partially observable problem. A tabular Q-agent is a useful baseline, but it may need a short observation history or a neural model with memory to handle the full problem well.

## Train the Q-learning baseline

```bash
python3 train_q_learning.py --episodes 15000 --evaluate-episodes 1000
```

For each step, the agent records one transition:

```text
(local observation, action, reward, next local observation)
```

It updates the action value using:

```text
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') − Q(s, a)]
```

The Q-table starts at zero. During training, the agent initially explores almost randomly. Its exploration probability gradually decays to 5%. Evaluation sets it to zero, so the reported result reflects learned choices rather than lucky random exploration.

This is a baseline, not the final claim: the agent receives only the current 5×5 view. That means two different places that look identical become the same table state. Report this limitation in the project, then compare it with an improved state representation or DQN if you have time.

## State representation and calculated rewards

The Q-table no longer uses only tile symbols. Its state key contains both:

1. the current 5×5 local window; and
2. which visible squares were already visited during this episode.

This gives the agent a small amount of memory without revealing the global map. It can distinguish “a new corridor to explore” from “a corridor I just came from.”

Rewards are also broken into explicit terms, returned in `info["reward_terms"]` after each non-terminal move:

- base step cost: `-0.1`
- first visit to a tile: `+0.2`
- revisit penalty: `-0.05`, becoming slightly stronger for repeated loops
- first time treasure becomes visible: `+0.5`
- visible-treasure progress: `0.4 × (old Manhattan distance − new distance)`
- visible-enemy danger: `-0.4 / (enemy Manhattan distance + 1)`

The calculated treasure and enemy terms only activate when those objects are visible, so the reward does not reveal hidden map information.


## Course deliverables

- `slides/DungeonGridAgent_5min_Presentation_v2.pptx`: November 19 five-minute presentation.
- `paper/`: LaTeX source and compiled four-page conference-style paper, due December 2.
- `README.md`: setup, validation, demo, and result-reproduction commands.

The presentation result is preliminary and uses one training/evaluation seed. The final report will evaluate several seeds, report mean and variation, and compare the random and Q-learning baselines on more than one dungeon configuration.
