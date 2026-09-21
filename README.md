![CI](https://github.com/felipebapa/mountain_car/actions/workflows/ci.yml/badge.svg?branch=main)

A hands-on repo for understanding how Reinforcement Learning works.
Train, inspect, and visualise RL agents on [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) (or any other Gymnasium environment).

**This repo is a set of exercises.** The CLI, training loops and persistence are
written; the algorithms themselves are left as marked `EXERCISE` stubs for you
to fill in. Start with **[EXERCISES.md](EXERCISES.md)**.

## Team — Group 6

- Felipe Barreto
- Angie Paola Espinosa Hurtado
- Luis Jorge García Camargo
- Valeria Sofía Guerrero Mejía
- Andrés Felipe Miranda Díaz
- Juan Pablo Moreno Mendoza
- Luis Eduardo Uribe Álvarez

## MountainCar-v0 environment

An under-powered car sits in a valley. Its engine is too weak to drive straight
up the right-hand hill, so the only way out is to rock back and forth and build
up momentum. The goal is to reach the flag at position `0.5`.

### State (observation) — 2 continuous values

| Index | Variable | Description | Range |
|:---:|---|---|---|
| 0 | position | Position of the car along the x-axis | -1.2 to 0.6 |
| 1 | velocity | Velocity of the car | -0.07 to 0.07 |

### Actions — 3 discrete

| Value | Action |
|:---:|---|
| 0 | Accelerate to the left |
| 1 | Don't accelerate |
| 2 | Accelerate to the right |

### Rewards

| Event | Reward |
|---|---|
| Every step taken | **-1** |
| Reaching the flag (position >= 0.5) | episode ends |

The reward is `-1` per step and nothing else, so the total return is simply the
negative of the episode length: **less negative is better**. Episodes are cut
off after 200 steps, which gives a floor of `-200` for a policy that never
reaches the flag. Anything around `-110` or better is considered solved.

This flat reward is what makes MountainCar interesting: there is no gradient to
follow toward the goal, so the agent has to stumble onto the flag by
exploration before it can learn anything at all.

## Install

```bash
uv sync
```

## Usage

All commands are exposed through the `mountaincar` CLI:

```bash
uv run mountaincar <command>
```

| Command | What it does |
|---|---|
| `version` | Show the package version |
| `list` | List the agents and whether each has a save file |
| `inspect` | Print the state/action spaces and some random transitions |
| `init <agent>` | Create a new, untrained agent and save it |
| `train <agent>` | Train an agent (resumes from its save if one exists) |
| `load <agent>` | Print a saved agent's info, optionally evaluate it |
| `sim <agent>` | Play episodes with a trained agent, printed step by step |
| `render <agent>` | Play episodes in a graphical window |
| `delete <agent>` | Delete an agent's save file |

`<agent>` is either `qlearning` or `dqn`.

### Example session

```bash
# See what the environment looks like
uv run mountaincar inspect --steps 3

# Train the tabular agent
uv run mountaincar train qlearning --episodes 10000

# How did it do?
uv run mountaincar load qlearning --eval

# Watch it drive
uv run mountaincar render qlearning --episodes 3
```

### Reproducing our results

`scripts/experimento.py` trains an agent, saves it to `saves/` (the same path the
CLI uses), evaluates it on 100 greedy episodes (no exploration) and writes the
training curve, the evaluation plot and a metrics summary to `results/<agent>/`:

```bash
uv run python scripts/experimento.py qlearning --episodes 20000   # ~2 min
uv run python scripts/experimento.py dqn --episodes 2500         # ~13 min on CPU
uv run python scripts/comparacion.py                              # Q-Learning vs DQN plot
```

Both runs use a fixed seed (`--seed 0` by default). Afterwards the CLI works on the
trained agents, e.g. `uv run mountaincar load dqn --eval` or
`uv run mountaincar render dqn --episodes 3`.

The notebooks in `notebooks/` hold our first hyperparameter experiments; run them
from the repository root inside the `uv sync` environment.

## Agents

Both agents live in `src/mountain_car/agents/` and are written from scratch
(no Stable-Baselines3 or similar), so every part of the algorithm is visible --
and, in this repo, **partly left for you to write**. See [EXERCISES.md](EXERCISES.md).

### `qlearning` — tabular Q-Learning

The observation is only 2-dimensional and the environment publishes hard bounds
for both dimensions, so the state space is discretised into an
`n_bins x n_bins` grid (400 states by default) and stored in a plain Q-table.

Defaults: `n_bins=20`, `lr=0.1`, `gamma=0.99`, epsilon `1.0 -> 0.01` decaying by
`0.9995` per episode. A correct implementation scores about `-133` and reaches
the flag in 100/100 episodes, after roughly 20k episodes (~4 min).

### `dqn` — Deep Q-Network

A small MLP on the raw 2-D observation, trained with experience replay and a
target network. A correct implementation scores about `-106` and reaches the
flag in 100/100 episodes, after roughly 2500 episodes (~5 min on CPU) -- better
than the tabular agent, and past the conventional "solved" threshold of `-110`.

Getting there takes more than transcribing the DQN pseudocode. MountainCar has
a reward structure that defeats the textbook version of the algorithm, and
Exercise 3 is about finding out how and why. That exercise ships with a ladder
of progressive clues, so it is a guided investigation rather than a wall.

> A note on hardware: none of this needs a GPU. The network is tiny and the
> batches are small, so a gradient step costs about 0.5 ms on CPU and the
> bottleneck is stepping the environment, not matrix multiplication. On a GPU
> this would most likely be *slower*, because per-kernel launch overhead would
> dominate work this small.

## What we implemented

### Q-Learning — [`qlearning.py`](src/mountain_car/agents/qlearning.py)

- **`discretize`**: each dimension is split into 20 bins with `np.digitize`, so a
  state is one cell of a 20×20 grid, used as a tuple key into the Q-table.
- **`select_action`**: epsilon-greedy; with `deterministic=True` it never explores.
- **`_update`**: the TD update

  ```
  target  = r + gamma * max_a' Q(s', a')     (just r if the episode terminated at the flag)
  Q(s, a) += lr * (target - Q(s, a))
  ```

### DQN — [`dqn.py`](src/mountain_car/agents/dqn.py)

- **`QNetwork`**: MLP `2 → 128 → 128 → 3`, ReLU on the hidden layers, no output activation.
- **`_learn`**: samples a mini-batch of 64 transitions from the replay buffer and
  minimises the MSE against the Bellman target computed with the frozen target network
  (synced every 10 episodes):

  ```
  target = r + gamma * max_a' Q_target(s', a') * (1 - terminated)
  loss   = MSE(Q(s, a), target)
  ```

- **`select_action`**: exploration is temporally correlated. When the agent decides to
  explore, it commits to that action for 25 steps instead of drawing a new one every
  step, which is what produces the sustained pushes needed to rock out of the valley.
  `deterministic=True` stays purely greedy, so evaluation never explores.

## Training schemes

- [Q-Learning training scheme (PDF)](docs/Esquema_del_entrenamiento_de_Q-Learning.pdf)
- [DQN training scheme (PDF)](docs/Esquema_del_entrenamiento_de_DQN.pdf)

## Results

Evaluation is over **100 greedy episodes** (no exploration) with the final agent of
each run.

| Agent | Training episodes | Evaluation mean ± std | Best / worst episode | Reached the flag |
|---|---:|---:|---:|---:|
| Q-Learning | 20,000 | −132.54 ± 19.21 | −114 / −167 | 100/100 |
| DQN | 2,500 | **−107.70 ± 14.06** | **−88** / −142 | 100/100 |

### Q-Learning

![Q-Learning result](docs/evidencia/qlearning_resultado.png)

**Comment.** For the first ~2,000 episodes the reward sits at −200: the agent has never
reached the flag, so there is nothing to propagate. Once it does, that information flows
backwards through the Q-table and the curve climbs. It then oscillates between −140 and
−175 for the rest of the run: with only 400 cells, different states share a cell and the
policy cannot get any finer. In evaluation it reaches the flag in **100/100 episodes**
with a mean of **−132.54**, short of the −110 threshold.

### DQN

![DQN result](docs/evidencia/dqn_resultado.png)

**Comment.** DQN is flat at −195 while exploration dominates, lifts off around episode
600 once the replay buffer holds successful episodes, and settles near −130. With
exploration switched off it reaches the flag in **100/100 episodes** with a mean of
**−107.70**, clearing the −110 threshold with **8 times fewer episodes** than the table.

The training curve stays at −130 while evaluation gives −107.70. The gap is exploration:
during training the 25-step runs spoil the episode in progress, and evaluation has none.

The advantage over the table comes from not discretising. The network takes position and
velocity as continuous numbers and generalises between nearby states, while the table
treats every grid cell separately.

### Q-Learning vs DQN

| | Q-Learning | DQN |
|---|---:|---:|
| Evaluation mean (100 greedy episodes) | −132.54 | **−107.70** |
| Standard deviation | ±19.21 | **±14.06** |
| Best / worst evaluation episode | −114 / −167 | **−88 / −142** |
| Reached the flag | 100/100 | 100/100 |
| Reaches the −110 "solved" threshold | No | **Yes** |
| Training episodes | 20,000 | **2,500** |

DQN needs about 25 fewer steps per episode and clears the threshold that the table does
not, learning from 8 times fewer episodes. Each DQN episode costs more, though, because
it runs a gradient step per environment step.

## Exercise 3: why DQN would not learn

With 2a and 2b correct, DQN trains without errors and learns nothing: −200.00 for 1,000
episodes, with no variation. Four measurements located the cause.

1. **The learning code was fine.** The same agent, unmodified, trained for 200 episodes
   on `CartPole-v1` went from 21.12 to 284.28. The network, the update and the buffer
   work, so the failure was specific to MountainCar.
2. **The agent had never seen the goal.** Over 300 episodes of random actions there were
   0 flag reaches; all 300 were cut by the 200-step limit. The average maximum position
   was −0.389 and the best of the 300 was −0.163, with the goal at 0.5.
3. **The network had learned that nothing it does matters.** Across 200 random states,
   the mean spread between the three action values was 0.0058, and the mean value was
   approaching −100, the discounted sum of −1 forever. Given the data it saw, that was
   correct.
4. **The required behaviour was unreachable.** Escaping the valley takes about 20
   consecutive pushes in the same direction. Drawing a fresh action each step from three
   options gives that a probability of (1/3)^20, so more episodes never help.

The fix is in the data being collected. Drawing each step independently makes
consecutive actions cancel out, so they have to be made dependent in time: when the
agent explores, it holds that action for 25 steps. Flag reaches go from 0/300 to 46/300.
The update rule, the reward and the environment were left untouched.

### Choosing 25 steps

Values 20, 25 and 30 were trained 3 independent times each, with 100 evaluation
episodes per run:

| Steps | Mean of the 3 runs | Std across runs |
|---|---:|---:|
| 20 | −113.61 | 13.79 |
| 25 | −103.69 | 4.06 |
| 30 | −104.58 | 2.41 |

All nine runs solved the environment 100/100. Values 25 and 30 are indistinguishable,
while 20 reaches similar means but is far less stable: one of its three runs dropped to
−132.65. 25 was adopted for its stability. The figures above come from the run that is
being delivered, which is why its mean differs from the sweep average.

## Project layout

```
src/mountain_car/
├── cli.py              # argparse CLI, one command per function
└── agents/
    ├── qlearning.py    # tabular Q-Learning
    └── dqn.py          # DQN: QNetwork, ReplayBuffer, DQNAgent
scripts/
├── experimento.py      # train + evaluate + plots and summary
└── comparacion.py      # Q-Learning vs DQN comparison plot
notebooks/              # first hyperparameter experiments
results/                # raw per-episode rewards and run summaries
saves/                  # agent save files land here (not committed)
docs/
├── evidencia/          # evidence of the best result of each agent
├── Esquema_del_entrenamiento_de_Q-Learning.pdf
└── Esquema_del_entrenamiento_de_DQN.pdf

EXERCISES.md            # the exercises: what to implement, in what order
```
