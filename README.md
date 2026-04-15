# Dexterous Hand RL

Training a simulated Shadow Hand to solve manipulation tasks with reinforcement learning. Built with MuJoCo 3, Stable-Baselines3, and Gymnasium.

The hand has 24 degrees of freedom and learns three tasks: grasping objects, reorienting them in hand, and peg-in-hole insertion.

## Requirements

* Python 3.11+
* [uv](https://github.com/astral-sh/uv) for dependency management and command execution

## Setup

Clone the repository and install dependencies with `uv`:

```bash
git clone https://github.com/AryaPeer/Dexterous-Hand.git
cd Dexterous-Hand
uv sync
```

This installs the project dependencies, including MuJoCo, Stable-Baselines3, PyTorch, Weights & Biases, and testing tools.

## Training

All training goes through `main.py`. Each task has its own command and default hyperparameters.

```bash
# Grasping with PPO
uv run python main.py train --n-envs 256 --total-timesteps 30000000

# In-hand reorientation with PPO
uv run python main.py train-reorient --n-envs 256 --total-timesteps 400000000

# Peg-in-hole with SAC
uv run python main.py train-peg --n-envs 32 --total-timesteps 40000000

# Tactile version of peg-in-hole
uv run python main.py train-tactile --n-envs 32 --total-timesteps 40000000 --variant both
```

## Evaluation

```bash
uv run python main.py evaluate
uv run python main.py evaluate-reorient
uv run python main.py evaluate-peg
uv run python main.py evaluate-tactile
```

## Tasks

### Grasping (`ShadowHandGrasp-v0`)

Pick up one of several object types from a table.

### Reorientation (`ShadowHandReorient-v0`)

Rotate a cube in hand to match a target orientation.

### Peg-in-Hole (`ShadowHandPeg-v0`)

Grasp a peg, align it with a hole, and insert it.

### Peg-in-Hole with Tactile (`ShadowHandPegTactile-v0`)

Adds simulated 4x4 fingertip taxel grids to the hand. Tactile observations are processed through a small CNN before being passed to the policy. This task is used to compare insertion performance with and without tactile feedback.

## Testing

Run the test suite with:

```bash
uv run pytest
```

## Acknowledgments

The Shadow Hand MJCF model is sourced from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
