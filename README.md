# Dexterous Hand RL

Training a simulated Shadow Hand to solve manipulation tasks with reinforcement learning. Built with MuJoCo 3, Stable-Baselines3, and Gymnasium.

The hand has 24 degrees of freedom and learns three tasks: grasping a cube, reorienting a cube in hand, and peg-in-hole insertion.

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
```

## Evaluation

```bash
uv run python main.py evaluate
uv run python main.py evaluate-reorient
uv run python main.py evaluate-peg
```

## Tasks

### Grasping (`ShadowHandGrasp-v0`)

Pick a cube up from a table.

### Reorientation (`ShadowHandReorient-v0`)

Rotate a cube in hand to match a target orientation.

### Peg-in-Hole (`ShadowHandPeg-v0`)

Grasp a peg, align it with a hole, and insert it.

## Scope

This is a simulation-only benchmark. Observations include the ground-truth
object pose and velocities, which would not be directly available on real
hardware. Sim-to-real transfer would need either a pose estimator in front of
the policy or an asymmetric actor-critic setup (full state to the value
network, restricted observations to the policy).

## Testing

Run the test suite with:

```bash
uv run pytest
```

## Acknowledgments

The Shadow Hand MJCF model is sourced from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
