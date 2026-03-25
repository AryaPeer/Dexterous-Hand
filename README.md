# Dexterous Hand RL

Training a simulated Shadow Hand to solve contact-rich manipulation tasks with reinforcement learning. Built with MuJoCo 3, Stable-Baselines3, and Gymnasium.

The hand has 24 degrees of freedom and learns three tasks of increasing difficulty: grasping objects, reorienting them in hand, and peg-in-hole insertion. The project also includes a tactile sensing ablation that adds simulated touch sensors to the fingertips.

## Project Structure

```text
dexterous_hand/
├── envs/               # Gymnasium environments for each task
├── rewards/            # Multi-component shaped reward functions
├── tactile/            # Tactile sensor simulation and CNN encoder
├── curriculum/         # Curriculum learning callbacks
├── utils/              # MuJoCo helpers and quaternion math
└── config.py           # Centralized configuration via dataclasses
scripts/
├── training/           # Task-specific training scripts
└── evaluation/         # Task-specific evaluation scripts
assets/
└── shadow_hand/        # MJCF model and mesh files
tests/                  # Unit tests
main.py                 # CLI entrypoint
```

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

## Running Commands

Use `uv run` to execute project commands inside the managed environment. This keeps the workflow consistent and avoids mixing global Python installs with project-managed dependencies.

General pattern:

```bash
uv run python main.py <command> [options]
```

## Training

All training goes through `main.py`. Each task has its own command and default hyperparameters.

```bash
# Grasping with PPO
uv run python main.py train --n-envs 4 --total-timesteps 50000000

# In-hand reorientation with PPO
uv run python main.py train-reorient --n-envs 4 --total-timesteps 200000000

# Peg-in-hole with SAC
uv run python main.py train-peg --n-envs 4 --total-timesteps 100000000

# Tactile ablation for peg insertion
uv run python main.py train-tactile --n-envs 4 --variant both
```

Notes:

* Increase `--n-envs` if you have more CPU cores available.
* Training outputs and saved models are written to `runs/`.

## Evaluation

```bash
uv run python main.py evaluate
uv run python main.py evaluate-reorient
uv run python main.py evaluate-peg
uv run python main.py evaluate-tactile
```

## Tasks

### Grasping (`ShadowHandGrasp-v0`)

Pick up one of several object types from a table. The environment uses 99-dimensional observations and 20-dimensional actions. The reward is shaped across four phases: reach, grasp, lift, and hold.

### Reorientation (`ShadowHandReorient-v0`)

Rotate a cube in hand to match a target orientation. The task uses curriculum learning, starting with easier rotations and progressing toward larger target rotations. The environment uses 115-dimensional observations.

### Peg-in-Hole (`ShadowHandPeg-v0`)

Grasp a peg, align it with a hole, and insert it. The task uses a four-stage curriculum that tightens insertion clearance and removes early assistance over time. The environment uses 125-dimensional observations. SAC is used here for better sample efficiency on this task.

### Peg-in-Hole with Tactile (`ShadowHandPegTactile-v0`)

This variant adds simulated 4x4 fingertip taxel grids. Tactile observations are processed through a small CNN before being passed to the policy. The full observation size is 365 dimensions. This task is used to compare insertion performance with and without tactile feedback.

## Testing

Run the test suite with:

```bash
uv run pytest
```

## Acknowledgments

The Shadow Hand MJCF model is sourced from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
