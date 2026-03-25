import sys

USAGE = """\
Usage: python main.py <command> [options]

  train               Train grasping (PPO)
  train-reorient      Train reorientation (PPO)
  train-peg           Train peg-in-hole (SAC)
  train-tactile       Tactile ablation study
  evaluate            Evaluate grasp model
  evaluate-reorient   Evaluate reorientation model
  evaluate-peg        Evaluate peg model
  evaluate-tactile    Compare tactile vs baseline
"""

COMMANDS = {
    "train":             "scripts.training.train_grasp",
    "train-reorient":    "scripts.training.train_reorient",
    "train-peg":         "scripts.training.train_peg",
    "train-tactile":     "scripts.training.train_tactile",
    "evaluate":          "scripts.evaluation.eval_grasp",
    "evaluate-reorient": "scripts.evaluation.eval_reorient",
    "evaluate-peg":      "scripts.evaluation.eval_peg",
    "evaluate-tactile":  "scripts.evaluation.eval_tactile",
}


def main() -> None:
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n{USAGE}")
        sys.exit(1)

    sys.argv = [sys.argv[0]] + sys.argv[2:]
    module = __import__(COMMANDS[command], fromlist=["main", "parse_args", "train"])

    if hasattr(module, "parse_args"):
        module.train(module.parse_args())
    else:
        module.main()


if __name__ == "__main__":
    main()
