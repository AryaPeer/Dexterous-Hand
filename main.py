import sys

USAGE = """\
Usage: python main.py <command> [options]

  train               Train grasping (PPO)
  train-reorient      Train reorientation (PPO)
  train-peg           Train peg-in-hole (SAC)
  evaluate            Evaluate grasp model
  evaluate-reorient   Evaluate reorientation model
  evaluate-peg        Evaluate peg model

  train-grasp-mjx     Train grasping on MJX (SBX PPO)
  train-reorient-mjx  Train reorientation on MJX (SBX PPO)
  train-peg-mjx       Train peg-in-hole on MJX (SBX PPO)
  resume-grasp-mjx    Resume grasping from a saved checkpoint
  resume-reorient-mjx Resume reorientation from a saved checkpoint
  resume-peg-mjx      Resume peg-in-hole from a saved checkpoint
  evaluate-grasp-mjx     Evaluate SBX grasp model
  evaluate-reorient-mjx  Evaluate SBX reorientation model
  evaluate-peg-mjx       Evaluate SBX peg model
"""

COMMANDS = {
    "train": "scripts.training.cpu.train_grasp",
    "train-reorient": "scripts.training.cpu.train_reorient",
    "train-peg": "scripts.training.cpu.train_peg",
    "evaluate": "scripts.evaluation.cpu.eval_grasp",
    "evaluate-reorient": "scripts.evaluation.cpu.eval_reorient",
    "evaluate-peg": "scripts.evaluation.cpu.eval_peg",
    "train-grasp-mjx": "scripts.training.gpu.train_grasp",
    "train-reorient-mjx": "scripts.training.gpu.train_reorient",
    "train-peg-mjx": "scripts.training.gpu.train_peg",
    "resume-grasp-mjx": "scripts.training.gpu.resume_grasp",
    "resume-reorient-mjx": "scripts.training.gpu.resume_reorient",
    "resume-peg-mjx": "scripts.training.gpu.resume_peg",
    "evaluate-grasp-mjx": "scripts.evaluation.gpu.eval_grasp",
    "evaluate-reorient-mjx": "scripts.evaluation.gpu.eval_reorient",
    "evaluate-peg-mjx": "scripts.evaluation.gpu.eval_peg",
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
