import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <command>")
        print()
        print("Training:")
        print("  train               Train grasping policy (PPO)")
        print("  train-reorient      Train reorientation policy (PPO)")
        print("  train-peg           Train peg-in-hole policy (SAC)")
        print("  train-tactile       Train tactile ablation study")
        print()
        print("Evaluation:")
        print("  evaluate            Evaluate grasping model")
        print("  evaluate-reorient   Evaluate reorientation model")
        print("  evaluate-peg        Evaluate peg-in-hole model")
        print("  evaluate-tactile    Compare tactile vs baseline")
        print()
        print("Testing:")
        print("  uv run pytest       Run test suite")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "train":
        from scripts.training.train_grasp import parse_args, train

        config = parse_args()
        train(config)
    elif command == "train-reorient":
        from scripts.training.train_reorient import parse_args, train

        config = parse_args()
        train(config)
    elif command == "train-peg":
        from scripts.training.train_peg import parse_args, train

        config = parse_args()
        train(config)
    elif command == "train-tactile":
        from scripts.training.train_tactile import main as tactile_main

        tactile_main()
    elif command == "evaluate":
        from scripts.evaluation.eval_grasp import main as eval_main

        eval_main()
    elif command == "evaluate-reorient":
        from scripts.evaluation.eval_reorient import main as eval_main

        eval_main()
    elif command == "evaluate-peg":
        from scripts.evaluation.eval_peg import main as eval_main

        eval_main()
    elif command == "evaluate-tactile":
        from scripts.evaluation.eval_tactile import main as eval_main

        eval_main()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
