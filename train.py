import argparse


def main():
    parser = argparse.ArgumentParser(description="MRI-GNP hierarchical multitask trainer")
    parser.add_argument(
        "--experiment-config",
        "-c",
        required=True,
        help="Path to the experiment YAML file.",
    )
    parser.add_argument(
        "--global-config",
        "-g",
        required=True,
        help="Path to the global YAML file.",
    )
    parser.add_argument(
        "--run-on-which-gpu",
        type=int,
        default=None,
        help="Optional CUDA device override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load one batch and run a single forward pass without training.",
    )
    args = parser.parse_args()

    from mri_gnp.config import load_configs
    from mri_gnp.trainer import train_pipeline

    global_config, experiment_config = load_configs(
        global_config_path=args.global_config,
        experiment_config_path=args.experiment_config,
    )
    train_pipeline(
        global_config=global_config,
        experiment_config=experiment_config,
        gpu_override=args.run_on_which_gpu,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
