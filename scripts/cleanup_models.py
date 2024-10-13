import argparse
import re
from pathlib import Path


def get_non_highest_checkpoint_paths(
    search_path, checkpoint_template, keep_one_best=False
):
    full_template = f"*{checkpoint_template}*.ckpt"
    checkpoint_files = list(search_path.glob(full_template))

    if not checkpoint_files:
        return []
    checkpoint_files = sorted(
        checkpoint_files,
        key=lambda f: float(
            re.search(
                rf"{checkpoint_template}-(\d+\.\d+)(-v\d+)?\.ckpt$",
                str(f.name),
            ).group(1)  # type: ignore
            if re.search(
                rf"{checkpoint_template}-(\d+\.\d+)(-v\d+)?\.ckpt$",
                str(f.name),
            )
            else 0.0
        ),
        reverse=True,
    )

    # Find the maximum score
    max_score = float(
        re.search(
            rf"{checkpoint_template}-(\d+\.\d+)(-v\d+)?\.ckpt$",
            str(checkpoint_files[0].name),
        ).group(1)  # type: ignore
    )

    # Keep all models with the maximum score
    highest_checkpoints = [
        f
        for f in checkpoint_files
        if float(
            re.search(
                rf"{checkpoint_template}-(\d+\.\d+)(-v\d+)?\.ckpt$",
                str(f.name),
            ).group(1)  # type: ignore
        )
        == max_score
    ]

    if keep_one_best:
        # Keep only one of the best models
        highest_checkpoints = [highest_checkpoints[0]]

    # Return all files except the ones with the highest score
    return [f for f in checkpoint_files if f not in highest_checkpoints]


def main():
    parser = argparse.ArgumentParser(description="Cleanup models script.")
    parser.add_argument("--real_run", action="store_true", help="Dry run mode.")
    parser.add_argument(
        "--keep_one_best",
        action="store_true",
        help="Keep only one of the best models.",
    )
    parser.add_argument(
        "--print_num_wandb_runs_in_folder",
        action="store_true",
        help="Print the number of wandb runs in the folder.",
    )
    args = parser.parse_args()

    dry_run = not args.real_run
    search_path = Path(".")
    search_paths = [
        search_path / "outputs",
        search_path / "outputs_old",
        search_path / "cross_validation_old",
        search_path / "cross_validation_runs_old",
        search_path / "emnlp24_outputs" / "outputs",
        search_path / "synced_outputs",
    ]

    checkpoint_templates = [
        "highest_val_all_AUROC",
        "highest_classless_accuracy_val_average",
        "highest_classless_accuracy_val_weighted_average",
        "highest_balanced_accuracy_val_weighted_average",
    ]
    for search_path in search_paths:
        for checkpoint_template in checkpoint_templates:
            total_sizes = []
            for subfolder in search_path.glob(pattern="*"):
                if subfolder.is_dir():
                    for sub_subfolder in subfolder.glob(pattern="fold_index=*"):
                        if sub_subfolder.is_dir():
                            non_highest_checkpoints = get_non_highest_checkpoint_paths(
                                search_path=sub_subfolder,
                                checkpoint_template=checkpoint_template,
                                keep_one_best=args.keep_one_best,
                            )
                            for checkpoint in non_highest_checkpoints:
                                size = checkpoint.stat().st_size / (1024 * 1024 * 1024)
                                total_sizes.append(size)
                                if not dry_run:
                                    # delete the checkpoint
                                    checkpoint.unlink()
            if total_sizes:
                print(
                    f"Total size of non-highest checkpoints for {checkpoint_template} {search_path}: {round(sum(total_sizes),2)} GB (total {len(total_sizes)} files)"
                )

    if args.print_num_wandb_runs_in_folder:
        for search_path in search_paths:
            for subfolder in search_path.glob("*"):
                if subfolder.is_dir():
                    for sub_subfolder in subfolder.glob("fold_index=*"):
                        if sub_subfolder.is_dir():
                            count = len(list(sub_subfolder.glob("wandb/*run*")))
                            print(sub_subfolder, count)


if __name__ == "__main__":
    main()
