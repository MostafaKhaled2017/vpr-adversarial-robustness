import logging
import multiprocessing
from datetime import datetime
from pathlib import Path

import commons
import torch

from src.attacks import instantiate_attacks
from src.checkpoints import (
    append_resume_history,
    maybe_copy_resume_checkpoint,
    resolve_run_directory,
    save_training_config,
    write_run_status,
)
from src.cli import parse_arguments
from src.components import build_training_components
from src.config import create_summary_writer, validate_cuda_runtime
from src.faiss_utils import validate_faiss_runtime
from src.losses import configure_metric_learning
from src.train_loop import run_training


def main():
    args = parse_arguments()
    validate_cuda_runtime(args)
    validate_faiss_runtime(args.device)

    start_time = datetime.now()
    timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = str(resolve_run_directory(args.log_dir, args.save_dir, args.run_dir, timestamp))
    args.tensorboard_dir = args.tensorboard_dir or str(Path(args.save_dir) / "tensorboard")

    commons.setup_logging(args.save_dir, allow_existing=args.run_dir is not None)
    commons.make_deterministic(args.seed)
    configure_metric_learning()

    config_filename = "training_config.yaml"
    if args.continue_training and (Path(args.save_dir) / config_filename).exists():
        config_filename = f"resume_config_{timestamp}.yaml"
    config_path = save_training_config(args, filename=config_filename)

    logging.info("Arguments: %s", args)
    logging.info("The outputs are being saved in %s", args.save_dir)
    logging.info("Training configuration saved to %s", config_path)
    logging.info("TensorBoard logs will be written to %s", args.tensorboard_dir)
    logging.info("Using %d GPUs and %d CPUs", torch.cuda.device_count(), multiprocessing.cpu_count())

    writer = create_summary_writer(args.tensorboard_dir)
    write_run_status(args.save_dir, "running", resume_checkpoint=args.resume if args.continue_training else None)
    try:
        (
            model,
            optimizer,
            scaler,
            train_loader,
            val_ds,
            test_ds,
            best_score,
            start_epoch,
            not_improved,
            resume_runtime_state,
        ) = build_training_components(args)
        if args.continue_training:
            append_resume_history(args.save_dir, args.resume, start_epoch)
        maybe_copy_resume_checkpoint(args, model)

        train_attacks = instantiate_attacks(model, args.attack, args)
        validation_attacks = [instantiate_attacks(model, [attack_string], args)[0] for attack_string in args.attack]
        outcome = run_training(
            args,
            model,
            optimizer,
            scaler,
            train_loader,
            val_ds,
            test_ds,
            best_score,
            start_epoch,
            not_improved,
            writer,
            train_attacks,
            validation_attacks,
            resume_runtime_state,
        )
        write_run_status(args.save_dir, outcome["state"], final_epoch=outcome["final_epoch"])
    except BaseException as exc:
        write_run_status(
            args.save_dir,
            "interrupted",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
