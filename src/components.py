import logging

import torch
from torch.cuda.amp import GradScaler

import util
from .config import unwrap_model
from .data import build_training_dataloader, setup_datasets
from .models import get_model_adapter


def build_training_components(args):
    adapter = get_model_adapter(args.model)
    if args.model in {"boq", "mixvpr"} and args.resume is None and not args.download_pretrained:
        model_name = "BoQ" if args.model == "boq" else "MixVPR"
        raise ValueError(
            f"{model_name} requires initial weights. Pass --resume with a local checkpoint or "
            "pass --download_pretrained to download the official weights."
        )
    if args.resume is None and args.download_pretrained and adapter.download_weights is None:
        raise ValueError(f"Model {args.model!r} does not support --download_pretrained")

    val_ds, test_ds = setup_datasets(args)

    model_bundle = adapter.build(args)
    model = model_bundle.model.to(args.device)
    args.features_dim = model_bundle.descriptor_dim

    if args.parallel > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=list(range(args.parallel)))
        if args.model == "supervlad" and torch.cuda.device_count() >= 2:
            from model.sync_batchnorm import convert_model

            model = convert_model(model)
            model = model.cuda()
    else:
        model = torch.nn.DataParallel(model)

    if args.optim == "adam":
        optimizer_kwargs = {"lr": args.lr}
        if args.weight_decay is not None:
            optimizer_kwargs["weight_decay"] = args.weight_decay
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    elif args.optim == "sgd":
        weight_decay = 0.001 if args.weight_decay is None else args.weight_decay
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)
    elif args.optim == "adamw":
        weight_decay = 9.5e-9 if args.weight_decay is None else args.weight_decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer {args.optim!r}")

    best_score = -1.0
    not_improved = 0
    start_epoch = 0
    if args.resume:
        if args.resume_model_only or not args.continue_training:
            adapter.load_weights(unwrap_model(model), args.resume, args)
            logging.info(
                "Loaded model weights from %s and reset optimizer, epoch, and early-stopping state.",
                args.resume,
            )
        else:
            model, optimizer, best_score, start_epoch, not_improved = util.resume_train(
                args,
                model,
                optimizer,
            )
            logging.info("Resuming from epoch %d with best validation score %.1f", start_epoch, best_score)
    elif args.download_pretrained:
        adapter.download_weights(unwrap_model(model), args)
        logging.info("Downloaded and loaded pretrained weights for %s.", args.model)
    scaler = GradScaler(enabled=args.mixed_precision)
    train_loader = build_training_dataloader(args)
    return model, optimizer, scaler, train_loader, val_ds, test_ds, best_score, start_epoch, not_improved
