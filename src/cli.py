import parser as parser_module

from .config import SUPPORTED_ATTACK_NAMES, UNSUPPORTED_ATTACK_NAMES
from .models import add_model_arguments, model_names

REQUIRED_RECALL_VALUES = (1, 5, 10, 100)


def parse_attack_names(attack_strings):
    attack_names = []
    for attack_string in attack_strings:
        attack_name = attack_string.split("(", 1)[0].strip()
        attack_names.append(attack_name)
        if attack_name in UNSUPPORTED_ATTACK_NAMES:
            raise NotImplementedError(
                f"{attack_name} is not supported in train.py because it relies on "
                "classification-specific AutoAttack behavior."
            )
        if attack_name not in SUPPORTED_ATTACK_NAMES:
            raise ValueError(
                f"Unsupported attack {attack_name!r}. Supported attacks are: "
                f"{sorted(SUPPORTED_ATTACK_NAMES)}"
            )
    return attack_names


def build_parser():
    parser = parser_module.build_parser()
    parser.description = "Perceptual adversarial training for visual place recognition models"
    parser.add_argument(
        "--model",
        choices=model_names(),
        default="supervlad",
        help="Descriptor model to train.",
    )
    parser.add_argument(
        "--train_resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Training image size. Defaults to --resize.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Optimizer weight decay. Model-compatible optimizer defaults are retained when omitted.",
    )
    parser.add_argument(
        "--download_pretrained",
        action="store_true",
        help="Download official initial weights when --resume is not supplied.",
    )
    parser.add_argument(
        "--attack",
        type=str,
        action="append",
        default=[],
        help="Attack expression(s) to harden against, following the perceptual-advex style.",
    )
    parser.add_argument("--num_epochs", dest="epochs_num", type=int, default=None, help="Number of training epochs.")
    parser.add_argument(
        "--batch_size",
        dest="train_batch_size",
        type=int,
        default=None,
        help="Training batch size alias kept for perceptual-advex style.",
    )
    parser.add_argument("--val_batches", type=int, default=10, help="Number of validation query batches to attack.")
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=None,
        help="Training batches per (virtual) epoch. Defaults to one pass over the full dataset.",
    )
    parser.add_argument("--log_dir", type=str, default="logs", help="Base folder for perceptual adversarial training runs.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of GPUs to use when CUDA is available.")
    parser.add_argument(
        "--only_attack_correct",
        action="store_true",
        default=False,
        help="Attack only queries that are cleanly retrieved before the attack.",
    )
    parser.add_argument(
        "--randomize_attack",
        action="store_true",
        default=False,
        help="Use one random configured attack per training step.",
    )
    parser.add_argument(
        "--maximize_attack",
        action="store_true",
        default=False,
        help="Use the strongest configured attack loss for each training step.",
    )
    parser.add_argument("--continue", dest="continue_training", action="store_true", default=False)
    parser.add_argument("--keep_every", type=int, default=1, help="Keep one intermediate checkpoint every N epochs.")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Clip gradients to this absolute value.")
    parser.add_argument("--lpips_model", type=str, default=None, help="Optional LPIPS model override.")
    parser.add_argument("--lr_schedule", type=str, default=None, help="Epochs when the learning rate drops by 10x.")
    parser.add_argument(
        "--lr_plateau_patience",
        type=int,
        default=None,
        help="Drop the learning rate after this many epochs without validation improvement. Overrides --lr_schedule.",
    )
    parser.add_argument(
        "--lr_plateau_factor",
        type=float,
        default=0.1,
        help="Multiplicative learning-rate factor applied on plateau.",
    )
    parser.add_argument(
        "--resume_model_only",
        action="store_true",
        help="Load only model weights from --resume and reset optimizer and epoch state.",
    )
    parser.add_argument(
        "--adv_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the adversarial rank loss.",
    )
    parser.add_argument(
        "--adv_align_weight",
        type=float,
        default=0.05,
        help="Weight for the descriptor alignment loss.",
    )
    parser.add_argument(
        "--adv_negatives",
        type=int,
        default=5,
        help="Number of hard negatives to use for retrieval attacks.",
    )
    parser.add_argument(
        "--adv_warmup_epochs",
        type=int,
        default=1,
        help="Number of initial epochs that train only on the clean loss.",
    )
    parser.add_argument(
        "--adv_margin",
        type=float,
        default=0.1,
        help="Margin used for the retrieval adversarial objective.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum score increase needed to refresh patience.",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help="TensorBoard log directory. Defaults to <save_dir>/tensorboard.",
    )
    parser.add_argument(
        "--gsv_cities_base_path",
        type=str,
        default=None,
        help="Path to the GSV-Cities training dataset.",
    )
    parser.add_argument(
        "--skip_initial_validation",
        action="store_true",
        default=False,
        help="Skip baseline validation before the first training epoch.",
    )
    add_model_arguments(parser)
    return parser


def parse_arguments(argv=None):
    args = build_parser().parse_args(argv)
    args = parser_module.validate_arguments(args)

    if args.train_batch_size is None:
        args.train_batch_size = 60
    if args.epochs_num is None:
        args.epochs_num = 15
    if args.lr_schedule is None:
        if args.optim == "sgd":
            args.lr_schedule = "30,60,80"
        else:
            args.lr_schedule = "120"
    if args.keep_every < 1:
        raise ValueError("--keep_every must be at least 1")
    if args.batches_per_epoch is not None and args.batches_per_epoch < 1:
        raise ValueError("--batches_per_epoch must be at least 1")
    if args.lr_plateau_patience is not None and args.lr_plateau_patience < 1:
        raise ValueError("--lr_plateau_patience must be at least 1")
    if not 0.0 < args.lr_plateau_factor < 1.0:
        raise ValueError("--lr_plateau_factor must be between 0 and 1 (exclusive)")
    if args.adv_negatives < 1:
        raise ValueError("--adv_negatives must be at least 1")
    if args.adv_warmup_epochs < 0:
        raise ValueError("--adv_warmup_epochs must be non-negative")
    if args.clip_grad <= 0:
        raise ValueError("--clip_grad must be positive")
    if args.weight_decay is not None and args.weight_decay < 0:
        raise ValueError("--weight_decay must be non-negative")
    if args.train_resize is None:
        args.train_resize = list(args.resize)
    if any(dimension <= 0 for dimension in args.train_resize):
        raise ValueError("--train_resize dimensions must be positive")
    if args.model == "boq":
        from .models.boq import resolve_descriptor_dimension

        args.boq_descriptors_dimension = resolve_descriptor_dimension(args)
    elif args.model == "mixvpr":
        from .models.mixvpr import resolve_descriptor_dimension

        args.mixvpr_descriptors_dimension = resolve_descriptor_dimension(args)

    args.recall_values = list(dict.fromkeys([*args.recall_values, *REQUIRED_RECALL_VALUES]))
    parse_attack_names(args.attack)
    return args
