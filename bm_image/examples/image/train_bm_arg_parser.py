"""Argument parser for Bridge Matching (BM) training.

Extends the original Flow Matching parser with BM-specific options
(target type, beta, lambda_d, etc.) so we share argument names and
defaults with the FM trainer wherever possible.
"""

from train_arg_parser import get_args_parser as _get_fm_args_parser


_BM_TARGETS = (
    "cbm_diffusion",
    "bm_diffusion_conditional",
    "bm_linear_marginal",
    "bm_linear_marginal_ot",
    "cfm_linear",
    "cfm_diffusion",
)


def get_args_parser():
    parser = _get_fm_args_parser()

    bm_group = parser.add_argument_group("Bridge Matching")
    bm_group.add_argument(
        "--target_type",
        default="cbm_diffusion",
        choices=list(_BM_TARGETS),
        help="BM target. cfm_* keeps d=0; all other targets train both u and d.",
    )
    bm_group.add_argument("--beta_value", type=float, default=1e-2)
    # Keep defaults aligned with BM target implementation for numerical stability.
    bm_group.add_argument("--sigma_floor", type=float, default=5e-2)
    bm_group.add_argument("--t_eps", type=float, default=1e-4)
    bm_group.add_argument(
        "--save_frequency",
        type=int,
        default=-1,
        help="Checkpoint save frequency in epochs. -1 follows eval_frequency; 0 disables periodic checkpointing.",
    )
    bm_group.add_argument("--lambda_d", type=float, default=1.0)
    bm_group.add_argument(
        "--lambda_eval_u",
        type=float,
        default=1.0,
        help="Scale applied to the u-flow during BM evaluation.",
    )
    bm_group.add_argument(
        "--lambda_eval_d",
        type=float,
        default=1.0,
        help="Scale applied to the d-flow during BM evaluation.",
    )
    bm_group.add_argument(
        "--compute_kid",
        action="store_true",
        help="Compute KID during BM evaluation.",
    )
    bm_group.add_argument(
        "--compute_inception_score",
        action="store_true",
        help="Compute Inception Score during BM evaluation.",
    )
    bm_group.add_argument(
        "--inception_splits",
        type=int,
        default=10,
        help="Number of splits for KID and Inception Score.",
    )
    bm_group.add_argument(
        "--lambda_forward_align",
        type=float,
        default=0.0,
        help="Weight for matching u_pred + d_pred to u* + d*.",
    )
    bm_group.add_argument("--posterior_temperature", type=float, default=1.0)
    bm_group.add_argument(
        "--posterior_topk",
        type=int,
        default=0,
        help="0 disables top-k posterior subsetting.",
    )
    bm_group.add_argument("--cfm_beta_min", type=float, default=0.1)
    bm_group.add_argument("--cfm_beta_max", type=float, default=20.0)
    bm_group.add_argument(
        "--bm_repo_path",
        default="",
        help="Override path to the BM repo (folder containing 'bridge_matching/').",
    )

    return parser
