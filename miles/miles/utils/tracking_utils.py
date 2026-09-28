import logging
from argparse import Namespace

import wandb

from miles.utils.tensorboard_utils import _TensorboardAdapter

from . import wandb_utils

logger = logging.getLogger(__name__)


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args: Namespace, metrics: dict, step_key: str):
    if args.use_wandb:
        try:
            wandb.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics to WandB: {e}")

    if args.use_tensorboard:
        try:
            metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
            _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
        except Exception as e:
            logger.warning(f"Failed to log metrics to TensorBoard: {e}")
