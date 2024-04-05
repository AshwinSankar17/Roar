import pytorch_lightning as pl

from roar.collections.common.callbacks import LogEpochTimeCallback
from roar.collections.tts.models import FastPitchModel
from roar.core.config import hydra_runner
from roar.utils import logging
from roar.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf/fastpitch", config_name="fastpitch_44100_align")
def main(cfg):
    if hasattr(cfg.model.optim, "sched"):
        logging.warning(
            "You are using an optimizer scheduler while finetuning. Are you sure this is intended?"
        )
    if cfg.model.optim.lr > 1e-3 or cfg.model.optim.lr < 1e-5:
        logging.warning("The recommended learning rate for finetuning is 2e-4")
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = FastPitchModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
