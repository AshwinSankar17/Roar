import pytorch_lightning as pl

from roar.collections.common.callbacks import LogEpochTimeCallback
from roar.collections.tts.models import AlignerModel
from roar.core.config import hydra_runner
from roar.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="aligner")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = AlignerModel(cfg=cfg.model, trainer=trainer)
    trainer.callbacks.extend(
        [pl.callbacks.LearningRateMonitor(), LogEpochTimeCallback()]
    )  # noqa
    trainer.fit(model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
