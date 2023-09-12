import pytorch_lightning as pl

from roar.collections.tts.models import HifiGanModel
from roar.core.config import hydra_runner
from roar.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf/hifigan", config_name="hifigan_44100")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    trainer.fit(model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
