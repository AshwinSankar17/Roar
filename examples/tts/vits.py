import pytorch_lightning as pl

from roar.collections.tts.models.vits import VitsModel
from roar.core.config import hydra_runner
from roar.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf/vits", config_name="vits")
def main(cfg):
    trainer = pl.Trainer(use_distributed_sampler=False, **cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = VitsModel(cfg=cfg.model, trainer=trainer)

    trainer.callbacks.extend([pl.callbacks.LearningRateMonitor()])
    trainer.fit(model)


if __name__ == '__main__':
    main()