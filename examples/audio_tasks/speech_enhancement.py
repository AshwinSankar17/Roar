import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from roar.collections.asr.models import EncMaskDecAudioToAudioModel
from roar.core.config import hydra_runner
from roar.utils import logging
from roar.utils.exp_manager import exp_manager


@hydra_runner(config_path="./conf", config_name="masking")
def main(cfg):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg, resolve=True)}")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = EncMaskDecAudioToAudioModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    model.maybe_init_from_pretrained_checkpoint(cfg)

    # Train the model
    trainer.fit(model)

    # Run on test data, if available
    if (
        hasattr(cfg.model, "test_ds")
        and cfg.model.test_ds.manifest_filepath is not None
    ):
        if trainer.is_global_zero:
            # Destroy the current process group and let the trainer initialize it again with a single device.
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

            # Run test on a single device
            trainer = pl.Trainer(devices=1, accelerator=cfg.trainer.accelerator)
            if model.prepare_test(trainer):
                trainer.test(model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
