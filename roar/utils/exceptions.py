class RoarBaseException(Exception):
    """ NeMo Base Exception. All exceptions created in NeMo should inherit from this class"""


class LightningNotInstalledException(RoarBaseException):
    def __init__(self, obj):
        message = (
            f" You are trying to use {obj} without installing all of pytorch_lightning, hydra, and "
            f"omegaconf. Please install those packages before trying to access {obj}."
        )
        super().__init__(message)


class CheckInstall:
    def __init__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)

    def __call__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)

    def __getattr__(self, *args, **kwargs):
        raise LightningNotInstalledException(self)
