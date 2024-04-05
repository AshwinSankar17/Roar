from typing import List, Optional

from omegaconf import DictConfig

# TODO: resolve import conflicts
from roar.collections.asr.parts.utils import adapter_utils
from roar.collections.tts.modules.aligner import AlignmentEncoder
from roar.collections.tts.modules.fastpitch import TemporalPredictor
from roar.collections.tts.modules.transformer import (
    FFTransformerDecoder,
    FFTransformerEncoder,
)
from roar.core.classes import adapter_mixins


class FFTransformerDecoderAdapter(
    FFTransformerDecoder, adapter_mixins.AdapterModuleMixin
):
    """Inherit from FFTransformerDecoder and add support for adapter"""

    def add_adapter(self, name: str, cfg: dict):
        cfg = self._update_adapter_cfg_input_dim(cfg)
        for fft_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            fft_layer.add_adapter(name, cfg)

    def is_adapter_available(self) -> bool:
        return any([FFT_layer.is_adapter_available() for FFT_layer in self.layers])

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        for FFT_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            FFT_layer.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        for FFT_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            names.update(FFT_layer.get_enabled_adapters())

        names = sorted(list(names))
        return names

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(
            self, cfg, module_dim=self.d_model
        )
        return cfg


class FFTransformerEncoderAdapter(
    FFTransformerDecoderAdapter, FFTransformerEncoder, adapter_mixins.AdapterModuleMixin
):
    """Inherit from FFTransformerEncoder and add support for adapter"""

    pass


class AlignmentEncoderAdapter(AlignmentEncoder, adapter_mixins.AdapterModuleMixin):
    """Inherit from AlignmentEncoder and add support for adapter"""

    def add_adapter(self, name: str, cfg: dict):
        for i, conv_layer in enumerate(self.key_proj):
            if i % 2 == 0:
                cfg = self._update_adapter_cfg_input_dim(
                    cfg, conv_layer.conv.out_channels
                )
                conv_layer.add_adapter(name, cfg)

        for i, conv_layer in enumerate(self.query_proj):
            if i % 2 == 0:
                cfg = self._update_adapter_cfg_input_dim(
                    cfg, conv_layer.conv.out_channels
                )
                conv_layer.add_adapter(name, cfg)

    def is_adapter_available(self) -> bool:
        return any(
            [
                conv_layer.is_adapter_available()
                for i, conv_layer in enumerate(self.key_proj)
                if i % 2 == 0
            ]
            + [
                conv_layer.is_adapter_available()
                for i, conv_layer in enumerate(self.query_proj)
                if i % 2 == 0
            ]
        )

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        for i, conv_layer in enumerate(self.key_proj):
            if i % 2 == 0:
                conv_layer.set_enabled_adapters(name=name, enabled=enabled)
        for i, conv_layer in enumerate(self.query_proj):
            if i % 2 == 0:
                conv_layer.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        for i, conv_layer in enumerate(self.key_proj):
            if i % 2 == 0:
                names.update(conv_layer.get_enabled_adapters())
        for i, conv_layer in enumerate(self.query_proj):
            if i % 2 == 0:
                names.update(conv_layer.get_enabled_adapters())

        names = sorted(list(names))
        return names

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig, module_dim: int):
        cfg = adapter_utils.update_adapter_cfg_input_dim(
            self, cfg, module_dim=module_dim
        )
        return cfg


class TemporalPredictorAdapter(TemporalPredictor, adapter_mixins.AdapterModuleMixin):
    """Inherit from TemporalPredictor and add support for adapter"""

    def add_adapter(self, name: str, cfg: dict):
        cfg = self._update_adapter_cfg_input_dim(cfg)
        for conv_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            conv_layer.add_adapter(name, cfg)

    def is_adapter_available(self) -> bool:
        return any([conv_layer.is_adapter_available() for conv_layer in self.layers])

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        for conv_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            conv_layer.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        for conv_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            names.update(conv_layer.get_enabled_adapters())

        names = sorted(list(names))
        return names

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(
            self, cfg, module_dim=self.filter_size
        )
        return cfg


"""Register any additional information"""
if adapter_mixins.get_registered_adapter(FFTransformerEncoder) is None:
    adapter_mixins.register_adapter(
        base_class=FFTransformerEncoder, adapter_class=FFTransformerEncoderAdapter
    )

if adapter_mixins.get_registered_adapter(FFTransformerDecoder) is None:
    adapter_mixins.register_adapter(
        base_class=FFTransformerDecoder, adapter_class=FFTransformerDecoderAdapter
    )

if adapter_mixins.get_registered_adapter(AlignmentEncoder) is None:
    adapter_mixins.register_adapter(
        base_class=AlignmentEncoder, adapter_class=AlignmentEncoderAdapter
    )

if adapter_mixins.get_registered_adapter(TemporalPredictor) is None:
    adapter_mixins.register_adapter(
        base_class=TemporalPredictor, adapter_class=TemporalPredictorAdapter
    )
