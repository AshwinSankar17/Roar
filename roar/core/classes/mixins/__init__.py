from roar.core.classes.mixins.access_mixins import AccessMixin, set_access_cfg
from roar.core.classes.mixins.adapter_mixin_strategies import (
    ResidualAddAdapterStrategy,
    ResidualAddAdapterStrategyConfig,
    ReturnResultAdapterStrategy,
    ReturnResultAdapterStrategyConfig,
)
from roar.core.classes.mixins.adapter_mixins import (
    AdapterModelPTMixin,
    AdapterModuleMixin,
    get_registered_adapter,
    register_adapter,
)
