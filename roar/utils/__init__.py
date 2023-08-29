from roar.utils.app_state import AppState
from roar.utils.cast_utils import (
    CastToFloat,
    CastToFloatAll,
    avoid_bfloat16_autocast_context,
    avoid_float16_autocast_context,
    cast_all,
    cast_tensor,
)
from roar.utils.roar_logging import Logger as _Logger
from roar.utils.roar_logging import LogMode as logging_mode

logging = _Logger()
try:
    from roar.utils.lightning_logger_patch import add_memory_handlers_to_pl_logger
    #TODO: Formalize the patch
    add_memory_handlers_to_pl_logger()
except ModuleNotFoundError:
    pass
