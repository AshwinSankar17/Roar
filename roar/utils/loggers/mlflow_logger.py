from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MLFlowParams:
    # name of experiment, if none, defaults to the globally set experiment name
    experiment_name: Optional[str] = None
    # no run_name because it's set by version
    # local or remote tracking seerver. If tracking_uri is not set, it defaults to save_dir
    tracking_uri: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    save_dir: Optional[str] = "./mlruns"
    prefix: str = ""
    artifact_location: Optional[str] = None
    # provide run_id if resuming a previously started run
    run_id: Optional[str] = None
