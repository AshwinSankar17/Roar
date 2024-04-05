import subprocess


def get_gpu_compute_capability():
    """Returns the compute capability of the GPU in use."""

    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"]
    ).decode("utf-8")
    compute_cap = output.split(",")[0]
    return compute_cap


def is_gpu_ampere_or_newer():
    """Returns True if the GPU is Ampere or newer, False otherwise."""

    compute_cap = get_gpu_compute_capability()
    major_version = int(compute_cap.split(".")[0])

    return major_version >= 8
