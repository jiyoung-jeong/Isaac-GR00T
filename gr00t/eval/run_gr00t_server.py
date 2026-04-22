from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
from typing import Literal

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


def _load_replace_dit_with_tensorrt():
    """Load TensorRT replacement helper from deployment script file path.

    We load by file path because `scripts/` is not a Python package.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "deployment" / "standalone_inference_script.py"
    spec = importlib.util.spec_from_file_location("standalone_inference_script", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load TensorRT helper from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.replace_dit_with_tensorrt


@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    """Inference mode for model_path policy."""

    trt_engine_path: str | None = None
    """TensorRT engine path (.trt), required when inference_mode='tensorrt'."""


def main(config: ServerConfig):
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # check if the model path exists
    if config.model_path and config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Create and start the server
    if config.model_path is not None:
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
        if config.inference_mode == "tensorrt":
            if not config.trt_engine_path:
                raise ValueError("trt_engine_path is required when inference_mode='tensorrt'")
            if not os.path.exists(config.trt_engine_path):
                raise FileNotFoundError(f"TensorRT engine path {config.trt_engine_path} does not exist")
            if not str(config.device).startswith("cuda"):
                raise ValueError("TensorRT mode requires a CUDA device")

            # Local load to avoid hard dependency when running in pure PyTorch mode.
            replace_dit_with_tensorrt = _load_replace_dit_with_tensorrt()

            device_idx = 0
            if ":" in str(config.device):
                device_idx = int(str(config.device).split(":")[1])
            replace_dit_with_tensorrt(policy, config.trt_engine_path, device=device_idx)
    elif config.dataset_path is not None:
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
