import os
import runpy
import sys

import ray

MILES_RAY_RUNTIME_ENV = {
    "env_vars": {
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "TENSORBOARD_DIR": os.environ.get("TENSORBOARD_DIR", ""),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "DEPRECATED_MEGATRON_COMPATIBLE": "1",  # 兼容旧版 Megatron，使用 Uni-OPD conda 环境的时候必须设置
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "INFO",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",  # 关闭一个 FutureWarning
        "http_proxy": "",
        "https_proxy": "",
    },
}


def get_miles_ray_runtime_env() -> dict:
    """
    构建 Miles Ray runtime 环境变量。
    优先使用 host 进程中已设置的环境变量，否则使用默认值。
    """
    runtime_env = {"env_vars": MILES_RAY_RUNTIME_ENV["env_vars"].copy()}

    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"][key] = os.environ[key]

    return runtime_env


def main():
    if not ray.is_initialized():
        address = os.getenv("RAY_ADDRESS", "auto")
        runtime_env: dict = get_miles_ray_runtime_env()

        print(f"[RayLauncher] Connecting to Ray cluster → {address}")
        print(f"[RayLauncher] Runtime env vars: {runtime_env['env_vars']}")
        ray.init(address=address, runtime_env=runtime_env)

        print("[RayLauncher] Ray initialized successfully")
    else:
        print("[RayLauncher] Ray already initialized")
    print("[RayLauncher] Cluster resources:", ray.cluster_resources())

    if len(sys.argv) < 2:
        print("Usage: python ray_launcher.py <script.py> [args...]")
        sys.exit(1)

    script = sys.argv[1]
    sys.argv = sys.argv[1:]

    print(f"[RayLauncher] Running script → {script}")
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
