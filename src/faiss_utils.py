from dataclasses import dataclass
from importlib import metadata
from typing import Any, Optional

import numpy as np
import torch


GPU_FAISS_PACKAGE = "faiss-gpu-cu12==1.14.1.post1"

# StandardGpuResources defaults to a 1.5 GiB contiguous scratch allocation,
# which fails alongside PyTorch's reserved pools on smaller GPUs. Flat L2
# searches at our batch sizes tile fine within a much smaller scratch buffer.
FAISS_GPU_TEMP_MEMORY_BYTES = 256 * 1024 * 1024


@dataclass
class FlatL2Index:
    index: Any
    gpu_resources: Optional[Any] = None

    @property
    def uses_gpu(self) -> bool:
        return self.gpu_resources is not None

    def add(self, features: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(features.astype(np.float32, copy=False)))

    def search(self, features: np.ndarray, k: int):
        return self.index.search(
            np.ascontiguousarray(features.astype(np.float32, copy=False)),
            k,
        )


def _import_faiss():
    import faiss

    return faiss


def _installed_faiss_distributions() -> str:
    installed = []
    for distribution_name in ("faiss-cpu", "faiss-gpu-cu12", "faiss-gpu"):
        try:
            installed.append(f"{distribution_name}=={metadata.version(distribution_name)}")
        except metadata.PackageNotFoundError:
            continue
    return ", ".join(installed) if installed else "none detected"


def _gpu_error(detail: str) -> RuntimeError:
    return RuntimeError(
        "FAISS GPU is required because --device cuda was requested, but it is not usable: "
        f"{detail}. Installed FAISS distributions: {_installed_faiss_distributions()}. "
        "Remove conflicting CPU/GPU FAISS distributions and install only "
        f"{GPU_FAISS_PACKAGE}, or rerun with --device cpu."
    )


def _create_gpu_flat_l2_index(faiss, dimension: int) -> FlatL2Index:
    required_apis = ("StandardGpuResources", "index_cpu_to_gpu", "get_num_gpus")
    missing_apis = [name for name in required_apis if not hasattr(faiss, name)]
    if missing_apis:
        raise _gpu_error(f"the imported faiss module is missing {', '.join(missing_apis)}")

    try:
        gpu_count = int(faiss.get_num_gpus())
    except Exception as exc:
        raise _gpu_error(f"faiss.get_num_gpus() failed with {exc!r}") from exc
    if gpu_count < 1:
        raise _gpu_error("faiss.get_num_gpus() reported zero CUDA devices")

    try:
        device_index = int(torch.cuda.current_device())
    except Exception as exc:
        raise _gpu_error(f"PyTorch could not select the current CUDA device: {exc!r}") from exc
    if device_index >= gpu_count:
        raise _gpu_error(
            f"PyTorch selected CUDA device {device_index}, but FAISS sees only {gpu_count} device(s)"
        )

    try:
        resources = faiss.StandardGpuResources()
        resources.setTempMemory(FAISS_GPU_TEMP_MEMORY_BYTES)
        cpu_index = faiss.IndexFlatL2(dimension)
        gpu_index = faiss.index_cpu_to_gpu(resources, device_index, cpu_index)
    except Exception as exc:
        raise _gpu_error(f"GPU index initialization failed with {exc!r}") from exc
    return FlatL2Index(index=gpu_index, gpu_resources=resources)


def create_flat_l2_index(dimension: int, device: str) -> FlatL2Index:
    if dimension < 1:
        raise ValueError("FAISS index dimension must be positive.")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported FAISS device {device!r}; expected 'cpu' or 'cuda'.")

    try:
        faiss = _import_faiss()
    except Exception as exc:
        if device == "cuda":
            raise _gpu_error(f"importing faiss failed with {exc!r}") from exc
        raise RuntimeError(f"FAISS is required for CPU retrieval, but importing it failed with {exc!r}.") from exc

    if device == "cuda":
        return _create_gpu_flat_l2_index(faiss, dimension)

    try:
        return FlatL2Index(index=faiss.IndexFlatL2(dimension))
    except Exception as exc:
        raise RuntimeError(f"FAISS CPU index initialization failed with {exc!r}.") from exc


def validate_faiss_runtime(device: str) -> None:
    probe = create_flat_l2_index(1, device)
    vectors = np.zeros((1, 1), dtype=np.float32)
    try:
        probe.add(vectors)
        distances, indexes = probe.search(vectors, 1)
    except Exception as exc:
        if device == "cuda":
            raise _gpu_error(f"the startup GPU add/search probe failed with {exc!r}") from exc
        raise RuntimeError(f"The startup FAISS CPU add/search probe failed with {exc!r}.") from exc

    if distances.shape != (1, 1) or indexes.shape != (1, 1) or int(indexes[0, 0]) != 0:
        detail = "the startup add/search probe returned an unexpected result"
        if device == "cuda":
            raise _gpu_error(detail)
        raise RuntimeError(f"FAISS CPU is not usable: {detail}.")
