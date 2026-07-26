import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType


VPR_EVALUATION_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "VPR-methods-evaluation"
TEST_DATASET_PATH = VPR_EVALUATION_ROOT / "test_dataset.py"


@lru_cache(maxsize=1)
def load_vpr_test_dataset_module() -> ModuleType:
    if not TEST_DATASET_PATH.is_file():
        raise FileNotFoundError(f"VPR evaluation dataset implementation was not found at {TEST_DATASET_PATH}")

    spec = importlib.util.spec_from_file_location("vpr_methods_evaluation_test_dataset", TEST_DATASET_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {TEST_DATASET_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_vpr_evaluation_dataset(args, dataset_name: str):
    upstream = load_vpr_test_dataset_module()
    test_root = Path(args.eval_datasets_folder).expanduser() / dataset_name / "images" / "test"
    dataset = upstream.TestDataset(
        str(test_root / "database"),
        str(test_root / "queries"),
        positive_dist_threshold=args.val_positive_dist_threshold,
        image_size=list(args.resize),
        use_labels=True,
    )
    dataset.database_num = dataset.num_database
    dataset.queries_num = dataset.num_queries
    dataset.dataset_name = dataset_name
    dataset.resize = list(args.resize)
    dataset.test_method = args.test_method
    return dataset
