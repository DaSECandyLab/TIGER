"""原 BO tuner_conf 中与算法有关的冻结常量。"""

_REFERENCE_CONFIG = {
    "max_sequence_length": 4096,
    "min_world_size": 1,
}


def get_ref_config(key: str) -> int:
    if key not in _REFERENCE_CONFIG:
        raise ValueError(f"{key} is not present in the internal BO reference config")
    return _REFERENCE_CONFIG[key]
