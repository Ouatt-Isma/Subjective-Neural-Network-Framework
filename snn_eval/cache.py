"""File-based result cache keyed by experiment name + parameters.

Cache key: first 12 hex chars of MD5(experiment + sorted JSON of params).
Files stored in results/cache/<name>_<hash>.pkl.
"""
import hashlib, json, os, pickle

CACHE_DIR = os.path.join("results", "cache")


def _key(experiment: str, params: dict) -> str:
    blob = json.dumps({"_exp": experiment, **params}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def _path(experiment: str, params: dict) -> str:
    return os.path.join(CACHE_DIR, f"{experiment}_{_key(experiment, params)}.pkl")


def load(experiment: str, params: dict):
    """Return cached result dict, or None on miss."""
    p = _path(experiment, params)
    if os.path.exists(p):
        print(f"[cache] hit  {p}")
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def save(experiment: str, params: dict, data) -> None:
    """Persist data to cache under results/cache/."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = _path(experiment, params)
    with open(p, "wb") as f:
        pickle.dump(data, f)
    print(f"[cache] saved {p}")
