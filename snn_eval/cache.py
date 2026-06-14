"""Persistent storage for trained models and experiment results.

  results/models/<exp>_<hash>/            ← PyTorch state_dicts (.pt files)
  results/cache/<exp>_<hash>_results.json ← pre-computed metrics (JSON)

Model cache is keyed by training-only params (arch, epochs, seed, …) so you
can reload saved models even if you change inference params (Np, Nm, T, …).

Results cache is keyed by ALL params so a different Np forces re-inference.
"""
import hashlib, json, os
import torch

MODELS_ROOT  = os.path.join("results", "models")
RESULTS_ROOT = os.path.join("results", "cache")


def _key(tag: str, params: dict) -> str:
    blob = json.dumps({"_tag": tag, **params}, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


# ---------- model state_dicts ----------

def _models_dir(experiment: str, train_params: dict) -> str:
    return os.path.join(MODELS_ROOT, f"{experiment}_{_key(experiment, train_params)}")


def save_models(experiment: str, train_params: dict, **named_models) -> None:
    """Save nn.Module state_dicts (or raw tensors) to results/models/."""
    d = _models_dir(experiment, train_params)
    os.makedirs(d, exist_ok=True)
    for name, model in named_models.items():
        state = model.state_dict() if isinstance(model, torch.nn.Module) else model
        torch.save(state, os.path.join(d, f"{name}.pt"))
    print(f"[models] saved → {d}/")


def load_models(experiment: str, train_params: dict, **named_model_instances) -> set:
    """Load state_dicts into existing instances in-place, per model.

    Loads every model whose .pt file exists and returns the set of loaded
    names (empty set on total miss), so callers can train only the missing
    ones. Delete a single .pt file to force retraining of just that model.
    """
    d = _models_dir(experiment, train_params)
    loaded = set()
    if not os.path.isdir(d):
        return loaded
    for name, model in named_model_instances.items():
        path = os.path.join(d, f"{name}.pt")
        if not os.path.exists(path):
            continue
        state = torch.load(path, map_location="cpu")
        if isinstance(model, torch.nn.Module):
            model.load_state_dict(state)
            model.eval()
        loaded.add(name)
    if loaded:
        print(f"[models] loaded {sorted(loaded)} ← {d}/")
    return loaded


# ---------- results JSON ----------

def _results_path(experiment: str, all_params: dict) -> str:
    return os.path.join(
        RESULTS_ROOT, f"{experiment}_{_key(experiment + '_r', all_params)}_results.json"
    )


def save_results(experiment: str, all_params: dict, data: dict) -> None:
    """Persist a JSON-serialisable results dict to results/cache/."""
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    path = _results_path(experiment, all_params)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[results] saved → {path}")


def load_results(experiment: str, all_params: dict):
    """Return previously saved results dict, or None on miss."""
    path = _results_path(experiment, all_params)
    if os.path.exists(path):
        print(f"[results] hit  ← {path}")
        with open(path) as f:
            return json.load(f)
    return None
