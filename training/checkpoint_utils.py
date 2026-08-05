"""Shared, agent-class-agnostic checkpoint save/load.

Each trainable agent class (BranchingMPDQN, DDQNAgent, HybridSACDDQN, PDQNAgent,
MPDQNAgent) owns a `state_dict()`/`load_state_dict()` pair -- only the class
itself knows which submodules/optimizers/counters it has. Everything else
(file format, metadata packaging, and reconstructing an agent instance without
the caller having to remember its original hyperparameters) is common across
every agent class and lives here once, satisfying docs/rules.md's
Reproducibility Rule ("any experiment must be reproducible") for saved models.
"""

from pathlib import Path
from typing import Any, Dict, Type, Union

import torch

FORMAT_VERSION = 1


def save_checkpoint(agent: Any, path: Union[str, Path], meta: Dict[str, Any]) -> None:
    """Serialize `agent` plus reconstruction metadata to a single ``.pt`` file.

    Args:
        agent: Any object exposing `.state_dict()`, plus `.state_dim`, `.n_rrh`,
            `.p_max_w` attributes (every agent class in `agents/` has these).
        path: Destination file path; parent directories are created if needed.
        meta: Must contain "config" (the full resolved training config used to
            build `agent`) and "ctor_kwargs" (the *extra* constructor keyword
            arguments -- beyond state_dim/n_rrh/p_max_w/device, which are
            captured automatically -- needed to reconstruct this agent class;
            e.g. `{"config": cfg}` for config-driven agents, or explicit scalar
            hyperparameters for agents like DDQNAgent that take flat kwargs).
            May also carry provenance fields (seed, episode, etc.) verbatim.
    """
    payload = {
        "format_version": FORMAT_VERSION,
        "agent_class": type(agent).__name__,
        "state_dim": agent.state_dim,
        "n_rrh": agent.n_rrh,
        "p_max_w": agent.p_max_w,
        "agent_state": agent.state_dict(),
        "meta": meta,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    agent_cls: Type,
    path: Union[str, Path],
    device: str = "cpu",
    **ctor_overrides: Any,
) -> Any:
    """Reconstruct and return a frozen `agent_cls` instance from a checkpoint.

    Reads `state_dim`/`n_rrh`/`p_max_w` and `meta["ctor_kwargs"]` out of the
    checkpoint itself -- the caller does not need to remember or guess the
    original hyperparameters. `ctor_overrides` lets the caller patch/override
    any constructor kwarg (most commonly `device`) without touching the stored
    metadata.

    Every `torch.nn.Module` attribute on the returned agent is set to `.eval()`
    mode before it is returned.
    """
    payload = torch.load(Path(path), map_location=device, weights_only=False)

    ctor_kwargs: Dict[str, Any] = {
        "state_dim": payload["state_dim"],
        "n_rrh": payload["n_rrh"],
        "p_max_w": payload["p_max_w"],
        "device": device,
    }
    ctor_kwargs.update(payload["meta"].get("ctor_kwargs", {}))
    ctor_kwargs.update(ctor_overrides)

    agent = agent_cls(**ctor_kwargs)
    agent.load_state_dict(payload["agent_state"])

    for value in vars(agent).values():
        if isinstance(value, torch.nn.Module):
            value.eval()

    return agent
