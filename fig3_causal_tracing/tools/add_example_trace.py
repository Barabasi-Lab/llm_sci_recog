#!/usr/bin/env python3
"""Optional: add a raw single-scientist causal trace (.pt written by the tracing code) to data/example_traces.npz,
so that make_fig3.py draws panel e from data instead of the static asset.

    python fig3_causal_tracing/tools/add_example_trace.py path/to/residual_w1/csaba_both__prob.pt both

Needs torch (only this helper, not make_fig3.py). The .pt pickles classes from the tracing code's `src` package; they
are stubbed here, so the tracing code does not need to be installed. Only `indirect_effects` (tokens x layers) is kept.
"""
import os
import pickle
import sys
import types

import numpy as np
import torch


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.split(".")[0] in ("src", "dataclasses_json"):
            return type(name, (_Stub,), {})
        return super().find_class(mod, name)


def main(pt, key):
    pm = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="pickle")
    d = torch.load(pt, map_location="cpu", weights_only=False, pickle_module=pm)
    r = d["result"]
    r = r if isinstance(r, dict) else r.__dict__
    assert r.get("kind", d.get("kind")) == "residual" and r.get("window", 1) == 1, "panel d,e use residual, window 1"
    ie = r["indirect_effects"].float().numpy()
    path = os.path.join("fig3_causal_tracing", "data", "example_traces.npz")
    ex = dict(np.load(path))
    ex[f"{key}_ie"] = ie.astype(np.float32)
    np.savez_compressed(path, **ex)
    print(f"added {key}_ie {ie.shape}, max {ie.max():.4f}, p(clean) {r['base_score']:.4f} -> {path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "both")
