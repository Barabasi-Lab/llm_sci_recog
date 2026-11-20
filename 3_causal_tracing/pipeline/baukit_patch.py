"""nnsight-0.5 compatibility patch for the causal-tracing library (imported by 01_trace_batch.py).

The tracing code is Arnab Sen Sharma's implementation (David Bau lab) of ROME causal tracing
(Meng et al., "Locating and Editing Factual Associations in GPT", NeurIPS 2022): package `src`, entry point
`src.trace.trace_important_states`. It is not included in this repository. Its two nnsight primitives use the
nnsight ~0.2/0.3 API (`module.output.save()` / `.value`), which raises `OutOfOrderError` under nnsight 0.5.
`install_baukit_patches()` replaces exactly those two with baukit `TraceDict` equivalents:

    src.functional.get_hs  -> get_hs_baukit       (capture hidden states of the clean, i.e. scientist, run)
    src.trace.patched_run  -> patched_run_baukit  (forward pass with states overwritten at (layer, token))

Everything else (subject alignment and padding, window, scoring, IE matrix) is the library's own code.
Copied verbatim from the project's run_two_examples.py, the version used for all Fig. 3 traces.
"""
import logging

import torch

log = logging.getLogger(__name__)


# --------------------------------------------------------- baukit compatibility patch
# The repo's tracing primitives use an old nnsight API (module.output.save()/.value),
# incompatible with the installed nnsight 0.5.x (OutOfOrderError). We replace only the
# two primitives -- state capture (get_hs) and patched forward (patched_run) -- with
# equivalent baukit implementations. All scoring/aggregation/plotting logic is untouched.
def install_baukit_patches():
    import baukit
    import src.functional as F
    import src.trace as T
    from src.utils.typing import TokenizerOutput

    def _untuple(x):
        return x[0] if isinstance(x, tuple) else x

    def _model_inputs(mt, inp):
        if not isinstance(inp, TokenizerOutput):
            inp = F.prepare_input(prompts=inp, tokenizer=mt.tokenizer)
        if "offset_mapping" in inp:
            inp.pop("offset_mapping")
        dev = mt._model.device
        d = {"input_ids": inp["input_ids"].to(dev)}
        if "attention_mask" in inp:
            d["attention_mask"] = inp["attention_mask"].to(dev)
        return d

    @torch.inference_mode()
    def get_hs_baukit(mt, input, locations, patches=None, return_dict=False):
        assert patches is None, "patch-injection path not used in this script"
        if isinstance(locations, tuple):
            locations = [locations]
        model_inp = _model_inputs(mt, input)
        layer_names = list({ln for ln, _ in locations})
        with baukit.TraceDict(mt._model, layer_names) as ret:
            mt._model(**model_inp)
        hs = {}
        for layer_name, index in locations:
            out = _untuple(ret[layer_name].output)
            hs[(layer_name, index)] = out[:, index, :].squeeze().detach().clone()
        if len(hs) == 1 and not return_dict:
            return list(hs.values())[0]
        return hs

    @torch.inference_mode()
    def patched_run_baukit(mt, inputs, states):
        model_inp = _model_inputs(mt, inputs)
        by_layer = {}
        for (layer_name, token_idx), vec in states.items():
            by_layer.setdefault(layer_name, []).append((token_idx, vec))
        layer_names = list(by_layer.keys())

        def edit_output(output, layer_name):
            if layer_name not in by_layer:
                return output
            h = _untuple(output)
            for token_idx, vec in by_layer[layer_name]:
                h[0, token_idx, :] = vec.to(h.dtype).to(h.device)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        if layer_names:
            with baukit.TraceDict(mt._model, layer_names, edit_output=edit_output):
                out = mt._model(**model_inp)
        else:
            out = mt._model(**model_inp)
        return out.logits[0][-1]

    F.get_hs = get_hs_baukit          # get_all_module_states() calls get_hs by name
    T.patched_run = patched_run_baukit
    log.info("baukit compatibility patches installed")
