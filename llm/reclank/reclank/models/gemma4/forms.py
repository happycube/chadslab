"""The programs of the Gemma 4 models: the forms of a layer and of a step
(the 26B and 12B, and the E4B), their compilation, and the parameters of a
step. The records, the Compiler, and the kernel operations are in
reclank/ir/program.py."""
from __future__ import annotations

import os

import numpy as np

from reclank.core import ops
from reclank.ir.program import (  # noqa: F401
    Compiler, kv_rot_forms)


# ---- the forms of the 26B model -------------------------------------------------

LAYER_FN = "decoder-layer"


def layer_def(model, attn="qc", t=1, fused=False):
    """Return (def decoder-layer (i) ...): one decoder layer as a function of
    its index i. step_form applies it to each layer in a for, so the form of
    a step holds the layer once; the compiler unrolls it (the records are
    those of a form for each layer). What differs between the layers is in
    an if of the plan of layer i: a window or not (the slots cos.s or
    cos.f, the first row the query sees), a key for the value (k_eq_v), the
    first and the last layer.

    The expression follows Model._decoder_layer and Model._attention for t
    tokens. The kernel operations select the kernels of one token or of a
    group from the row count of their input. The attention of a group finds
    the key rows of each query itself. The mode attn is "qc" for the int16
    cache and "f32" for the float cache. x is the hidden state. The layer
    changes it in place. A model with the
    mixture-of-experts block (the 26B) adds the router and the experts; the
    dense model (the 12B) does not.

    fused (a model with experts) gives the forms add_norm2 and ffn_out: the
    norms, the adds, and the scale of the end of the attention and of the
    end of the layer in fewer operations. The layer then takes the input
    norm h from the layer before (the name hn, which ffn_out makes and
    gives out of the scope of the layer: out). A
    compiler that keeps the buffers of a layer for the next one (the step of
    the GPU) uses it; the kernels of the CPU give the same bits in both
    forms.
    """
    cfg = model.cfg
    L = cfg.num_hidden_layers

    def w(name):
        return ("w", "i", name)

    def quote(x):
        return ("quote", x)

    sliding = ("plan", "i", "is_sliding")
    kind = ("if", sliding, quote("s"), quote("f"))
    base = ("ref", "base.", "i")
    qkv = ("if", ("plan", "i", "k_eq_v"),
           ("seq", ("let", ("q", "k"), ("int4_multi4", "h", w("self_attn.q_proj"), w("self_attn.k_proj"))),
            ("let", "v", ("copy", "k"))),
           ("seq", ("let", ("q", "k", "v"), ("int4_multi4", "h", w("self_attn.q_proj"),
                                             w("self_attn.k_proj"), w("self_attn.v_proj")))))
    lo = ("if", sliding, ("max", 0, ("-", "pos", ("-", ("plan", "i", "sliding_window"), 1), base)), 0)
    qc = {"qc": 1, "q8": 2, "qv": 3}.get(attn, 0)
    rot_before, rot_after = kv_rot_forms()
    if t == 1:
        attn_forms = (("let", "lo", lo),
                      ("let", "n", ("-", ("+", "pos", 1), base, "lo")),
                      ("let", "a", ("attn_" + attn, "i", "q", "lo", "n")))
    else:
        attn_forms = (("let", "a", ("attn_rows_" + attn, "i", "q")),)
    first_norm = ("rms_norm", "x", w("input_layernorm"))
    attention = (qkv,
                 ("qkv_norm_rope", "q", "k", "v", w("self_attn.q_norm"), w("self_attn.k_norm"),
                  ("ref", "cos.", kind), ("ref", "sin.", kind), "i"),
                 *rot_before,
                 ("let", "row", ("-", "pos", base)),
                 ("kv_write", "i", "k", "v", "row", qc),
                 *attn_forms,
                 *rot_after,
                 ("let", "o", ("int4", w("self_attn.o_proj"), "a")))
    nxt = ("+", "i", 1)
    last = ("=", "i", L - 1)
    if fused and not cfg.enable_moe_block:
        # the dense model (the 12B) in the fused form of a GPU step: two
        # add_norm2 (as the E4B). After the attention, x += rms_norm(o) and
        # the input norm of the feed-forward part come from one kernel. At
        # the end, x = (x + rms_norm(m)) layer_scalar and the input norm of
        # the next layer (hn), or the final norm (xn) after the last layer,
        # come from one more. The values are those of the separate forms;
        # the norms, the adds, and the quantization of x for the next
        # products (qx_fuse) are fewer kernels.
        wn = ("if", last, ("w", None, "norm"), ("w", nxt, "input_layernorm"))
        body = (("let", "h", ("if", (">", "i", 0), "hn", first_norm)),
                *attention,
                ("let", "hm", ("add_norm2", "x", "o", w("post_attention_layernorm"),
                               w("pre_feedforward_layernorm"))),
                ("let", ("g", "u"), ("int4_multi4", "hm", w("mlp.gate_proj"), w("mlp.up_proj"))),
                ("let", "m", ("gelu_mul_int4", "g", "u", w("mlp.down_proj"))),
                ("let", "hx", ("add_norm2", "x", "m", w("post_feedforward_layernorm"), wn,
                               w("layer_scalar"))),
                ("if", last, ("out", "xn", "hx"), ("out", "hn", "hx")))
        return ("def", LAYER_FN, ("i",), *body)
    fused = fused and cfg.enable_moe_block
    # The router and the experts come before the dense feed-forward part. The
    # two parts read the same x and do not depend on each other, so the order
    # does not change the result. On a GPU (reclank/backends/cuda/runtime.py), the CPU then
    # computes the experts while the GPU computes the dense part.
    if fused:
        experts = (("let", ("val", "idx"), ("router", "x", "i")),
                   ("let", "e", ("moe", "hm", "val", "idx", "i")))
    elif cfg.enable_moe_block:
        experts = (("let", ("val", "idx"), ("router", "x", "i")),
                   ("let", "e", ("moe", ("rms_norm", "x", w("pre_feedforward_layernorm_2")),
                                 "val", "idx", "i")))
    else:
        experts = ()
    if cfg.enable_moe_block:
        ffn = (("let", "f", ("add", ("rms_norm", "m", w("post_feedforward_layernorm_1")),
                             ("rms_norm", "e", w("post_feedforward_layernorm_2")))),)
    else:
        ffn = (("let", "f", "m"),)
    wn = ("if", ("<", nxt, L), ("w", nxt, "input_layernorm"), None)
    body = (("let", "h", ("if", (">", "i", 0), "hn", first_norm) if fused else first_norm),
            *attention,
            *((("let", "hm", ("add_norm2", "x", "o", w("post_attention_layernorm"),
                              w("pre_feedforward_layernorm_2"))),) if fused else
              (("set", "x", ("add", "x", ("rms_norm", "o", w("post_attention_layernorm")))),)),
            *experts,
            ("let", ("g", "u"), ("rms_norm_multi4", "x", w("pre_feedforward_layernorm"),
                                 w("mlp.gate_proj"), w("mlp.up_proj"))),
            ("let", "m", ("gelu_mul_int4", "g", "u", w("mlp.down_proj"))),
            *((("out", "hn", ("ffn_out", "m", w("post_feedforward_layernorm_1"), "e",
                              w("post_feedforward_layernorm_2"),
                              w("post_feedforward_layernorm"), "x", w("layer_scalar"), wn)),)
              if fused else
              (*ffn,
               ("set", "x", ("add", "x", ("rms_norm", "f", w("post_feedforward_layernorm")))),
               ("set", "x", ("mul", "x", w("layer_scalar"))))))
    return ("def", LAYER_FN, ("i",), *body)


def layer_form(model, i, attn="qc", t=1, fused=False):
    """Return decoder layer i as a form that compiles alone: the function
    of layer_def, then a layer that applies it to i."""
    return ("seq", layer_def(model, attn, t, fused), ("layer", i, (LAYER_FN, i)))


def step_form(model, attn="qc", t=1, fused=False):
    """Return a whole decode step of t tokens: every layer, then the final
    norm into xn. fused: see layer_form. A dense model in the fused form
    makes xn in its last layer."""
    layers = (layer_def(model, attn, t, fused),
              ("for", "i", 0, model.cfg.num_hidden_layers, ("layer", "i", (LAYER_FN, "i"))))
    if fused and not model.cfg.enable_moe_block:
        return ("seq", *layers)
    return ("seq", *layers, ("let", "xn", ("rms_norm", "x", ("w", None, "norm"))))


def format_form(form, indent=0):
    """Return a nested expression as Lisp text."""
    pad = " " * indent

    def atom(x):
        if isinstance(x, tuple):
            return None
        return str(x)

    if not isinstance(form, tuple):
        return pad + str(form)
    flat = [atom(x) for x in form]
    if all(a is not None for a in flat) and len(" ".join(flat)) < 70:
        return pad + "(" + " ".join(flat) + ")"
    if form[0] == "w":
        return pad + "(w %s %s)" % form[1:]
    head = str(form[0]) if not isinstance(form[0], tuple) else format_form(form[0])
    parts = [pad + "(" + head]
    for x in form[1:]:
        if isinstance(x, tuple) and x and x[0] == "w":
            parts[-1] += " (w %s)" % x[2]
        elif isinstance(x, tuple) and all(not isinstance(y, tuple) for y in x):
            parts[-1] += " (" + " ".join(str(y) for y in x) + ")"
        elif isinstance(x, tuple):
            parts.append(format_form(x, indent + 2))
        else:
            parts[-1] += " " + str(x)
    return "\n".join(parts) + ")"


def compile_layers(model, layers, attn="qc", t=1):
    """Compile decoder layers for t tokens into one Program. The buffer "x"
    is the input and the output."""
    c = Compiler(model)
    c.env["x"] = np.zeros((t, model.cfg.hidden_size), dtype=np.float32)
    c.p.slot("pos")
    for i in layers:
        c.compile(layer_form(model, i, attn, t))
    c.p.layers = list(layers)
    c.p.attn = attn
    c.p.tokens = t
    return c.p.finish()


def compile_step(model, attn="qc", t=1):
    """Compile a whole step of t tokens. "x" is the input embedding and "xn"
    the hidden state after the final norm, the input of the output head. A
    step of 2 to 16 tokens is the verify step of MTP."""
    c = Compiler(model)
    c.env["x"] = np.zeros((t, model.cfg.hidden_size), dtype=np.float32)
    c.p.slot("pos")
    c.compile(step_form(model, attn, t))
    c.p.layers = list(range(model.cfg.num_hidden_layers))
    c.p.attn = attn
    c.p.tokens = t
    return c.p.finish()


def ready(model, cache):
    """Return the attention mode of a step program for this cache, or None.

    The mode "qc" needs the int16 copy of every layer. It is on after 128
    tokens. Before that, the Python path runs the step, because the step
    that turns the int16 copy on also quantizes the old rows.
    """
    # The cache keeps only quantized rows (KVCache), so a program reads them
    # also with RECLANK_ATTN=0.
    return {"int8": "q8", "k16v8": "qv"}.get(getattr(cache, "kv", "int16"), "qc")


# The largest group that a dense model runs in parts (RECLANK_PARTS: an MTP
# verify group with a PartKVCache; parts.compile_parts t).
PARTS_GROUP_MAX = 16


def parts_group(model, t):
    """True when a step of t tokens runs in parts (RECLANK_PARTS): one token,
    or a group of up to PARTS_GROUP_MAX tokens of a dense model (the experts
    of a part take one token only)."""
    return t == 1 or (t <= PARTS_GROUP_MAX and not model.cfg.enable_moe_block)


def decode_step(model, cache, tokens, pos):
    """Run a step of one or more tokens with a program. Return the hidden
    state after the final norm, shape (tokens, hidden).

    The model keeps one program for each attention mode and token count. It
    makes a program on the first call with that mode and count.
    """
    tokens = [int(x) for x in tokens]
    attn = ready(model, cache)
    n_parts = int(os.environ.get("RECLANK_PARTS", "1"))
    if getattr(cache, "split", False) and (n_parts < 2 or not parts_group(model, len(tokens))):
        raise ValueError("a PartKVCache needs a step of one token, or a group of a dense "
                         "model, in parts")
    if n_parts > 1 and parts_group(model, len(tokens)):
        if attn in ("q8", "qv"):
            raise ValueError("RECLANK_PARTS needs the int16 cache, not RECLANK_KV_INT8")
        from reclank.models.gemma4 import parts
        return parts.decode_step(model, cache, tokens, pos, attn, n_parts)
    progs = model.__dict__.setdefault("_programs", {})
    key = (attn, len(tokens))
    prog = progs.get(key)
    if prog is None:
        prog = progs[key] = compile_step(model, attn, len(tokens))
    bind_step(prog, model, cache, pos)
    prog.names["x"][:] = model.embed(tokens)
    prog.run()
    return prog.names["xn"].copy()


def bind_step(prog, model, cache, pos):
    """Prepare the cache for the tokens of a step and bind the parameters.

    The cache work stays in Python: KVCache.prepare drops old rows and grows
    a buffer. The program then writes the new rows in C. In the mode "qc",
    the int16 copy of the cache must be on for every layer of the program.
    """
    kw = step_params(prog, model, cache, pos)
    prog.bind(**kw)
    return kw


def step_params(prog, model, cache, pos):
    """Prepare the cache for the tokens of a step. Return the parameters of
    the program as a dict. See bind_step."""
    kw = {"pos": pos}
    keep = []
    qc = getattr(prog, "attn", "qc") in ("qc", "q8", "qv")
    t = getattr(prog, "tokens", 1)
    for i in prog.layers:
        assert not qc or cache.qc_ready(i), "the program needs the int16 cache"
        cache.prepare(i, pos, t)
        cache.end[i] = pos + t
        kw["base.%d" % i] = cache.base[i]
        if qc:
            kw.update({"kq.%d" % i: cache.kq[i], "ks.%d" % i: cache.ks[i],
                       "vq.%d" % i: cache.vq[i], "vs.%d" % i: cache.vs[i]})
    positions = np.arange(pos, pos + t)
    for kind, sliding in (("s", True), ("f", False)):
        plan = next((model.cfg.plan[i] for i in prog.layers
                     if model.cfg.plan[i].is_sliding == sliding), None)
        if plan is None:
            continue
        cos, sin, _ca, _sa = model._rope(plan, positions)
        keep += [cos, sin]
        kw["cos." + kind] = cos
        kw["sin." + kind] = sin
    # The scores of the widest attention: all queries of all heads over all
    # the keys.
    need = max(t * model.cfg.plan[i].num_q_heads * (pos + t) for i in prog.layers)
    sc = getattr(prog, "scores", None)
    if sc is None or sc.size < need:
        sc = np.zeros(max(need, 2 * (sc.size if sc is not None else 0)), dtype=np.float32)
        prog.scores = sc
    kw["scores"] = sc
    prog.keep_bound = keep
    return kw


# ---- the E4B model ------------------------------------------------------------

E4B_PREFIX = "model.language_model."


E4B_LAYER_FN = "e4b-layer"


def e4b_layer_def(model, fused=False):
    """Return (def e4b-layer (i) ...): one decoder layer of the E4B model as
    a function of its index i, and the macros its forms use. e4b_step_form
    applies it to each layer in a for (the compiler unrolls it: the records
    of a form for each layer).

    The forms follow E4B.layer, E4B.attention, and E4B.mlp. What differs
    between the layers is in an if of the plan of layer i: a shared layer
    computes only the query and reads the key and the value of its source
    layer; the window (cos.s or cos.f); the first and the last layer.

    The macros: (tw k) and (mw k), the tensor and the matrix k of layer i
    (their names: (str "<prefix>layers." i "." k)); (add-norm o w [scale]),
    x += rms_norm(o) w, then x *= scale.

    fused uses add_norm and gelu_mul, which do two or three operations in
    one pass and give the same values. The GPU group uses it. fused 2 also
    computes each norm of x in the kernel that changes x (add_norm2): the
    norm before the feed-forward part, and the input norm of the next layer
    (or the final norm into xn). Layer 0 then starts with its own norm, and
    the other layers take h from the layer before (out). The GPU decode step
    uses it; the buffer h goes from one layer to the next, so it needs a
    compiler that does not reuse the buffers of a layer.
    """
    L = model.cfg.num_hidden_layers
    name = lambda i, k: ("str", ("quote", E4B_PREFIX + "layers."), i, ("quote", "." + k))  # noqa: E731
    # (tw k), (mw k): (t (str "<prefix>layers." i ".k")), the same with m
    tensor = lambda mname, h: ("defmacro", mname, ("k",), ("qq", (  # noqa: E731
        h, ("str", ("quote", E4B_PREFIX + "layers."), "i",
            ("quote", ("uq", ("str", ("quote", "."), "k")))))))
    # (add-norm o w [scale]): the forms of the fused kernel, or of the add,
    # the norm, and the scale
    if fused:
        add_norm = ("defmacro", "add-norm", ("o", "w", "&rest", "scale"),
                    ("qq", ("set", "x", ("add_norm", "x", ("uq", "o"), ("tw", ("uq", "w")),
                                         ("uqs", ("if", ("len", "scale"),
                                                  ("list", ("list", ("quote", "tw"), ("car", "scale"))),
                                                  ("list",)))))))
    else:
        add_norm = ("defmacro", "add-norm", ("o", "w", "&rest", "scale"),
                    ("qq", ("seq", ("set", "x", ("add", "x", ("rms_norm", ("uq", "o"),
                                                             ("tw", ("uq", "w"))))),
                            ("uqs", ("if", ("len", "scale"),
                                     ("list", ("qq", ("set", "x", ("mul", "x",
                                                                   ("tw", ("uq", ("car", "scale"))))))),
                                     ("list",))))))
    macros = (tensor("tw", "t"), tensor("mw", "m"), add_norm)
    kind = ("if", ("plan", "i", "is_sliding"), ("quote", "s"), ("quote", "f"))
    attn = ("if", ("plan", "i", "shared"),
            ("seq", ("let", "q", ("linear", ("mw", "self_attn.q_proj"), "h")),
             ("qkv_norm", "q", None, None, ("tw", "self_attn.q_norm.weight"), None, "i"),
             ("rope", "q", None, ("ref", "cos.", kind), ("ref", "sin.", kind), "i")),
            ("seq", ("let", ("q", "k", "v"), ("int4_multi4", "h", ("mw", "self_attn.q_proj"),
                                             ("mw", "self_attn.k_proj"), ("mw", "self_attn.v_proj"))),
             ("qkv_norm", "q", "k", "v", ("tw", "self_attn.q_norm.weight"),
              ("tw", "self_attn.k_norm.weight"), "i"),
             ("rope", "q", "k", ("ref", "cos.", kind), ("ref", "sin.", kind), "i"),
             ("kv_write_heads", "i", "k", "v")))
    if fused == 2:
        last = ("=", "i", L - 1)
        nxt = ("if", last, ("t", E4B_PREFIX + "norm.weight"),
               ("t", name(("+", "i", 1), "input_layernorm.weight")))
        head = (("if", ("=", "i", 0), ("let", "h", ("rms_norm", "x", ("tw", "input_layernorm.weight")))),)
        mid = (("let", "hf", ("add_norm2", "x", "o", ("tw", "post_attention_layernorm.weight"),
                              ("tw", "pre_feedforward_layernorm.weight"))),)
        ffn_in = "hf"
        tail = (("let", "hx", ("add_norm2", "x", "pp", ("tw", "post_per_layer_input_norm.weight"), nxt,
                               ("tw", "layer_scalar"))),
                ("if", last, ("out", "xn", "hx"), ("out", "h", "hx")))
    else:
        head = (("let", "h", ("rms_norm", "x", ("tw", "input_layernorm.weight"))),)
        mid = (("add-norm", "o", "post_attention_layernorm.weight"),)
        ffn_in = ("rms_norm", "x", ("tw", "pre_feedforward_layernorm.weight"))
        tail = (("add-norm", "pp", "post_per_layer_input_norm.weight", "layer_scalar"),)
    return ("seq", *macros,
            ("def", E4B_LAYER_FN, ("i",),
             *head,
             attn,
             ("let", "a", ("attn_e4b", "i", "q")),
             ("let", "o", ("linear", ("mw", "self_attn.o_proj"), "a")),
             *mid,
             ("let", ("g", "u"), ("int4_multi4", ffn_in, ("mw", "mlp.gate_proj"), ("mw", "mlp.up_proj"))),
             ("let", "d", ("linear", ("mw", "mlp.down_proj"),
                           ("gelu_mul", "g", "u") if fused else ("mul_v", ("gelu", "g"), "u"))),
             ("add-norm", "d", "post_feedforward_layernorm.weight"),
             ("let", "pg", ("gelu", ("linear", ("mw", "per_layer_input_gate"), "x"))),
             ("let", "pp", ("linear", ("mw", "per_layer_projection"), ("mul_pli", "pg", "pl", "i"))),
             *tail))


def e4b_step_form(model, fused=False):
    """Return a whole step of the E4B model.

    The step starts with the per-layer inputs, as E4B.per_layer_inputs does
    it. That is the projection of x, its scale and norm, and the sum with the
    token part "tok". Then come the layers (e4b_layer_def, in a for) and the
    final norm into xn.
    """
    cfg = model.cfg
    n = cfg.hidden_size_per_layer_input
    return ("seq",
            ("let", "proj", ("linear", ("m", E4B_PREFIX + "per_layer_model_projection"), "x")),
            ("set", "proj", ("mul", "proj", cfg.per_layer_model_projection_scale)),
            ("let", "pn", ("rms_norm_rows", "proj", n,
                           ("t", E4B_PREFIX + "per_layer_projection_norm.weight"))),
            ("let", "pl", ("mul", ("add", "pn", "tok"), cfg.per_layer_input_scale)),
            e4b_layer_def(model, fused),
            ("for", "i", 0, cfg.num_hidden_layers, ("layer", "i", (E4B_LAYER_FN, "i"))),
            *(() if fused == 2 else
              (("let", "xn", ("rms_norm", "x", ("t", E4B_PREFIX + "norm.weight"))),)))


def compile_e4b_step(model, t=1, fused=False, q4_kq=False):
    """Compile a whole E4B step of t tokens. "x" and "tok" are the inputs;
    "xn" is the hidden state after the final norm. fused selects the fused
    operations of e4b_layer_form (the GPU). q4_kq takes the Q4_0 matrices as
    GGUF products (the GPU, Compiler "m")."""
    cfg = model.cfg
    c = Compiler(model)
    c.q4_kq = q4_kq
    c.env["x"] = np.zeros((t, cfg.hidden_size), dtype=np.float32)
    c.env["tok"] = np.zeros((t, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input),
                            dtype=np.float32)
    c.p.slot("pos")
    c.compile(e4b_step_form(model, fused))
    c.p.tokens = t
    return c.p.finish()


def e4b_ready(model, cache):
    """Return True when an E4B step can run as a program.

    The program uses the fused float attention and the int4 blocks of a GGUF
    file. The cache must hold the buffers of every layer that stores a key.
    """
    if not (ops.attn_ready() and model.mode == "int4" and model._q4 and cache.n > 0):
        return False
    return all(i in cache.kv for i, p in enumerate(model.cfg.plan) if not p.shared)


def e4b_bind_step(prog, model, cache, pos):
    """Prepare the E4B cache for the tokens of a step and bind the parameters."""
    prog.bind(**e4b_step_params(prog, model, cache, pos))


def e4b_step_params(prog, model, cache, pos, rope=None):
    """Prepare the E4B cache for the tokens of a step. Return the parameters
    of the program as a dict. rope(kind, pos, t), if given, returns the
    addresses of the cosine and the sine rows of the positions (tables on
    the GPU, E4BGPU); else the dict holds the tables."""
    t = prog.tokens
    cache._reserve(pos + t)
    kw = {"pos": pos}
    for i, plan in enumerate(model.cfg.plan):
        if plan.shared:
            continue
        k, v = cache.kv[i]
        # position-major (positions, kv heads, head_dim): a head stride of head_dim
        kw.update({"k.%d" % i: k, "v.%d" % i: v, "hs.%d" % i: k.shape[2]})
    cache.n = max(cache.n, pos + t)
    keep = []
    for kind, sliding in (("s", True), ("f", False)):
        if rope is not None:
            kw["cos." + kind], kw["sin." + kind] = rope(kind, pos, t)
            continue
        plan = next(p for p in model.cfg.plan if p.is_sliding == sliding)
        cos, sin = model.rope_tables(plan, pos, t)
        cos = np.ascontiguousarray(cos, dtype=np.float32)
        sin = np.ascontiguousarray(sin, dtype=np.float32)
        keep += [cos, sin]
        kw["cos." + kind] = cos
        kw["sin." + kind] = sin
    need = max(p.num_q_heads * (pos + t) for p in model.cfg.plan)
    sc = getattr(prog, "scores", None)
    if sc is None or sc.size < need:
        sc = np.zeros(max(need, 2 * (sc.size if sc is not None else 0)), dtype=np.float32)
        prog.scores = sc
    kw["scores"] = sc
    prog.keep_bound = keep
    return kw


def decode_step_e4b(model, cache, tokens, pos):
    """Run an E4B step of one or more tokens with a program. Return the hidden
    state after the final norm, shape (tokens, hidden)."""
    cfg = model.cfg
    ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
    progs = model.__dict__.setdefault("_programs", {})
    prog = progs.get(ids.size)
    if prog is None:
        prog = progs[ids.size] = compile_e4b_step(model, ids.size)
    e4b_bind_step(prog, model, cache, pos)
    # The inputs, as E4B.forward and E4B.per_layer_inputs make them.
    prog.names["x"][:] = model.embed_rows(E4B_PREFIX + "embed_tokens", ids) * cfg.embed_scale
    tok = model.embed_rows(E4B_PREFIX + "embed_tokens_per_layer", ids)
    prog.names["tok"][:] = (tok * cfg.per_layer_embed_scale).reshape(ids.size, -1)
    prog.run()
    return prog.names["xn"].copy()
