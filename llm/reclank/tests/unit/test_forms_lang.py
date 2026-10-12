"""The forms of the compiler alone (ir/program.py): scope, out, def, for,
if, quote, ref, and the comparisons. They make the records of the forms
they stand for."""
from __future__ import annotations

import types

import numpy as np
import pytest

from reclank.ir import program as P


def _model(n=4):
    plan = [types.SimpleNamespace(is_sliding=i % 2 == 0, k_eq_v=i % 2 == 1) for i in range(n)]
    cfg = types.SimpleNamespace(rms_norm_eps=1e-6, plan=plan, num_hidden_layers=n)
    return types.SimpleNamespace(cfg=cfg)


def _records(c):
    return [(op, [v for t, v in args if t != P.T_SLOT]) for op, args in c.p.recs]


X, W = np.zeros(8, np.float32), np.ones(8, np.float32)


def _compile(forms, n=4):
    c = P.Compiler(_model(n))
    c.env["x"] = X
    c.env["w"] = W
    for f in forms:
        c.compile(f)
    return c


def test_for_and_def_unroll_to_the_same_records():
    unrolled = _compile([("layer", i, ("set", "x", ("add", "x", "w"))) for i in range(4)])
    looped = _compile([("def", "f", ("i",), ("set", "x", ("add", "x", "w"))),
                       ("for", "i", 0, 4, ("layer", "i", ("f", "i")))])
    assert _records(unrolled) == _records(looped)
    assert len(looped.p.recs) == 4


def test_layer_scope_and_out():
    c = _compile([("layer", 0, ("let", "a", ("add", "x", "w")), ("out", "carry", "a"))])
    assert "a" not in c.env                 # the let ended with the layer
    assert "carry" in c.env                 # out gave it to the program
    with pytest.raises(NameError):
        c.value("a")


def test_out_skips_function_and_loop_frames():
    c = _compile([("def", "f", ("i",), ("out", "last", ("add", "x", "w"))),
                  ("for", "i", 0, 3, ("layer", "i", ("f", "i")))])
    assert isinstance(c.env["last"], np.ndarray)


def test_if_plan_quote_ref():
    c = _compile([("let", "s", ("if", ("plan", 0, "is_sliding"), ("quote", "yes"), ("quote", "no")))])
    assert c.env["s"] == "yes"
    c = _compile([("let", "b", ("ref", "base.", ("+", 1, 2)))])
    assert isinstance(c.env["b"], P.Slot) and c.env["b"].name == "base.3"
    with pytest.raises(TypeError):
        _compile([("if", "x", 1, 2)])       # an array is not a known condition


def test_comparisons_and_setq():
    c = _compile([("let", "k", ("and", ("<", 1, 2), ("=", 3, 3), ("not", ("!=", 4, 4)))),
                  ("let", "n", 1), ("scope", ("setq", "n", 5))])
    assert c.env["k"] is True and c.env["n"] == 5


def test_lambda_is_a_value_and_keeps_its_lets():
    looped = _compile([("for", "i", 0, 4, ("layer", "i", ("set", "x", ("add", "x", "w"))))])
    lam = _compile([("let", "f", ("lambda", ("a",), ("add", "a", "w"))),
                    ("def", "twice", ("g", "v"), ("g", ("g", "v"))),
                    ("let", "y", ("twice", "f", "x")),
                    ("let", "z", (("lambda", ("a",), ("add", "a", "w")), "y")),
                    ("let", "u", ("f", "z"))])
    assert len(lam.p.recs) == 4 and [op for op, _ in lam.p.recs] == [op for op, _ in looped.p.recs]
    # the lambda keeps the w of where it was made, not that of the call
    c = _compile([("let", "w2", "w"),
                  ("let", "f", ("scope", ("let", "k", 3), ("lambda", ("a",), ("+", "a", "k")))),
                  ("let", "r", ("scope", ("let", "k", 100), ("f", 1)))])
    assert c.env["r"] == 4


def test_rest_parameters():
    c = _compile([("def", "count", ("a", "&rest", "more"), ("+", "a", ("cfg", "num_hidden_layers"))),
                  ("let", "n", ("count", 1, 2, 3)),
                  ("def", "rest", ("a", "&rest", "more"), "more"),
                  ("let", "m", ("rest", 1, 2, 3))])
    assert c.env["n"] == 5 and c.env["m"] == (2, 3)
    with pytest.raises(TypeError):
        _compile([("def", "f", ("a", "b"), "a"), ("f", 1)])


def test_macro_expands_to_forms():
    # (twice-add v): (add (add v w) w); (each-layer i body...): a for of layers
    forms = [("defmacro", "twice-add", ("v",), ("qq", ("add", ("add", ("uq", "v"), "w"), "w"))),
             ("defmacro", "each-layer", ("var", "n", "&rest", "body"),
              ("qq", ("for", ("uq", "var"), 0, ("uq", "n"), ("layer", ("uq", "var"), ("uqs", "body"))))),
             ("each-layer", "i", 2, ("set", "x", ("twice-add", "x")))]
    by_macro = _compile(forms)
    plain = _compile([("layer", i, ("set", "x", ("add", ("add", "x", "w"), "w"))) for i in range(2)])
    # (the inner add makes a temporary of its own in each compile)
    assert [op for op, _ in by_macro.p.recs] == [op for op, _ in plain.p.recs]
    assert len(by_macro.p.recs) == 4 and len(by_macro.p.temps) == len(plain.p.temps) == 2
    c = _compile([("let", "s", ("gensym", "t")), ("let", "s2", ("gensym", "t")),
                  ("let", "l", ("list", 1, ("+", 1, 1), ("quote", "x")))])
    assert c.env["s"] != c.env["s2"] and c.env["l"] == (1, 2, "x")


def test_car_cdr_cons_str():
    c = _compile([("let", "l", ("list", 1, 2, 3)),
                  ("let", "a", ("car", "l")), ("let", "d", ("cdr", "l")),
                  ("let", "e", ("cons", 0, "d")),
                  ("let", "s", ("str", ("quote", "layers."), "a", ("quote", ".w"))),
                  # a macro that takes its forms apart: (first-of f ...) is f
                  ("defmacro", "first-of", ("&rest", "fs"), ("car", "fs")),
                  ("let", "f", ("first-of", ("+", 2, 3), ("quote", "never")))])
    assert (c.env["a"], c.env["d"], c.env["e"], c.env["s"], c.env["f"]) == \
        (1, (2, 3), (0, 2, 3), "layers.1.w", 5)
    with pytest.raises(TypeError):
        _compile([("car", 1)])
