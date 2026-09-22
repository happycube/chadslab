#!/usr/bin/env python3
"""Find the nibble layout of the w4a16 packed weights."""
import numpy as np
from np_gemma import SafeTensors, ops

W4 = "../gemma4-12b-qat-pytorch/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-w4a16-ct/snapshots/1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee/model.safetensors"
UQ = "../gemma4-12b-qat-pytorch/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-q4_0-unquantized/snapshots/b6ed86275a6a5735884e208bfed95b445a684ca2/model.safetensors"
name = "model.language_model.layers.0.mlp.gate_proj"
with SafeTensors(W4) as w4, SafeTensors(UQ) as uq:
    pk = w4.get(name + ".weight_packed", dtype=None)
    sc = ops.bf16_to_f32(w4.get_bf16(name + ".weight_scale"))
    ref = uq.get(name + ".weight")
    print("packed", pk.shape, pk.dtype, "scale", sc.shape, sc.dtype, "ref", ref.shape)
    row = 5
    p = pk[row].view(np.uint8)
    low = (p & 0x0F).astype(np.int8)
    high = ((p >> 4) & 0x0F).astype(np.int8)

    def signed(v):
        return np.where(v >= 8, v - 16, v).astype(np.int8)

    def offset(v):
        return (v - 8).astype(np.int8)

    cols = 3840
    cands = {}
    for nm, fn in (("signed", signed), ("offset", offset)):
        l = fn(low)
        h = fn(high)
        a = np.empty(cols, np.int8)
        a[0::2] = l
        a[1::2] = h
        cands["even " + nm] = a
        b = np.empty(cols, np.int8)
        b[:1920] = l
        b[1920:] = h
        cands["half " + nm] = b
        c = np.empty(cols, np.int8)
        for bo in range(120):
            c[bo*32:bo*32+16] = l[bo*16:bo*16+16]
            c[bo*32+16:bo*32+32] = h[bo*16:bo*16+16]
        cands["block32 " + nm] = c
    target = ref[row]
    s = np.repeat(sc[row], 32)
    for k, v in cands.items():
        deq = v.astype(np.float32) * s
        err = float(np.abs(deq - target).mean())
        print("%-14s mean abs %.3e  corr %.4f" % (k, err, float(np.corrcoef(deq, target)[0, 1])))
