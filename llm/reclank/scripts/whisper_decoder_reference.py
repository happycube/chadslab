"""Whisper decoder references for scripts/check_whisper_decoder.py: transformers'
greedy transcription of real clips and the decoder's logits over it, in float32 on
the CPU (torch).

    whisper_decoder_reference.py MODEL_DIR refs.npz out.npz [--lang sk] [--task transcribe]

refs.npz is whisper_reference.py's (the input features "f<i>"). out.npz: "names",
"t<i>" the tokens of clip i (the prompt <|startoftranscript|> <|lang|> <|task|>
<|notimestamps|>, then generate's greedy tokens to <|endoftext|>), "l<i>" the
logits (len(t<i>) - 1, vocab) of the decoder over t<i> (row j: the next token
after t<i>[:j + 1]), "e<i>" the encoder's last hidden state it read, and "text<i>"."""
import argparse

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("refs")
ap.add_argument("out")
ap.add_argument("--lang", default="sk")
ap.add_argument("--task", default="transcribe")
a = ap.parse_args()
torch.set_num_threads(16)
proc = WhisperProcessor.from_pretrained(a.model)
model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float32).eval()
tok = proc.tokenizer
eot = tok.convert_tokens_to_ids("<|endoftext|>")
prompt = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|%s|>" % a.lang, "<|%s|>" % a.task, "<|notimestamps|>"])
d = np.load(a.refs)
res = {"names": d["names"]}
for i in range(len(d["names"])):
    f = torch.from_numpy(d["f%d" % i])[None]
    with torch.no_grad():
        gen = model.generate(f, language=a.lang, task=a.task, return_timestamps=False,
                             do_sample=False, num_beams=1)[0]
        # generate gives the new tokens (no prompt, no <|endoftext|>) in transformers 4.48
        seq = torch.tensor(prompt + [t for t in gen.tolist() if t not in prompt and t != eot] + [eot])
        enc = model.model.encoder(f).last_hidden_state
        logits = model(encoder_outputs=(enc,), decoder_input_ids=seq[None, :-1]).logits[0]
    t = seq.numpy().astype(np.int32)
    text = proc.tokenizer.decode(t, skip_special_tokens=True)
    res["t%d" % i], res["l%d" % i] = t, logits.numpy().astype(np.float32)
    res["e%d" % i], res["text%d" % i] = enc[0].numpy().astype(np.float32), np.array(text)
    print(i, d["names"][i], len(t), repr(text[:120]), flush=True)
np.savez(a.out, **res)
