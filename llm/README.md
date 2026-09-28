
## Initial notes

### First concept

This may, someday, be a mostly?-from-scratch implementation of a gemma 4 runtime.  Or it might just be this .md file I had Gemini deep research make on 2026.09.20.  Time will tell :)

What I'd like to wind up with is a runtime for Gemma 4+ with a good bit (if not all) handwritten code, and then see if I can do something new/interesting.  A Rust implementation could be quite useful to people, but (psst) I've never actually written Rust yet.  This feels like a good time to learn that too.

I'll post what I do have so if anyone stumbles on this and thinks "oh, this is interesting" they can do it, or follow the links to what's already been done...

### Second thoughts

While having something handwritten would be great, I'm not sure I'd actually implement the entire plan.  So how much does it make sense to use LLMs to implement this piece by piece, then spend some time actually understanding what it puts together?

(Of course such code would be recycled from the originals)

In any case, I definitely need the comparison data gemini suggested.

### What I actually started doing

I threw Deepseek 4.1 flash (via DS API, using DS Harness) at this.  $0.32 and a while later, it has a very 
very slow numpy-only version that can output a few tokens.  At this second it's trying to figure out how to 
keep bf16 weights in memory for a numpy-only runtime without tanking performance from 2s/token to 
16-17s/token.  A fully optimized version on the HW I'm using (xeon 61xx with 4-channel DDR4) should be doing 
around 4-5 t/s easily, and when working in 4-bit mode which it will be in the end, 10ish.

But I still want to go with Gemma 4 12B for this... aside from being multimodal which I haven't even asked 
it to dig into yet, it's a usable model that appears to outperform GPT 3.5.  Not that you can do a fair 
comparison since you can't actually *use* old OpenAI/Anthropic models anymore.

## What Happened Next

Anthropic released Opus 5.5, making Claude Code a usable product again (for now?)

I threw it at making Gemma go fast, and then that led to making a simple CPU runtime.

My main contribution was telling it to do something LISPy with the handoff, and then guiding it through bottlenecks.

Then I decided to have it look at Qwen, starting with 3.6 35B/A3B and got that running decently...

Finally, I threw it at Qwen Flash Next, and things started to get really interesting around 2026.09.27.  With the NVFP4 weights repackaged (and the dense layers quantized to 8-bit), I was able to split the prefill load and get 32K tokens prefilled for ~300t/s between my Xeon 8268 (192GB, 4-channel DDR4 2666?) and the 8gb of my 5060ti Chrome wasn't using.

So at this moment, it's code that works on my machine and has a lot of potential, if I can de-slopify it enough!

