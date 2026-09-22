
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

