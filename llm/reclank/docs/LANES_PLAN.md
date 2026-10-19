# Two lanes: two decodes at once, the CPU and the GPU in turn

## 1. The idea

A decode step of Qwen3.8-Flash-Next with the cold experts on the CPU leaves
both sides idle half the time. In each layer the GPU runs the attention (or
DeltaNet) and the router, signals the CPU, runs the hot experts and the
shared expert, and waits for the CPU's cold experts; the CPU waits for the
signal, computes, and waits for the next layer. EXL3 5.05, a plain step
(Xeon 8268, 3090):

| per step | time |
|---|---|
| the step and its argmax | 21.4 ms |
| the CPU programs of the cold experts (48 tasks) | 5.5 ms (114 us a task) |
| the runner waiting for the GPU's signals | 13.9 ms |

and the GPU, in an nsys trace of the MTP decode, waited for the CPU 24% of
its time and was idle between kernels 14%. Two decodes (two requests, two
KV caches) whose steps run at the same time fill those gaps: while lane A's
cold experts run on the CPU the GPU runs lane B's attention, and the other
way round. The products of a step are bandwidth-bound GEMVs and each lane
reads the weights itself, so a lane is not faster; the pair is. The ceiling
is roughly the time the GPU waits and idles: 1.2-1.5x the tokens a second of
one lane, with each lane somewhat slower than alone.

## 2. The design

- **Lane A** is the model as it is: its step program, its cache, the model
  thread. **Lane B** (reclank/models/qwen/lanes.py) is a second step program
  of the same model: its own buffers (a compile of its own: `_pool`, its own
  `Arenas` for the packed temporaries, which the programs of a model share
  only because they run one at a time), its own pinned cold lists (`ip`),
  its own KV cache and `DeviceCache` (owner `cache1`), its own QSA scratch;
  the weights, the hot experts, the pool, and the head program are shared.
- **Two CUDA streams.** `gg_stream` becomes the stream of the calling thread
  (thread-local, the model thread's by default); `rc_gpu_lane(1)` gives a
  thread lane B's stream. Lane B's step runs from a helper thread
  (`rc_gpu_run` releases the GIL), lane A's from the model thread.
- **One owner of the CPU teams.** The cold experts run in the parts
  (`rc_run_parts`: OpenMP teams of the calling thread), and a second thread
  with teams of its own puts the threads over the CPUs (server/worker.py:
  58 ms steps in place of 46). So lane B's thread never runs a CPU program:
  at each of its tasks it waits for its GPU signal, then puts the program in
  a mailbox and waits; the model thread runs it while it waits for lane A's
  signals (gg_cpu_task), and after lane A's run until lane B is done
  (`rc_gpu_lane_serve`). The tasks of the two lanes alternate on the one team.
- **Lockstep pairs.** A pair: the host work of both lanes on the model
  thread (inputs, parameters, `bind`, the locks of GpuMem), the two runs at
  once, lane B's stream synchronized at its end, then the heads (A, then B)
  and HotCache (A's selection, then B's) on the model thread. The slots of
  the hot experts change only between pairs, when neither lane runs, as they
  change between steps now.
- **Graphs.** Lane B's graphs are recorded in a run of its own (no capture
  while another stream launches).

## 3. Invariants and checks

- Each lane's tokens are those of the same decode alone, given the same hot
  slots; with the rows form of the combine (the bits do not depend on which
  experts the GPU holds) they are the same tokens as a lane alone.
- The parts run one program at a time (the mailbox): no team of a second
  thread exists (the threads of the process stay at the parts' count).
- A step of one lane alone (no pair) runs as before: no mailbox, no second
  stream.

## 4. The steps

1. The runtime: the thread-local stream, `rc_gpu_lane`, the mailbox in
   gg_cpu_task, `rc_gpu_lane_serve`, `rc_gpu_lane_sync`.
2. Lane B (lanes.py): its program, cache, prefill (the model's prompt
   programs on lane B's cache, one at a time), step inputs, head; a pair
   (`Lanes.step_pair`) and a greedy decode of two prompts.
3. The check and the measure: the tokens of each lane against the lane
   alone; the tokens a second of the pair against one lane (work/lanes).
4. The server: two requests at once (the model thread runs pairs while two
   decodes are open; a prompt of a new request runs alone between pairs;
   one lane when only one request is open). MTP lanes (a verify and drafts
   for each) after that.

## 5. Status

Steps 1-3 are done (runtime: the thread-local stream, rc_gpu_lane, the
mailbox, rc_gpu_lane_serve; lanes.py; scripts/check_lanes.py). EXL3 5.05, two
chat prompts, greedy, 128 tokens each:

| | one lane alone | the pair (both) |
|---|---|---|
| fixed hot set (RECLANK_GPU_HOT_DYN=0) | 38.4 tok/s | 63.3 tok/s (1.65x), each lane the tokens of its run alone |
| HotCache | 45.8 tok/s | 62.6 tok/s (1.37x) |

One lane alone runs as before (plain 45.5, MTP 64.7 tok/s, the same KL).
What the pair needed besides the plan: lane B's C scratches of its own
(rc_gpu_scratch_own: the programs of a model share gg_scratch_get's, and the
hot MoE of the two steps read each other's tables, an out-of-range
address); its thread on a CPU of its own (it inherited the model thread's
one CPU, and each hand-off took a time slice: 290 ms steps); each program's
blocks locked right after its bind (the other's bind can move them).

Plain pairs give about what MTP gives one lane (65 tok/s), so with MTP on,
a pair of plain lanes is a way to serve two requests at once, not more
tokens a second; MTP lanes (a verify group and drafts for each) would be.

MTP lanes (lanes.verify_pair, greedy_mtp_pair; scripts/check_lanes.py with
MTP=1): lane B has verify groups of its own (compiled as its step: its
arenas, its sel buffers for HotCache, its C scratches), its own MTP cache,
and commits through Lane.view; the MTP programs, the drafter's head, and
HotCache are the model's, run by each lane in turn. A round: the drafts of A
then of B (3 each, on the model thread, nothing beside them), the two verify
groups as a pair, the heads, the commits. EXL3 5.05 (RECLANK_GPU_PARTS=1,
as M=qfe), the two chat prompts, greedy, 256 tokens each:

| | lane A alone | lane B alone | the pair (both) |
|---|---|---|---|
| HotCache | 61.0 tok/s (2.98 a round, 49 ms) | 78.4 tok/s (3.66, 47 ms) | 94-97 tok/s (6.7 a round, 70 ms: drafts 14, verify pair + heads + commits 56) |

1.37x the mean of one lane, 1.2x the faster prompt alone. With a fixed hot
set each lane gives the tokens of its run alone (128 of 128); with HotCache
lane A parts at a near tie (an expert on the GPU or the CPU).

Where a round goes (69.9 ms with lane B's drafts before the pair): the
pair 49.5, lane A's drafts 6.9, lane B's 5.9 (3 MTP runs and heads each,
2.3 ms a draft), the commits 4.2 (A 2.2, B 2.0), the heads of the verify
groups 2.1, HotCache and the rest 1.2. In the pair the CPU team is busy 67%
(lane A's tasks 17 ms, lane B's 16 ms; rc_gpu_lane_stats): each lane's
verify group takes about 49 ms beside the other, and that sets the pair.

Lane B's drafts in the pair (RECLANK_LANES_OVERLAP, on: verify_pair_drafting,
its own MTP and head programs, HotCache's scores after the pair): the pair
becomes lane B's drafts + its verify group, 54.4 ms (lane A done 5 ms before
it and idle), so the round takes 1-3% less (4.69-4.77 s in place of
4.82-4.89 s for 69 rounds). The rest of that time is in lane A's work after
its verify group (head, commit, the next drafts: about 10 ms) at the end of
the pair, on the model thread beside lane B's verify group: about 61-63 ms a
round, with HotCache's work after the pair and no allocation in it.

## 6. The server (--lanes 2)

`--lanes 2` (Qwen3.8-Flash-Next on the GPU) serves two requests at once,
each lane a cache of `--ctx` positions (the hot experts leave room for both).

- **One model thread.** A generation is a generator on the model thread
  that hands its runs over (core/speculative.py DeviceOp: a step or a verify
  group, or a pause between the parts of a prompt, RECLANK_LANE_PART 4096).
  The LaneScheduler (server/worker.py) resumes each lane under its state
  (QwenEngine.on: Lane.view swaps the model's per-lane attributes), runs the
  two runs as a pair when both lanes decode (verify_pair, step_pair), and
  alone otherwise; a prompt of one lane leaves the other
  RECLANK_LANE_PROMPT_SHARE (0.5) of the time between its parts.
- **The contexts.** The lanes share the contexts in host memory
  (RowCache.share_with): the context on a lane's device is that lane's, any
  other goes on from either lane (a copy: 160K positions in about 0.5 s,
  against minutes of prompt).
- **The route** (server/backend.py choose_lane, on the model thread):
  1. a lane the client names (x-reclank-lane, llama.cpp's id_slot);
  2. the lane whose device holds the conversation, if it is free (on a busy
     one it is the request in flight: a title or a request beside it goes
     to the other lane);
  3. the lane of the session, if free (a conversation compacted or edited
     takes the place of its own context, not another's);
  4. a free lane, the one whose last request ended first;
  5. all busy: the lane with the least wait (LaneScheduler.wait: the
     prompt tokens its requests have still to read, and 2048 for each one's
     answer, RECLANK_LANE_ANSWER_COST): a decode ends sooner than a long
     prompt. The conversation's or the session's lane wins by 512 tokens
     (its context's copy).
  The session: x-session-affinity / session_id (pi-ai, dsh's library, with
  compat.sendSessionAffinityHeaders: by default it sends none to a server
  that is not OpenAI's; dsh withholds that switch, its cacheRetention: long
  sends prompt_cache_key), x-session-id (OpenRouter's form, vLLM's router),
  x-claude-code-session-id, conversation_id (Codex), prompt_cache_key (the
  OpenAI API: Codex, OpenCode, pi-ai with long retention), metadata
  session_id or a user_id with session_ (Claude Code's Anthropic form). Not
  "user" (a user's, not a conversation's), not x-client-request-id (a new
  one each request elsewhere).

EXL3 5.05, 2 x 163840 positions: two streams (sampled, 400 tokens) 58-67
tok/s for both against 51-55 for one; one stream alone (greedy) about 6%
slower than one lane of 256K (53 fixed hot experts a layer in place of 62).

An agent's run (a parent of 23.8K tokens answers 320 tokens on lane B while
two subagents send new prompts of 11.6K and 11.9K): the pause between the
parts of a prompt held only while the other lane had a run pending, and a
lane that has just run has none, so with the prompt on lane A the decode got
one round a part (the parent 5.7 tok/s, 55.9 s); it holds now while the
other lane decodes (16.8 tok/s, 19.0 s). The second subagent waited behind
the first one's prompt on lane A (the oldest end); it waits now on the
decoding lane. All three done in 44.4 s in place of 55.9 s. Left: a part
(4096 tokens, ~8 s at ~500 tok/s) holds the other lane's decode for its
whole time (a short answer beside a prompt: ~3 tok/s); RECLANK_LANE_PART
sets it.

A prompt alone reads at 530-680 tok/s; with the page cache full (a download
of 500 GB writing to the HDD, 3 GB free) the kernel compacts memory for the
server's allocations and the same prompt reads at 50-70 tok/s (a 0.47 GB
context to host memory in 14 s in place of 0.07 s).

## 7. Memory

Lane B's cache takes its context's KV (5.8 GB for 262144 positions on EXL3
5.05): lane B gets a smaller context (`--lane-ctx`, 32768 by default, 0.7
GB), out of the pool of warm experts.
