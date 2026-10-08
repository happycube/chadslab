# Plan: a test framework

## 1. The problem

The repo has 51 check scripts (scripts/check_*.py) and no runner. Each
script is correct for the change that made it. But:

- No one runs them all. A check can fail for weeks with no sign.
- They do not record a result, so no history is available to compare with.
- Most of them test one call: a kernel, a step, a group. The faults that
  started this plan were in the state between calls:
  1. A GPU prompt of 8000 tokens or more, then decode steps: the rows of
     the window moved (GPUKV host_end, fixed in 8eeea07).
  2. A chat turn that cut the cache on the GPU: an illegal memory access,
     and the server failed each later request (fixed in 418838e).
  3. A cut that kept n but not the window rows before it: logits wrong by
     up to 14, with no error.
  None of the checks drove a Session through turns, and none ran a long
  prompt on the GPU and then compared the steps.
- A quality change can look good or bad for the wrong reason. On README
  text, int8 forms of the 26B gave 83% top-token agreement, but the model
  is not sure of 41% of those tokens. On a chat text the same forms gave
  98% to 99% (KL 0.002).
- The numbers of speed change with the load of the user and with the
  setting of intel_pstate.

## 2. The goals

1. One command runs a set of tests and writes a result file.
2. Tiers by cost: seconds, minutes, and hours.
3. Scenario tests of the state: the turns of a chat, the cuts, the long
   prompts, the switches between sessions, the server.
4. Quality tests on chat text with stored references, for each model,
   backend, and form.
5. GPU memory faults found at once, not after some requests.
6. The speed with bands and the load of the machine.
7. No new package. The venv has no pytest; the runner is a script, and
   the tests stay scripts that a person can also run alone.

## 3. The design

### 3.1 A result line

np_gemma/testkit.py gives each check the same end:

    from np_gemma import testkit
    testkit.result("gpu_session", ok, cut_1700_d=0.0, cut_40_d=0.0, reuse=8725)

It prints one JSON line with the prefix "RESULT ". The line has the name,
pass or fail, the metrics, the time, and the git commit. It also sets the
exit code. The 51 scripts keep their text output; each one gets the result
line at its end (one line of change in most of them).

### 3.2 The registry and the runner

tests/suite.py lists the tests. A test has:

    name        "gpu_session"
    cmd         ["scripts/check_gpu_session.py", "--cuts", "1700,1300,300,40,0"]
    tier        0, 1, 2, 3, or 4 (section 4)
    needs       model files, "gpu", GB of GPU memory, GB of RAM, "avx512"
    env         NP_GEMMA_* values
    paths       the source files it covers (for the selection by change)
    timeout     seconds

scripts/run_tests.py:

    python scripts/run_tests.py --tier 1              # all of tier 0 and tier 1
    python scripts/run_tests.py --changed             # the tests of the files in git diff
    python scripts/run_tests.py gpu_session mt        # by name
    python scripts/run_tests.py --tier 3 --model 26b

- It checks the needs first. A file that is not there, too little free
  GPU memory, or no AVX-512 gives SKIP with the reason, not FAIL. (The
  servers on ports 8080 and 8082 use GPU memory.)
- It runs the GPU tests one at a time (a lock file). CPU tests can run
  at the same time as a GPU test if the RAM allows.
- It writes results/<date>-<commit>.jsonl with each result line. The file
  also has the machine (CPU, GPU, driver), the load average before and
  after, the intel_pstate setting, and the free memory.
- It compares with the last run that passed: a new FAIL, a metric that
  moved past its band, a test that became SKIP. It prints a table and
  writes results/latest.md.
- It never stops the servers of the user.

### 3.3 The oracles

A test needs a true answer. The runtime has four kinds:

1. A float64 or NumPy reference (kernels): max relative difference.
2. The same bits by another path. Examples: a group against steps, the
   program against Python, and parts against one part. Also the GPU
   against the CPU program on the same rows, and MTP against greedy. Many checks
   already do this; it finds most faults of order and of state.
3. The same result from another history (metamorphic). The same final
   ids must give the same logits through a cut, a reuse, a session
   switch, or a fresh session. For the same bits, fix the hot set
   (NP_GEMMA_GPU_HOT_DYN=0). Also turn off the mixed groups
   (NP_GEMMA_GPU_MIX=0).
   check_gpu_session.py uses this form.
4. A reference model: transformers in float32 (or the float32 mode of the
   runtime) for quality, llama.cpp for answers and speed.

## 4. The tiers

### Tier 0: seconds, no model

- The kernels against float64: check_gelu, check_rms_norm, check_q6k,
  check_flash_c, check_int4_q8, check_ct_kernel, check_slide.
- The tokenizers and the chat templates: check_tokenizer, check_qwen_tok,
  check_chat_template.
- The server with a fake backend: check_server (the parse of tool calls,
  the stream, the stops, request_thinking, max_context).
- The cache logic without a model (new): the counts of KVCache and GPUKV
  under random operations (prepare, write, truncate, attach, detach,
  sync). After each operation the test checks:
  - base <= end and host_end <= end;
  - the cache holds every row that a query at end can see;
  - a cut that the cache takes keeps the rows of the window before it. A fake GPU buffer
  (a NumPy array with guard rows) finds a write before base.
- The style check of the Markdown files (check_ste100 with the baselines).

Run: before each commit (about 1 minute).

### Tier 1: minutes, one model, short prompts

The equivalence checks of each model (26B, E4B, Qwen3.6, Qwen3.8):

- the 26B and the E4B: check_program, check_mt, check_parts,
  check_gpu_split, check_gpu, check_mtp (ids);
- Qwen: check_qwen_gpu, check_qwen_gpu_groups, check_qwen4_gpu,
  check_qwen4_mtp;
- the AVX2 forms with NP_GEMMA_ARCH=avx2.

Run: before a commit that changes that model or its kernels
(--changed), and each night.

### Tier 2: scenarios of state (new)

The faults of section 1 were of this kind. The tests drive the objects
that the server uses, through many calls:

1. Session turns (scripts/check_sessions.py). For each model and
   backend, a long prompt past two windows and steps. Then turns that:
   - append, or cut at 0, 40, 300, 1300, and 1700 tokens;
   - switch to another session and back (the server keeps 4);
   - reach max_len, or use MTP.
   The oracle is 3.3 (3). check_gpu_session.py is the first case.
2. Random scenarios. A seed makes a list of operations (append n tokens,
   cut k tokens, switch session, MTP on or off, a step or a group). The
   test runs the list and compares each result with a reference session
   that makes the same rows with no cut. A failed seed goes into a list
   of fixed cases.
3. Long context. A prompt of 32K and of 80K tokens on the GPU. Then steps
   against the CPU program on the same rows (as check_gpu_prompt). Then a
   question on a fact at 10%, 50%, and 90% of the context, with a known
   answer.
4. The server in the process, with a real model and a real HTTP port on
   localhost. The requests:
   - chat turns with tools and thought parts, as a harness sends them;
   - a stream that the client closes;
   - two requests at the same time;
   - a prompt past --max-context.
   After each request the server must still answer.
5. Replay. The --debug records of the servers (~/serve-8082-turns) hold
   the requests of real harness sessions. A replay sends the requests of
   a record set in order to a Backend in the process. It checks that no
   error occurs, that the last turn gives the answer of a fresh session,
   and the count of reused tokens. A record set that found a fault stays as a fixed case
   (without private data: the user chooses which sets to keep).

Run: each night, and before a commit that changes model.py, gpu.py,
server.py, assistant.py, or a cache.

### Tier 3: quality

1. The texts (tests/texts/): 6 chat transcripts. Each is a prompt from
   apply_chat_template and the answer of the model in float32. The topics:
   code, prose, a tool call with its result, a math question, a text in
   German, and a long text (8K tokens). Later: an image and an audio question
   (MULTIMODAL_PLAN.md).
2. The references: the float32 mode of the runtime (and transformers
   where it runs) gives the top 64 logits of each position. They go in
   models2/test-refs/ (about 1 MB for each text and model); the repo keeps
   a manifest with the hash of each file.
3. The metrics: the mean KL over the top 64, the top-token agreement, the
   perplexity of the answer, and the positions that disagree with their
   confidence. Raw README text is a second metric only.
4. The matrix:
   - each model, on the CPU, the GPU, and AVX2;
   - the default forms and the switches of speed (int8 x, float16
     attention, the mixed groups, the hot sets);
   - MTP (the ids against plain).
5. The bands: each model has a limit for each metric (for example, the
   26B GPU default: KL <= 0.003, agreement >= 98%). A change that makes a
   metric worse than its band is a FAIL, also if the band still passes
   the old limit by chance. The band changes only with a note in the
   commit.
6. Tasks with a known answer (scripts/quality_suite.py,
   tests/quality/suite.py): 93 items through the server API, so the
   template, the think part and the parse of tool calls count as a client
   sees them. math (25, exact numbers), code (15 functions, unit tests in
   a subprocess with limits), tools (15: the tool and its arguments, no
   tool when none fits, an answer from a tool result), instruct (15
   checkable forms), mcq (20), needle (a passphrase at 3 depths of texts of
   --needle tokens, 4K/16K/64K by default). Each answer is also checked
   for faults (a finish by length, an empty answer, template tags in the
   content, no reasoning when a think part was asked for). Greedy by
   default; the runs (every answer, grade, note) go to ~/quality-runs;
   "compare" gives the scores side by side and the items that changed.
   tests/quality/validate.py checks the graders against known answers
   (reference solutions pass, wrong ones fail). A drop of 2 or more items
   is a FAIL. The same questions on llama.cpp give a second reference.

Run: each night on one model in turn, and before a change of a default.

### Tier 4: speed

- bench_llama_method.py (pp512, pp2048, pp8192, tg128) and bench_mtp for
  each model and backend. Each number has a band (for example 5%).
- The runner records the load average, and it marks a run with a load
  above 4 as not valid for speed (no FAIL). It gives the intel_pstate
  setting with the numbers.
- llama.cpp numbers from the same run, on the same load, where the
  build is there.

Run: on demand, and each week.

## 5. GPU memory faults

1. compute-sanitizer (CUDA 12.9 is installed): a tier of short GPU runs
   under memcheck (a small prompt, some steps, a cut, a group). It finds
   a write out of a buffer at the first kernel, also when the write does
   not stop the program. It is slow, so the runs are short.
2. A check mode (NP_GEMMA_GPU_CHECK=1):
   - Python asserts in GPUKV: prepare(pos) with pos >= base, the rows of
     each record inside the buffer, end <= cap.
   - After each graph, cudaDeviceSynchronize and the error of the last
     kernel, so the error names the record, not a later event.
3. The server: after a fatal CUDA error (an illegal address, an ECC
   error) the context is lost. The server must log the trace and stop
   with an exit code, so that a supervisor starts it again. Now it gives
   the same error to each later request. A tier 2 test checks this with
   a fault that a test switch makes.

## 6. The phases

### Phase 0: the result line and the list

1. np_gemma/testkit.py (result, the git commit, the times).
2. The result line in the 51 scripts. Mark the scripts that are research
   (check_expert_basis, check_freq, check_spectrum): they go in no tier.
3. tests/suite.py with each script, its tier, its needs, and its paths.
4. Test: each script gives its result line; the scripts that fail now
   are in a list with the reason.

### Phase 1: the runner

1. scripts/run_tests.py: the selection (tier, name, --changed, --model),
   the needs, the lock of the GPU, the result file, the comparison with
   the last pass, results/latest.md.
2. Test: tier 0 and tier 1 on all the models. A test that is made to
   fail shows FAIL. A model that is not on disk shows SKIP.

### Phase 2: the scenarios

1. The cache logic test of tier 0 (with the fake GPU buffer). Test: it
   finds the faults of 8eeea07 and 418838e on the old code.
2. check_sessions.py (the turns, the random scenarios, the fixed cases).
3. The long context tests and the server tests in the process.
4. The replay of the --debug records.
5. Test: each one finds the old faults on a copy of the old code.

### Phase 3: quality

1. The 6 texts and the tool for the references (tests/make_refs.py).
2. scripts/check_quality.py --model --backend --form, with the bands.
3. The 50 questions and their answers.
4. Test: the numbers of the README (26B int8 forms, KL 0.002 and 98.9%)
   come out again.

### Phase 4: the GPU faults

1. NP_GEMMA_GPU_CHECK and the compute-sanitizer tier.
2. The stop of the server after a fatal CUDA error.
3. Test: the cut of 418838e on the old code gives the error of the
   record at once.

### Phase 5: speed

1. The bands of speed, the load, and the pstate in the result file.
2. A week of runs to set the bands.

## 7. Risks

- The GPU is shared with the servers of the user. The GPU tiers need
  free memory; the runner gives SKIP, so a skip must not look like a pass
  (results/latest.md lists the skips first).
- Bit equality needs a fixed hot set and no mixed groups. Tests with the
  defaults need bands, not bits.
- The references of tier 3 must change when a model file changes; the
  manifest holds the hash of the model file too.
- Tier 2 and tier 3 take hours on all the models. The nightly run takes
  one model each night.
- The replay records hold the prompts of the user. They stay out of git
  unless the user chooses a set.
