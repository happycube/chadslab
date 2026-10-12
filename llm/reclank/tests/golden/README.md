# Golden dumps

The check of the refactor: the tokens and the logits of each model family
on each backend must stay bit for bit the same when code moves.

1. Copy models.example.json to OUT/models.json and give the paths of the
   model files (outside this tree).
2. `tests/golden/run_golden.sh OUT` runs every family on every backend
   (each in its own process) and writes OUT/<backend>/<family>.{npz,json,log}.
   GOLDEN_CPUS and GOLDEN_THREADS set the CPU set and the OpenMP team. Keep
   them the same for the runs that are compared.
3. `python tests/golden/golden.py compare OUT_BEFORE OUT_AFTER` checks
   each pair: the same tokens, the same sha256 of the logits of each step,
   and the same tokens of the server path. It also gives the speed of
   both runs.

See golden.py for what a run does. The dumps are data: keep them out of
the repository.
