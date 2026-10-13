# Model-building benchmarks

These benchmarks are intentionally separate from the pytest suite. They measure elapsed time and
peak Python allocation for model construction, IR conversion, decomposition, and HiGHS operations.

Run the complete suite:

```console
uv run python benchmarks/model_building.py run --output benchmark.json
```

Run selected benchmarks and sizes:

```console
uv run python benchmarks/model_building.py run \
  --benchmark auto_names --size 10000 --size 100000 --rounds 7 \
  --label after-name-counter --output after.json
```

Compare results produced from two revisions under the same Python environment:

```console
uv run python benchmarks/model_building.py compare before.json after.json
```

The reported time is the median of the measured rounds. Setup is excluded from the timed region.
Peak memory is measured in a separate run with `tracemalloc`, so tracing overhead does not affect
the elapsed-time samples. Use the same machine, environment, sizes, width, rounds, and warmups when
comparing revisions.
