"""Reproducible model-building benchmarks kept separate from the test suite."""

from __future__ import annotations

import gc
import json
import platform
import statistics
import sys
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter_ns
from typing import Annotated, Any

import typer

from descartes_mip import Model, quicksum
from descartes_mip.subproblems.highs import HighsSubproblemFactory

type Operation = Callable[[], object]
type Prepare = Callable[[], Operation]

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Measure and compare descartes-mip model-building performance.",
)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    variables: int
    constraints: int
    nonzeros: int
    rounds: int
    median_seconds: float
    min_seconds: float
    peak_bytes: int


def _measure(
    name: str,
    prepare: Prepare,
    *,
    variables: int,
    constraints: int,
    nonzeros: int,
    rounds: int,
    warmups: int,
) -> BenchmarkResult:
    for _ in range(warmups):
        prepare()()

    samples: list[float] = []
    for _ in range(rounds):
        operation = prepare()
        gc.collect()
        gc.disable()
        try:
            started = perf_counter_ns()
            operation()
            samples.append((perf_counter_ns() - started) / 1_000_000_000)
        finally:
            gc.enable()

    operation = prepare()
    gc.collect()
    tracemalloc.start()
    try:
        operation()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return BenchmarkResult(
        name=name,
        variables=variables,
        constraints=constraints,
        nonzeros=nonzeros,
        rounds=rounds,
        median_seconds=statistics.median(samples),
        min_seconds=min(samples),
        peak_bytes=peak_bytes,
    )


def _build_model(size: int, width: int) -> Model:
    model = Model("benchmark")
    variables = [model.add_var(f"x_{index}", lb=0, ub=1) for index in range(size)]
    model.minimize(quicksum(variables))
    for row in range(size):
        terms = (variables[(row + offset) % size] for offset in range(width))
        model.add_constraint(quicksum(terms) <= width, f"c_{row}")
    return model


def _auto_names(size: int, _width: int) -> Prepare:
    def prepare() -> Operation:
        def operation() -> Model:
            model = Model("benchmark")
            for _ in range(size):
                model.add_var(lb=0, ub=1)
            return model

        return operation

    return prepare


def _quicksum(size: int, _width: int) -> Prepare:
    def prepare() -> Operation:
        model = Model("benchmark")
        variables = tuple(model.add_var(f"x_{index}", lb=0, ub=1) for index in range(size))
        return lambda: quicksum(variables)

    return prepare


def _add_constraints(size: int, width: int) -> Prepare:
    def prepare() -> Operation:
        model = Model("benchmark")
        variables = [model.add_var(f"x_{index}", lb=0, ub=1) for index in range(size)]
        constraints = tuple(
            quicksum(variables[(row + offset) % size] for offset in range(width)) <= width
            for row in range(size)
        )

        def operation() -> Model:
            for row, constraint in enumerate(constraints):
                model.add_constraint(constraint, f"c_{row}")
            return model

        return operation

    return prepare


def _snapshot(size: int, width: int) -> Prepare:
    def prepare() -> Operation:
        model = _build_model(size, width)
        return model._snapshot

    return prepare


def _decompose(size: int, width: int) -> Prepare:
    def prepare() -> Operation:
        model = _build_model(size, width)
        return model.decompose

    return prepare


def _highs_compile(size: int, width: int) -> Prepare:
    def prepare() -> Operation:
        decomposition = _build_model(size, width).decompose()
        factory = HighsSubproblemFactory()

        def operation() -> object:
            return tuple(
                factory.compile(decomposition._model, subproblem, {})
                for subproblem in decomposition._subproblem_irs
            )

        return operation

    return prepare


def _highs_update_solve(size: int, width: int) -> Prepare:
    def prepare() -> Operation:
        decomposition = _build_model(size, width).decompose()
        factory = HighsSubproblemFactory()
        solvers = tuple(
            factory.compile(decomposition._model, subproblem, {})
            for subproblem in decomposition._subproblem_irs
        )
        objectives = tuple((0.0,) * len(solver.variable_indices) for solver in solvers)

        def operation() -> object:
            return tuple(
                solver.solve(objective, None)
                for solver, objective in zip(solvers, objectives, strict=True)
            )

        return operation

    return prepare


BENCHMARKS: dict[str, Callable[[int, int], Prepare]] = {
    "auto_names": _auto_names,
    "quicksum": _quicksum,
    "add_constraints": _add_constraints,
    "snapshot": _snapshot,
    "decompose": _decompose,
    "highs_compile": _highs_compile,
    "highs_update_solve": _highs_update_solve,
}


class BenchmarkName(StrEnum):
    ALL = "all"
    AUTO_NAMES = "auto_names"
    QUICKSUM = "quicksum"
    ADD_CONSTRAINTS = "add_constraints"
    SNAPSHOT = "snapshot"
    DECOMPOSE = "decompose"
    HIGHS_COMPILE = "highs_compile"
    HIGHS_UPDATE_SOLVE = "highs_update_solve"


@app.command("run")
def run_benchmarks(
    benchmark: Annotated[
        list[BenchmarkName] | None,
        typer.Option(
            "--benchmark",
            "-b",
            help="Benchmark to run. Repeat to select multiple benchmarks.",
        ),
    ] = None,
    model_sizes: Annotated[
        list[int] | None,
        typer.Option(
            "--size",
            "-n",
            help="Model size. Repeat to measure multiple sizes.",
        ),
    ] = None,
    width: Annotated[int, typer.Option(min=1, help="Nonzeros per constraint.")] = 4,
    rounds: Annotated[int, typer.Option(min=1, help="Measured repetitions.")] = 5,
    warmups: Annotated[int, typer.Option(min=0, help="Warmup repetitions.")] = 1,
    output: Annotated[Path | None, typer.Option(help="Optional JSON result path.")] = None,
    label: Annotated[str | None, typer.Option(help="Revision or experiment label.")] = None,
) -> None:
    selected = [BenchmarkName.ALL] if benchmark is None else benchmark
    sizes = [100, 1_000] if model_sizes is None else model_sizes
    if BenchmarkName.ALL in selected and selected != [BenchmarkName.ALL]:
        raise typer.BadParameter("all cannot be combined with individual benchmarks")
    names = tuple(BENCHMARKS) if selected == [BenchmarkName.ALL] else tuple(selected)
    results: list[BenchmarkResult] = []
    for model_size in sizes:
        if model_size <= 0:
            raise typer.BadParameter("sizes must be positive", param_hint="--size")
        if width > model_size:
            raise typer.BadParameter(
                "width must be no greater than every size", param_hint="--width"
            )
        for name in names:
            constraints = model_size if name not in {"auto_names", "quicksum"} else 0
            nonzeros = (
                model_size if name == "quicksum" else model_size * width if constraints else 0
            )
            result = _measure(
                name,
                BENCHMARKS[name](model_size, width),
                variables=model_size,
                constraints=constraints,
                nonzeros=nonzeros,
                rounds=rounds,
                warmups=warmups,
            )
            results.append(result)
            typer.echo(
                f"{name:20} n={model_size:<8} median={result.median_seconds:>9.6f}s "
                f"peak={result.peak_bytes / 1_048_576:>8.2f} MiB"
            )

    payload = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "label": label,
            "width": width,
            "warmups": warmups,
        },
        "results": [asdict(result) for result in results],
    }
    if output is not None:
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"wrote {output}")


def _result_key(result: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(result["name"]),
        int(result["variables"]),
        int(result["constraints"]),
        int(result["nonzeros"]),
    )


@app.command("compare")
def compare_results(
    before_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    after_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    before_by_key = {_result_key(result): result for result in before["results"]}
    after_by_key = {_result_key(result): result for result in after["results"]}
    common = sorted(before_by_key.keys() & after_by_key.keys())
    if not common:
        raise typer.BadParameter("benchmark files contain no matching results")

    typer.echo("benchmark             variables   before (s)    after (s)   speedup   peak change")
    for key in common:
        old = before_by_key[key]
        new = after_by_key[key]
        old_time = float(old["median_seconds"])
        new_time = float(new["median_seconds"])
        speedup = old_time / new_time if new_time else float("inf")
        old_peak = int(old["peak_bytes"])
        new_peak = int(new["peak_bytes"])
        peak_change = (new_peak - old_peak) / old_peak if old_peak else 0.0
        typer.echo(
            f"{key[0]:20} {key[1]:>9} {old_time:>12.6f} {new_time:>12.6f} "
            f"{speedup:>8.2f}x {peak_change:>+11.1%}"
        )


if __name__ == "__main__":
    app()
