# API reference

## Modeling

```python
Model(name: str = "model")
Model.add_var(name=None, *, lb=None, ub=None, vartype=VariableType.CONTINUOUS) -> Variable
Model.add_constraint(constraint: Constraint, name=None) -> Constraint
Model.minimize(expression) -> None
Model.maximize(expression) -> None
Model.relax(constraint_or_iterable, *, penalty: float) -> None
```

`VariableType` は `CONTINUOUS`、`INTEGER`、`BINARY` を持つEnumです。変数名と制約名は
省略でき、その場合は `var_0`、`constraint_0` のような一意で安定した名前を割り当てます。

変数と `LinearExpr` は、加減算、単項マイナス、有限スカラーとの乗除算、`<=`、`>=`、
`==` をサポートします。`quicksum()` は線形式のiterableを集約します。非線形積、異なる
モデル間の式、非有限値は拒否されます。

## Decomposition

```python
decomposition = model.decompose()
```

`Decomposition` は問題方向、目的定数、各部分問題の変数・ハード制約・目的項・緩和項、
緩和制約、成立する定数ハード制約を保持します。求解は行わず、有限境界も要求しません。

## Solving

```python
result = model.solve(
    highs_options=None,
    max_iterations=100,
    time_limit=None,
    parallel=False,
    workers=None,
    polyak_factor=1.0,
    initial_multipliers=None,
    callback=None,
)
```

- `highs_options` はhighspyのネイティブオプションです。`output_flag=False` が既定です。
  `time_limit` は全体時間管理に予約されているため、ここには指定できません。
- 全変数に有限な上下限が必要です。二値変数は省略時に `0/1` となります。
- `initial_multipliers` は `Mapping[Constraint, float]` です。不等式は `[0, penalty]`、
  等式は `[-penalty, penalty]` の範囲で指定します。
- `parallel=True` では部分問題をスレッドで並列求解します。各部分問題は独立した永続
  HiGHSインスタンスを持ち、結果は部分問題ID順に集約されます。
- 全部分問題が最適性を証明した反復だけが履歴・境界・乗数更新に採用されます。

## Results

```python
result.value(variable) -> float
result.violation(constraint) -> float
result.multiplier(constraint) -> float
```

同じモデルのハンドルを複数の結果へ渡せます。一覧用途として `variable_values`、
`violations`、`multipliers` の名前キー辞書も公開します。

不等式を `g(x) <= 0` に正規化したとき、違反量は `max(0, g(x))` です。等式では
`abs(h(x))` です。`hard_feasible` は全緩和制約の違反量が許容誤差以下であること、
`soft_feasible` はL1ペナルティ付き問題の実行可能解が得られたことを表します。

`IterationRecord` は反復ごとの主双対境界、残差、違反量、乗数、変数値、ステップ幅と
部分問題記録を保持します。`SubproblemRecord` はID、状態、目的寄与、所要時間を保持します。
