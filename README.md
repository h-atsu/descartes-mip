# descartes-mip

`descartes-mip` は、ソルバー非依存の軽量な線形モデラーと、自動ラグランジュ分解を
提供するPythonライブラリです。指定した結合制約を取り除いた変数–制約グラフから
独立な部分問題を抽出し、既定ではhighspyを直接利用して求解します。

設計は[設計文書](docs/design.md)、公開インターフェースは
[APIリファレンス](docs/api.md)、移行計画は[実装計画](docs/plan.md)を参照してください。

## Installation

```bash
pip install descartes-mip
```

Python 3.12以上とHiGHSが必要です。PuLPには依存しません。

## Quick start

```python
from descartes_mip import Model, VariableType

model = Model("production")
plant_a = model.add_var("plant_a", lb=0, ub=8, vartype=VariableType.INTEGER)
plant_b = model.add_var("plant_b", lb=0, ub=8, vartype=VariableType.INTEGER)

model.minimize(3 * plant_a + 5 * plant_b)
demand = model.add_constraint(plant_a + plant_b >= 10, "demand")
model.relax(demand, penalty=20)

decomposition = model.decompose()
print([part.variables for part in decomposition.subproblems])

result = model.solve(max_iterations=100, time_limit=30, parallel=True, workers=2)
print(result.best_dual_bound, result.best_primal_bound)
print(result.value(plant_a), result.value(plant_b))
print(result.violation(demand), result.multiplier(demand))
```

`Model.solve()` は部分問題ごとに永続的なHiGHSモデルを一度だけ構築します。ラグランジュ
反復では目的係数と残り時間だけを更新するため、モデル行列を毎回作り直しません。

すべての決定変数には有限な上下限が必要です。二値変数だけは省略時に `0/1` が設定されます。
緩和制約にはL1違反ペナルティを適用し、最小化では `best_primal_bound` がソフト問題の
実行可能上界、`best_dual_bound` がcertified下界です。最大化では向きが反転します。

## Development

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check src
uv build
```
