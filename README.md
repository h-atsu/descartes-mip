# descartes-mip

`descartes-mip` は、PuLPで記述したMILPから、利用者が指定したソフト制約を
ラグランジュ緩和し、独立な部分問題を自動生成・求解するためのライブラリです。

現在は段階的に開発中です。設計、数理的な契約、マイルストーンは
[設計文書](docs/design.md)を参照してください。

## 現在利用できるAPI

PuLPモデルから、指定した制約を除いた変数–制約グラフの連結成分を抽出できます。

```python
import pulp

from descartes_mip import SoftConstraint, decompose

problem = pulp.LpProblem("example", pulp.LpMinimize)
x = problem.add_variable("x", lowBound=0)
y = problem.add_variable("y", lowBound=0)
problem += x + 2 * y
coupling = x + y >= 10
problem += coupling, "coupling"

decomposition = decompose(
    problem,
    relax=[SoftConstraint(coupling, penalty=100.0)],
)

for subproblem in decomposition.subproblems:
    print(subproblem.id, subproblem.variables)
```

`Decomposition` は診断用の不変データで、変数、ハード制約、目的係数、緩和制約の
部分式を確認できます。

分解した部分問題をHiGHSで逐次求解し、投影付きPolyak劣勾配法で乗数を更新できます。

```python
from descartes_mip import solve

result = solve(
    problem,
    relax=[SoftConstraint(coupling, penalty=100.0)],
    max_iterations=100,
    time_limit=300.0,
)

print(result.best_primal_bound)
print(result.best_dual_bound)
print(result.variable_values)
```

最小化では `best_primal_bound` がソフト問題の実行可能上界、
`best_dual_bound` がcertified下界です。最大化ではそれぞれ実行可能下界とcertified上界に
なります。各部分問題が最適性を証明できない場合、その時点で
`SubproblemSolveError` を送出します。

v1では、`solve()` に渡すモデルの全決定変数に、有限な下限と上限を明示する必要が
あります。これにより、ペナルティ由来のbox内にある任意のラグランジュ乗数について、
すべての部分問題が有界であることを単純に保証します。ハード制約から上下限を推論する
処理は行わないため、暗黙に有界な変数にも明示的な境界を設定してください。

`time_limit` は `solve()` 全体のwall-clock予算です。各部分問題を解く直前に残り時間を
`solver_factory(remaining_time)` へ渡します。時間内に全部分問題の最適性を証明できなかった
反復は破棄し、直前までに完了した反復のcertified境界と解だけを返します。ソルバーの
停止精度により小さな超過はあり得ますが、期限を超えても反復全体が最適に解けた場合は、
その反復を有効な結果として採用します。

終了理由は `result.termination_reason`、実測時間は `result.elapsed_time`、時間切れになった
部分問題は `result.timed_out_subproblem` で確認できます。最初の反復が未完了の場合、
`result.has_certified_bounds` と `result.soft_feasible` はともに `False` です。

## 開発環境

Python 3.12と[uv](https://docs.astral.sh/uv/)を使用します。

```bash
uv sync
uv run pytest
```

公式の検証用MILPソルバーはHiGHSです。モデリング層とソルバー接続にはPuLPを
利用します。

## 開発状況

- [x] Milestone 1: プロジェクト基盤と設計文書
- [x] Milestone 2: PuLPモデルの解析と自動分解
- [x] Milestone 3: ラグランジュ反復求解
- [x] Milestone 4: 時間制御・結果・障害処理
- [ ] Milestone 5: 並列求解と公開品質
