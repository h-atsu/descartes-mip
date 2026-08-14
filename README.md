# descartes-mip

`descartes-mip` は、PuLPで記述したMILPから、利用者が指定したソフト制約を
ラグランジュ緩和し、独立な部分問題を自動生成・求解するためのライブラリです。

設計と数理的な契約は[設計文書](docs/design.md)、公開インターフェースの詳細は
[APIリファレンス](docs/api.md)を参照してください。実行可能な一連の例は
[チュートリアル](examples/tutorial.py)にあります。

## インストール

```bash
pip install descartes-mip
```

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

分解した部分問題をHiGHSで求解し、投影付きPolyak劣勾配法で乗数を更新できます。
既定は逐次求解で、`parallel=True` を指定すると独立な部分問題を並列に求解します。

```python
from descartes_mip import solve

result = solve(
    problem,
    relax=[SoftConstraint(coupling, penalty=100.0)],
    solver=pulp.HiGHS,
    solver_options={"msg": False, "gapRel": 1e-9},
    max_iterations=100,
    time_limit=300.0,
    parallel=True,
    workers=2,
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

`time_limit` は `solve()` 全体のwall-clock予算です。各部分問題を解く直前に、指定された
ソルバークラスから残り時間付きの独立インスタンスを生成します。時間内に全部分問題の
最適性を証明できなかった反復は破棄し、直前までに完了した反復のcertified境界と解だけを返します。ソルバーの
停止精度により小さな超過はあり得ますが、期限を超えても反復全体が最適に解けた場合は、
その反復を有効な結果として採用します。

終了理由は `result.termination_reason`、実測時間は `result.elapsed_time`、時間切れになった
部分問題は `result.timed_out_subproblem` で確認できます。最初の反復が未完了の場合、
`result.has_certified_bounds` と `result.soft_feasible` はともに `False` です。

並列求解でも部分問題ごとに新しいソルバーを生成し、結果と履歴は部分問題ID順に
決定論的に集約します。サポートするソルバークラスは `pulp.HiGHS` と
`pulp.COIN_CMD` です。それ以外のクラスやソルバーインスタンスは
`ModelValidationError` で拒否します。固有設定は `solver_options` へ渡せますが、全体時間を
管理する `timeLimit` は予約済みなので、代わりに `solve()` の `time_limit` を指定してください。

## 開発環境

Python 3.12と[uv](https://docs.astral.sh/uv/)を使用します。

```bash
uv sync
uv run pytest
```

公式の検証用MILPソルバーはHiGHSです。モデリング層とソルバー接続にはPuLPを
利用し、CBCについても基本的な互換テストを行っています。

## 開発状況

- [x] Milestone 1: プロジェクト基盤と設計文書
- [x] Milestone 2: PuLPモデルの解析と自動分解
- [x] Milestone 3: ラグランジュ反復求解
- [x] Milestone 4: 時間制御・結果・障害処理
- [x] Milestone 5: 並列求解と公開品質
