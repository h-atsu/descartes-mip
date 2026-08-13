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
部分式を確認できます。現時点では構造解析のみで、反復求解APIはMilestone 3で追加します。

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
- [ ] Milestone 3: ラグランジュ反復求解
- [ ] Milestone 4: 時間制御・結果・障害処理
- [ ] Milestone 5: 並列求解と公開品質
