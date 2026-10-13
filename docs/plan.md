# PuLP非依存モデラーへの全面移行計画

## Summary

- PuLP依存と既存のPuLP向けAPIを完全に削除し、独自 `Model` APIとhighspy直接接続へ置き換える。
- パッケージを `modeling`、`decomposition`、`algorithms`、`subproblems` の4層に分割し、将来ハンガリアン法やDPを追加できる依存構造にする。
- `slack` は実態に合わせて `violation` へ全面改名し、互換エイリアスは残さない。
- バージョンを `0.2.0` へ更新する。
- 本計画を `docs/plan.md` として保存し、以降の設計基準にする。

## Directory Structure

```text
src/descartes_mip/
├── __init__.py
├── exceptions.py
├── py.typed
├── modeling/
│   ├── __init__.py
│   ├── model.py
│   ├── expressions.py
│   └── ir.py
├── decomposition/
│   ├── __init__.py
│   ├── analyzer.py
│   └── records.py
├── algorithms/
│   ├── __init__.py
│   ├── lagrangian.py
│   └── records.py
└── subproblems/
    ├── __init__.py
    ├── base.py
    └── highs.py

tests/
├── modeling/
├── decomposition/
├── algorithms/
├── subproblems/
└── integration/

docs/
├── plan.md
├── design.md
└── api.md
```

- `modeling` はユーザー向けモデル記述とソルバー非依存IRだけを所有する。
  - `model.py`: `Model` とモデルのライフサイクル
  - `expressions.py`: `Variable`、`Constraint`、`LinearExpr`、演算子オーバーロード、`quicksum`
  - `ir.py`: 求解開始時に生成する不変の `ModelIR`、`SubproblemIR`、公開Enum
- `decomposition` はIRを読み、自動分解結果を生成する。モデルやソルバーの状態は変更しない。
- `algorithms` はラグランジュ反復、乗数更新、時間管理、並列集約、公開結果レコードを担当する。
- `subproblems` は部分問題の求解契約と実装を担当する。`highs.py` は唯一の初期実装とし、将来 `hungarian.py`、`dynamic_programming.py` などを同じ境界へ追加する。
- 依存方向は `modeling <- decomposition/subproblems <- algorithms <- root facade` とし、逆参照を禁止する。`Model.solve()` と `Model.decompose()` は遅延importで各サービスへ委譲し、循環importを避ける。
- ルート `__init__.py` は公開APIの再エクスポートだけに限定する。旧 `_model.py`、`solver.py`、`backends.py`、`types.py` は削除し、互換ラッパーは残さない。

## Public API

```python
from descartes_mip import Model, VariableType, quicksum

model = Model("example")
x = model.add_var(lb=0, ub=10, vartype=VariableType.INTEGER)
y = model.add_var(name="y", lb=0, ub=10)

model.minimize(2 * x + 3 * y)
coupling = model.add_constraint(x + y >= 10)
model.relax(coupling, penalty=100.0)

structure = model.decompose()
result = model.solve(
    highs_options={"mip_rel_gap": 0.0},
    max_iterations=100,
    time_limit=300.0,
)

print(result.value(x))
print(result.violation(coupling))
print(result.multiplier(coupling))
```

- `Model.add_var(name=None, lb=None, ub=None, vartype=VariableType.CONTINUOUS)` を提供する。
- `VariableType` は `CONTINUOUS`、`INTEGER`、`BINARY` の公開Enumとする。binaryは省略境界を `0/1` にし、明示境界は `[0, 1]` 内だけ許可する。
- 変数・線形式は加減算、単項マイナス、有限スカラーとの乗除算、`<=`・`>=`・`==`、組み込み `sum()` と `quicksum()` をサポートする。
- 非線形演算、ゼロ除算、非有限係数、異なるモデルの変数を混在させた式を拒否する。
- 変数名・制約名は任意とし、省略時は衝突しない `var_0`、`constraint_0` 形式の安定名を割り当てる。明示名の重複は各名前空間内で拒否する。
- `Model.minimize()` / `maximize()` は目的を設定または置換する。未設定モデルはゼロ目的の最小化として扱う。
- `Model.relax(constraint_or_iterable, penalty=...)` は登録済み制約だけを受け付ける。同じ制約の重複指定、他モデルの制約、非正・非有限ペナルティを拒否する。
- `Model.decompose()` は現在のモデルと緩和指定のスナップショットから診断用 `Decomposition` を返す。
- `Model.solve()` は `max_iterations`、`time_limit`、`parallel`、`workers`、`polyak_factor`、`callback` を維持する。初期乗数は `Mapping[Constraint, float]` で指定する。
- `highs_options` はhighspyのネイティブオプション名を受け付け、`output_flag=False` を既定とする。全体予算から設定する `time_limit` は予約し、辞書内での指定を拒否する。
- `SoftConstraint`、`SolverClass`、`SUPPORTED_SOLVERS`、トップレベル `solve()` / `decompose()` は削除する。

## Result and Violation Semantics

- `result.value(variable)` を変数値取得の主APIとし、同じ変数ハンドルを複数の結果へ渡して比較できるようにする。
- `result.violation(constraint)` は元制約の違反量を返す。
  - 正規化した `g(x) <= 0` に対して `max(0, g(x))`
  - 等式 `h(x) == 0` に対して `abs(h(x))`
- `result.multiplier(constraint)` は緩和対象制約の最終ラグランジュ乗数を返す。
- 一覧・ログ用途として、自動生成名を含む `variable_values`、`violations`、`multipliers` 辞書も提供する。
- `IterationRecord.slacks` と `LagrangianResult.slacks` は削除し、すべて `violations` に統一する。
- `hard_feasible` は全違反量が既存許容誤差以下かどうかで判定する。`soft_feasible` はL1ペナルティ付き問題の実行可能解が得られたことを表す。
- 別モデルのハンドル、求解スナップショット後に追加された変数、非緩和制約に対する `violation()` / `multiplier()` は明示的な検証エラーにする。

## Solver and Backend Implementation

- モデリング中は変数IDをキーにした疎な線形式を保持し、分解・求解開始時に不変IRへ変換する。IRにはPuLPやhighspyのオブジェクトを保持しない。
- `subproblems/base.py` にprivateな契約を定義する。
  - `SubproblemSolverFactory`: `SubproblemIR` から永続ソルバーをコンパイル
  - `CompiledSubproblem`: 目的係数更新、残り時間設定、求解
  - `SubproblemSolution`: ソルバー非依存の状態、目的値、変数値
- この契約にはHiGHS固有型を含めず、将来のハンガリアン法やDPが同じ入出力で実装できるようにする。ただし `0.2.0` では公開Protocolやソルバー選択引数を提供しない。
- `highs.py` は部分問題ごとに `highspy.Highs` を一度だけ構築し、列、行列、境界、整数性を一括登録する。
- 各ラグランジュ反復では目的係数全体と残り時間だけを更新して再求解し、モデルを再構築しない。
- 並列時は部分問題ごとに独立した永続HiGHSインスタンスを持たせ、同一インスタンスを同時実行しない。結果は部分問題ID順に決定論的に集約する。
- highspyの操作戻り値とモデル状態をすべて検査し、optimal以外をtime limit、infeasible、unbounded、solver errorへ分類する。
- 未完了反復の破棄、certified境界、全体wall-clock期限、目的方向の正規化、投影付きPolyak更新という既存の数理仕様は維持する。
- `solve()` では全変数に有限上下限を要求する。binary以外で省略境界が残っていれば変数ごとに不足側を報告する。`decompose()` にはこの制約を課さない。

## Packaging and Documentation

- `pyproject.toml` からPuLPを削除し、説明・キーワードを独自モデラーと自動ラグランジュ分解に合わせる。
- lockfileを再生成し、PuLPパッケージが依存グラフに残っていないことを確認する。
- バージョンを `0.2.0` に更新する。
- `docs/plan.md` に本計画を保存する。
- `docs/design.md` は層間依存、IR、分解、L1違反量、永続部分問題ソルバーの設計文書へ書き換える。
- `docs/api.md` とREADME、チュートリアルを新しいModel中心APIへ全面更新する。

## Test Plan

- `tests/modeling`: 式演算、`quicksum`、Enum、自動命名、名前衝突、クロスモデル参照、非線形・非有限入力、IRスナップショットを検証する。
- `tests/decomposition`: 連結成分、孤立変数、目的定数、定数制約、複数緩和制約、決定論的順序を検証する。
- `tests/algorithms`: L1違反量、乗数box、Polyak更新、主双対境界、最大化、callback、時間切れ、未完了反復破棄を検証する。
- `tests/subproblems`: テストダブルにより、部分問題が一度だけコンパイルされ、反復ごとに目的係数だけ更新されることを検証する。
- `tests/integration`: highspyで連続・整数・二値モデル、各制約方向、infeasible・unbounded・time limit、逐次・並列の一致を検証する。
- 同じ変数ハンドルについて複数の `LagrangianResult` を比較でき、アクセサーと名前辞書の値が一致することを確認する。
- ソース、テスト、文書、依存関係、lockfileにPuLPおよび旧 `slack` APIが残っていないことを検索で確認する。
- pytest、Ruff、format、ty、パッケージbuildを最終受け入れ条件とする。

## Assumptions

- 対象は線形目的・線形制約を持つMILPに限定する。
- L1ソフト制約、明示的な緩和対象指定、投影付きPolyak劣勾配法を維持する。
- ハンガリアン法、DP、自動構造認識、ユーザー実装ソルバーの公開差し込みAPIは後続リリースとする。
- PuLP互換アダプター、optional dependency、旧APIの非推奨期間は設けない。
