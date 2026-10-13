# Solver-independent automatic Lagrangian decomposition

## Architecture

実装は次の一方向の依存関係を持ちます。

```text
modeling <- decomposition ----\
       \                       -> algorithms <- root facade
        `---- subproblems ----/
```

- `modeling`: 公開モデラーと不変IR
- `decomposition`: 緩和制約を除いた変数–制約グラフの連結成分解析
- `subproblems`: 永続部分問題ソルバー契約とhighspy実装
- `algorithms`: ラグランジュ反復、時間管理、並列集約、結果

IRは整数の変数・制約IDと疎な線形項だけを保持し、モデラーやソルバーのオブジェクトを
参照しません。モデルは `decompose()` または `solve()` の開始時にスナップショット化されます。

## Decomposition

利用者が緩和指定した制約をグラフから除外し、残るハード制約で接続された変数を
Union-Findで成分化します。目的項と緩和制約の部分式は該当成分へ分配します。変数を持たない
ハード制約は成立すれば診断情報へ残し、不成立なら分解時に拒否します。

## Lagrangian algorithm

内部では最大化も最小化へ、すべての不等式も `g(x) <= 0` へ正規化します。不等式の
L1ペナルティ `rho max(0, g(x))` は乗数を `[0, rho]`、等式の `rho |h(x)|` は
`[-rho, rho]` に射影することと等価です。

各反復では、全ての部分問題が最適性を証明した場合だけcertified双対境界、ソフト問題の
実行可能主境界、Polyakステップ、次の乗数を確定します。未完了反復は破棄します。

## Persistent subproblem solvers

`SubproblemSolverFactory` は不変 `SubproblemIR` を `CompiledSubproblem` へ変換します。
コンパイル済み部分問題は目的係数と残り時間を受け取り、ソルバー非依存の状態・目的値・
変数値を返します。

HiGHS実装は行列、境界、整数性を最初に一度だけ登録します。反復ごとに `changeColsCost()` と
`time_limit` だけを更新します。この契約にはHiGHS型を含めないため、将来は同じ場所に
割当問題、動的計画法などの専用実装を追加できます。

## Guarantees and limits

- 対象は線形目的・線形制約を持つMILPです。
- `solve()` は全変数へ有限上下限を要求し、全乗数boxで部分問題が有界となる条件を保守的に保証します。
- 全体時間はmonotonic wall-clockで管理します。ソルバー停止精度による小さな超過は許容します。
- 並列モードでも各HiGHSインスタンスは一つの部分問題だけが所有し、集約順は決定論的です。
- v0.2.0ではHiGHSだけを公開実装とし、部分問題ソルバー選択APIは公開しません。
