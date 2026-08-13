# descartes-mip

`descartes-mip` は、PuLPで記述したMILPから、利用者が指定したソフト制約を
ラグランジュ緩和し、独立な部分問題を自動生成・求解するためのライブラリです。

現在は段階的に開発中です。設計、数理的な契約、マイルストーンは
[設計文書](docs/design.md)を参照してください。

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
- [ ] Milestone 2: PuLPモデルの解析と自動分解
- [ ] Milestone 3: ラグランジュ反復求解
- [ ] Milestone 4: 時間制御・結果・障害処理
- [ ] Milestone 5: 並列求解と公開品質
