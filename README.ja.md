<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/reasoning-dataset-lab/readme.png" alt="reasoning-dataset-lab" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/reasoning-dataset-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/reasoning-dataset-lab/"><img src="https://img.shields.io/pypi/v/reasoning-dataset-lab" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/"><img src="https://img.shields.io/badge/docs-landing_page-blue" alt="Landing page"></a>
</p>

**パズル、クイズ、検証項目のデータセットを扱うラボ。コードが、モデルが項目を記述する前に、回答の正当性を証明する。** `rdlab` はコマンドラインである。データセットを事前に登録されたマップとして計画し、キーが構築および検証された項目を作成し、ローカルモデルにそれらを言葉で表現するよう指示し、すべての項目の由来を記録する。

モデルが選択したキーは意見である。隠された状態から構築され、ソルバーによって検証されたコードは、項目に関する事実である。`rdlab` は、キーがジャッジに依存しないように構築されている。

## インストール

```bash
pip install reasoning-dataset-lab
```

Python 3.13 以降（3.15 が主なターゲット。CI テストは 3.13、3.14、および 3.15 を対象とする）。[aspire-si-gates](https://pypi.org/project/aspire-si-gates/) および [role-set](https://pypi.org/project/role-set/) に依存する。

## モデルを使用せずに実行できること

```bash
rdlab map check examples/t10-pilot-a.map.yaml
rdlab project fingerprint . --map examples/t10-pilot-a.map.yaml
rdlab --help
```

- `rdlab map check` は、データセットマップ（その軸、各セルのターゲットとジェネレーターの容量、共有の最小値と最大値、バリエーションの範囲、許容基準）を検証し、マップの標準的な SHA-256 を出力する。終了コード 0 は有効なマップを示す。終了コード 1 は拒否を示すもので、標準エラー出力に `{code, message, hint}` JSON オブジェクトとして出力される。終了コード 2 は、使用方法またはファイルエラーを示す。
- `rdlab project fingerprint` は、5 つのプロジェクトファイル、マップ、および `canon/` を固定された順序でハッシュ化するため、ビルドは、どのファイルからビルドされたかを正確に示すことができる。
- `rdlab models` は、Pilot B ビルドで使用するモデルをリストし、`rdlab variety` はビルドの決定的なバリエーションレポートを計算する。

## GPU とローカルモデルが必要な場合

`rdlab pilot-b` は、Pilot B のクリティックテストセットをビルドする。これは、型付きの事実テーブルから作成された、支持、不支持、および判断不能の主張の 3 つの組である。実際のローカルモデルを、Ollama を使用してループバックで実行する（異なるファミリーの 1 人の書き手と 2 人のブラインドリーダー）。GPU が必要であり、各モデルを評価に使用できることを示すライセンス登録ファイルも必要である。モデルを自動的にダウンロードしたり、クラウドエンドポイントを呼び出したりすることはない。すべてのフラグ、メモリプラン、および `num_predict` 設定は、[ハンドブック](https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/handbook/) に記載されている。

## マップファーストの原則

1. **まず証明し、次に記述する。** コードは、正式な状態を構築し、キーを証明してから、モデルが文章を作成する。構築によって証明できない種類の項目は、`judged` とマークされ、複数のファミリーによる一致ゲートを通過する。
2. **事前に登録されたセルと容量。** `map.yaml` は、すべての意図されたセル、そのターゲット、ジェネレーターの容量、共有の最小値と最大値、およびバリエーションの範囲を、項目が存在する前に宣言する。ジェネレーターがまだビルドできないセルは、マップ内に `capacity: 0` とともに、その理由とともに残るため、カバレッジのギャップが明確になり、マップが縮小されて隠されることはない。
3. **1 つのフィンガープリント。** プロジェクトは、5 つのファイル、マップ、およびその標準形式をバインドし、決定的なハッシュがそれを示す。

## 測定されたこと

ジェネレーターの値セット（主張の値が入れ替わるリスト）は、使用前に認証される。v3 の実行では、50 個のセットのうち 44 個が認証され、セットが修正される前の認証数は 26 個であった。「認証」とは、異なるファミリーの 2 人のブラインドリーダーが、すべての順序ペアを矛盾するものとして判断し、独立したブラインドキーがすでに同意したことを意味する。これは証拠であり、証明ではない。モデルリーダーもその一部であった。レンダリングされた項目の表現は、そのキーと異なる場合があり、人の手によるチェックがその確認を行う。`rdlab` が記述するものは、人がチェックするまでトレーニングデータにはならない。その実行とファイルは、[certification/v3](certification/v3) にある。

## セキュリティとデータ

- **読み取り:** マップ、プロジェクトファイル、および指定された標準形式。
- **書き込み:** 指定された出力ディレクトリ。
- **ネットワーク:** Ollama をループバックで使用するのみ（`127.0.0.1`）、および `pilot-b` と `models --verify-live` のためのみ。それ以外は拒否される。テレメトリーは行われない。
- **権限:** ファイルとローカルポートを超えた権限は必要ない。

脆弱性を非公開で報告してください。 [SECURITY.md](SECURITY.md) を参照してください。

## ステータス

バージョン 0.4.0。最初の PyPI リリース。これはアルファ版であり、インターフェイスとデータ構造は予告なしに変更される可能性がある。このリポジトリ自体のテストのカバレッジは 95% であり、最小値は 90% である。

## ライセンス

MIT、© mcp-tool-shop。 [LICENSE](LICENSE) を参照してください。

---

[MCP Tool Shop](https://mcp-tool-shop.github.io/) によって構築
