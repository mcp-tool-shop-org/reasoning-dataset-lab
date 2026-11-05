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

**パズル、クイズ、検証項目のデータセットを扱うラボで、まずマップを構築します。コードが、モデルが項目を記述する前に、回答の正しさを証明します。** `rdlab` はコマンドラインです。データセットを事前に登録されたマップとして計画し、キーが構築および検証された項目を作成し、ローカルモデルにそれらを言葉で表現するよう指示し、すべての項目の由来を記録します。

モデルが選択したキーは意見です。隠された状態から構築され、ソルバーによって検証されたコードが作成したキーは、その項目に関する事実です。`rdlab` は、キーが審査員に依存しないように構築されています。

## インストール

```bash
pip install reasoning-dataset-lab
```

Python 3.13 以降（3.15 が主なターゲットです。CI テストでは 3.13、3.14、および 3.15 を使用します）。[aspire-si-gates](https://pypi.org/project/aspire-si-gates/) および [role-set](https://pypi.org/project/role-set/) に依存します。

## モデルを使用せずに実行できること

```bash
rdlab map check examples/t10-pilot-a.map.yaml
rdlab project fingerprint . --map examples/t10-pilot-a.map.yaml
rdlab --help
```

- `rdlab map check` は、データセットマップ（その軸、各セルのターゲットとジェネレーターの容量、共有の最小値と最大値、バリエーションの範囲、許容基準）を検証し、マップの標準的な SHA-256 を出力します。終了コード 0 は有効なマップを示します。終了コード 1 は拒否を示し、標準出力に `{code, message, hint}` JSON オブジェクトとして出力されます。終了コード 2 は、使用方法またはファイルエラーを示します。
- `rdlab project fingerprint` は、5 つのプロジェクトファイル、マップ、および `canon/` を固定された順序でハッシュ化するため、ビルドは、どのファイルからビルドされたかを正確に把握できます。
- `rdlab models` は、Pilot B ビルドで使用するモデルをリストし、`rdlab variety` はビルドの決定的なバリエーションレポートを計算します。

## GPU とローカルモデルが必要な場合

`rdlab pilot-b` は、Pilot B の批判的テストセットを構築します。これは、型付きの事実テーブルから作成された、支持、不支持、および判断不能の主張の 3 つの組み合わせです。実際のローカルモデルを、ループバック経由で Ollama を使用して実行します（1 つの書き込みモデルと、異なるファミリーからの 2 つのブラインドリーダー）。GPU が必要であり、各モデルが評価に使用できることを示すライセンス登録ファイルも必要です。モデルを自動的にダウンロードしたり、クラウドエンドポイントに接続したりすることはありません。すべてのフラグ、メモリプラン、および `num_predict` 設定は、[ハンドブック](https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/handbook/) に記載されています。

## 審査員の選考

どのローカルモデルが主張を審査すべきですか？ `rdlab` は、キーを構築する方法と同じように、構築によって回答を封印し、審査が完了するまで回答を明らかにしないことで、この質問に答えます。

```bash
rdlab bakeoff-part --output-dir <dir> ...      # build a sealed test part from certified value sets
rdlab bakeoff-probe --part-dir <dir> ...       # fail the part if its answers can be guessed from surface features
rdlab bakeoff-judge --part-dir <dir> ...       # run one judge without opening the sealed keys
rdlab bakeoff-judge-score --judge-dir <dir>    # only now join the judge's rows with the keys
```

勝者の選定は、事前に登録された内容を厳密に守り、寛容な代替手段はありません。`scripts/colab/launch_judges.sh` は、5 人の審査員を、独自の Google Colab アカウント内の、それぞれがハッシュでロックされた環境で、ライブ Colab のレートから読み取った支出制限の下で実行します。複数のカードに分散して実行することも可能ですが、まだ実際の実行は完了していません。

## マップファーストの原則

1. **まず証明し、次に言葉で表現する。** コードは、正式な状態を構築し、キーを証明してから、モデルが文章を記述します。構築によって証明できない種類の項目は、`judged` とマークされ、複数のファミリーによる一致ゲートを通過することでのみ追加されます。
2. **事前に登録されたセルと容量。** `map.yaml` は、意図されたすべてのセル、そのターゲット、ジェネレーターの容量、共有の最小値と最大値、およびバリエーションの範囲を、項目が存在する前に宣言します。ジェネレーターがまだ構築できないセルは、マップ内に `capacity: 0` とともに、その理由とともに保持されるため、カバレッジのギャップが明確になり、マップが縮小されて隠されることはありません。
3. **1 つのフィンガープリント。** プロジェクトは、5 つのファイル、マップ、およびその標準形式をバインドし、決定的なハッシュによって、どのファイルが使用されたかを識別します。

## 測定されたこと

ジェネレーターの値セット（主張の値が入れ替えられるリスト）は、使用前に認証されます。v3 の実行では、50 個のセットのうち 44 個が認証され、セットが修正される前の認証数は 26 個でした。ここで「認証」とは、異なるファミリーの 2 人のブラインドリーダーが、すべての順序ペアを矛盾するものとして判断し、独立したブラインドキーがすでに同意したことを意味します。これは証拠であり、証明ではありません。モデルリーダーもその一部でした。レンダリングされた項目の表現は、そのキーと異なる場合があり、人の手による確認がそのチェックになります。`rdlab` が記述するものは、人が確認するまでトレーニングデータにはなりません。実行とそのファイルは、[certification/v3](certification/v3) にあります。

## セキュリティとデータ

- **読み取り:** マップ、プロジェクトファイル、および指定された標準形式。
- **書き込み:** 指定された出力ディレクトリ。
- **ネットワーク:** `rdlab` コマンドは、ループバック上の Ollama（`127.0.0.1`）にのみアクセスし、`pilot-b` および `models --verify-live` のためにのみアクセスします。それ以外は拒否されます。テレメトリーはありません。
- **オプトイン Colab:** `scripts/colab/` は、指定された項目と審査員を、独自の Colab アカウントにアップロードし、結果をダウンロードします。回答のキーは、ローカルでスコアリングが行われるため、マシンから離れることはありません。
- **権限:** これらのファイルとローカルポートを超えた権限は必要ありません。

脆弱性を非公開で報告してください。 [SECURITY.md](SECURITY.md) を参照してください。

## ステータス

バージョン 0.5.0。これはアルファ版です。インターフェイスとデータ構造は、予告なしに変更される可能性があります。1,162 個のテストが合格しました（2 個はスキップ）。CI は、90% 以上のカバレッジを維持しています。

## ライセンス

MIT、© mcp-tool-shop。 [LICENSE](LICENSE) を参照してください。

---

[MCP Tool Shop](https://mcp-tool-shop.github.io/) によって構築されました。
