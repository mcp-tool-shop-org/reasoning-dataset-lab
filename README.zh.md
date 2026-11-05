<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**首先构建地图，然后用于处理谜题、测验和验证项目的数据集，其中代码在任何模型生成项目内容之前，就先证明答案的正确性。** `rdlab` 是命令行。它将数据集规划为一个预先注册的地图，构建出具有已构造和证明的键的项目，仅要求本地模型将其转化为文字，并记录每个项目的来源。

模型选择的键是一种观点。从隐藏状态构建，并由求解器证明的键，是关于该项目的事实。`rdlab` 的构建方式是，确保键永远不依赖于任何评判者。

## 安装

```bash
pip install reasoning-dataset-lab
```

Python 3.13 或更高版本（3.15 是主要目标；CI 测试 3.13、3.14 和 3.15）。它依赖于 [aspire-si-gates](https://pypi.org/project/aspire-si-gates/) 和 [role-set](https://pypi.org/project/role-set/)。

## 无需任何模型即可执行的操作

```bash
rdlab map check examples/t10-pilot-a.map.yaml
rdlab project fingerprint . --map examples/t10-pilot-a.map.yaml
rdlab --help
```

- `rdlab map check` 验证数据集地图（其轴、每个单元格的目标与生成器的容量、共享的上下限、多样性范围、验收标准），并打印地图的规范 SHA-256。退出代码 0 表示地图有效。退出代码 1 表示拒绝，以 `{code, message, hint}` JSON 对象的形式输出到标准错误流。退出代码 2 表示用法或文件错误。
- `rdlab project fingerprint` 以固定顺序对五个项目文件、地图和 `canon/` 进行哈希处理，以便构建过程可以准确地说明它是从哪些内容构建的。
- `rdlab models` 列出 Pilot B 构建过程中使用的模型，`rdlab variety` 计算构建过程的多样性报告。

## 需要 GPU 和本地模型才能执行的操作

`rdlab pilot-b` 构建 Pilot B 评测测试集：由来自类型化事实表的数据构建的支持、不支持和无法判断的三元组。它通过 Ollama 在环回模式下运行真实的本地模型（一个写入器和两个来自不同“家族”的盲读器），需要 GPU，并且需要一个许可证注册文件，该文件说明每个模型可以用于评估。它不会自动下载任何模型，也不会调用任何云端端点。完整的标志、内存计划和 `num_predict` 设置都可以在 [手册](https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/handbook/) 中找到。

## 评判者“烘焙”比赛

哪个本地模型应该评判某个声明？`rdlab` 的回答方式与它构建键的方式相同：通过构造，并在评判结束之前将答案密封。

```bash
rdlab bakeoff-part --output-dir <dir> ...      # build a sealed test part from certified value sets
rdlab bakeoff-probe --part-dir <dir> ...       # fail the part if its answers can be guessed from surface features
rdlab bakeoff-judge --part-dir <dir> ...       # run one judge without opening the sealed keys
rdlab bakeoff-judge-score --judge-dir <dir>    # only now join the judge's rows with the keys
```

获奖者选择完全遵循预注册，没有任何宽松的备选方案。`scripts/colab/launch_judges.sh` 在您自己的 Google Colab 帐户上运行这五个评判者，每个评判者都在其自己的哈希锁定的环境中运行，并受到来自实时 Colab 费率读取的支出限制。将一个已计数的运行拆分到多个卡片上已经构建，但尚未完成实际运行。

## “先地图”原则

1. **先证明，后表达。** 代码构建一个正式的状态，并在模型编写任何文本之前证明答案的正确性。无法通过构造证明的某种类型的项目将被标记为 `judged`，并且只能通过多“家族”一致性门进行处理。
2. **预注册的单元格和容量。** `map.yaml` 在任何项目存在之前，声明每个预期的单元格、其目标、生成器的容量、共享的上下限以及多样性范围。生成器目前无法构建的单元格将保留在地图中，并带有 `capacity: 0` 和说明原因，以便可以清楚地看到覆盖范围的差距，并且地图不会被缩小以隐藏它。
3. **唯一的指纹。** 项目绑定五个文件、地图及其规范，并使用确定性哈希来标识它们。

## 已测量的内容

生成器的值集（声明的值在其中进行交换的列表）在实际使用之前经过认证。在 v3 运行中，有 44 个（总共 50 个）集合通过了认证，而在集合修复之前，只有 26 个集合通过了认证。这里，“认证”意味着来自不同“家族”的两个盲读器都认为每个有序对都是矛盾的，并且一个独立的盲键已经同意。这是一种证据，而不是证明：模型读者是其中的一部分。渲染项目的措辞仍然可能与其键不同，并且由人工阅读来检查这一点。在人工检查之前，`rdlab` 编写的任何内容都不能用作训练数据。运行及其文件位于 [certification/v3](certification/v3) 中。

## 安全性和数据

- **读取：** 您指向的地图、项目文件和规范。
- **写入：** 您提供的输出目录。
- **网络：** `rdlab` 命令仅连接到环回模式下的 Ollama（`127.0.0.1`），并且仅用于 `pilot-b` 和 `models --verify-live`。任何其他操作都会被拒绝。没有遥测数据。
- **可选的 Colab：** `scripts/colab/` 将您指向的项目和评判者上传到您自己的 Colab 帐户，并下载结果。答案键永远不会离开您的机器；评分在本地运行。
- **权限：** 除了这些文件和本地端口之外，没有其他权限。

私下报告漏洞；请参阅 [SECURITY.md](SECURITY.md)。

## 状态

版本 0.5.0。这是一个 alpha 版本：接口和数据结构可能会在没有事先通知的情况下发生变化。1,162 个测试通过（2 个跳过）；CI 将覆盖率保持在 90% 或更高。

## 许可证

MIT，© mcp-tool-shop。请参阅 [LICENSE](LICENSE)。

---

由 [MCP Tool Shop](https://mcp-tool-shop.github.io/) 构建
