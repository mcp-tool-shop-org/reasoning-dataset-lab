<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
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

**Um laboratório com foco no mapa para conjuntos de dados de quebra-cabeças, questionários e itens de verificação, onde o código comprova a resposta correta antes que qualquer
modelo formule o item.** `rdlab` é a linha de comando. Ele planeja um conjunto de dados como um mapa pré-registrado, cria itens cujas chaves
são construídas e comprovadas, solicita a um modelo local apenas que os formule e registra a origem de cada item.

Uma chave que um modelo escolheu é uma opinião. Uma chave que o código construiu a partir de um estado oculto e que foi comprovada por um solucionador é um fato
sobre o item. `rdlab` é construído de forma que a chave nunca dependa de um avaliador.

## Instalação

```bash
pip install reasoning-dataset-lab
```

Python 3.13 ou mais recente (3.15 é o objetivo principal; os testes de CI usam 3.13, 3.14 e 3.15). Depende de
[aspire-si-gates](https://pypi.org/project/aspire-si-gates/) e [role-set](https://pypi.org/project/role-set/).

## O que você pode fazer sem nenhum modelo

```bash
rdlab map check examples/t10-pilot-a.map.yaml
rdlab project fingerprint . --map examples/t10-pilot-a.map.yaml
rdlab --help
```

- `rdlab map check` valida um mapa de conjunto de dados (seus eixos, o alvo de cada célula em relação à capacidade do gerador, limites e tetos compartilhados, limites de variedade, critérios de aceitação) e imprime o SHA-256 canônico do mapa. A saída 0 é um mapa válido. A saída 1 é uma rejeição, como um objeto JSON `{code, message, hint}` em stderr. A saída 2 é um erro de uso ou arquivo.
- `rdlab project fingerprint` calcula o hash dos cinco arquivos do projeto, do mapa e de `canon/` em uma ordem fixa, para que uma construção possa indicar
exatamente a partir do que foi construída.
- `rdlab models` lista os modelos que uma construção Pilot B usa e `rdlab variety` calcula o relatório de variedade determinística
para uma construção.

## O que precisa de uma GPU e modelos locais

`rdlab pilot-b` cria o conjunto de testes de avaliação Pilot B: tríades de afirmações suportadas, não suportadas e que não podem ser determinadas, criadas a partir
de tabelas de fatos tipadas. Ele executa modelos locais reais por meio do Ollama em loopback (um escritor e dois leitores cegos de famílias diferentes), precisa de uma GPU e precisa de um arquivo de registro de licenças que indique que cada modelo pode ser usado para avaliação. Ele nunca baixa
um modelo para você e nunca chama um endpoint na nuvem. As flags completas, o plano de memória e as configurações `num_predict` estão em
[handbook](https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/handbook/).

## O princípio do "mapa em primeiro lugar"

1. **Comprove primeiro, formule depois.** O código constrói um estado formal e comprova a chave antes que um modelo escreva qualquer texto. Um
tipo de item que não pode ser comprovado por construção é marcado como `judged` e entra apenas por meio de um filtro de unanimidade multifamiliar.
2. **Células e capacidade pré-registradas.** `map.yaml` declara cada célula pretendida, seu alvo, a capacidade do gerador
para ela, limites e tetos compartilhados e os limites de variedade, antes que qualquer item exista. Uma célula que o gerador ainda não consegue
construir permanece no mapa com `capacity: 0` e uma razão declarada, para que uma lacuna de cobertura seja visível e o mapa nunca seja
reduzido para escondê-la.
3. **Uma impressão digital.** Um projeto vincula cinco arquivos, o mapa e seu cânone, e um hash determinístico indica qual é.

## O que foi medido

Os conjuntos de valores do gerador (as listas dentro das quais os valores de uma afirmação são trocados) são certificados antes do uso. Na execução v3, 44
de 50 conjuntos foram certificados, em comparação com 26 antes que os conjuntos fossem reparados. Aqui, "certificado" significa que dois leitores cegos de famílias diferentes consideraram que cada par ordenado era uma contradição, e uma chave cega independente já havia concordado. É evidência, não
prova: os leitores do modelo fizeram parte disso. A formulação de um item renderizado ainda pode diferir de sua chave, e a leitura manual de uma pessoa é a verificação disso. Nada que `rdlab` escreve é um dado de treinamento até que uma pessoa o tenha verificado. A execução e seus
arquivos estão em [certification/v3](certification/v3).

## Segurança e dados

- **Leituras:** o mapa, os arquivos do projeto e o cânone que você aponta para ele.
- **Escritas:** o diretório de saída que você fornece.
- **Rede:** apenas Ollama em loopback (`127.0.0.1`), e apenas para `pilot-b` e `models --verify-live`. Qualquer outra coisa
é rejeitada. Sem telemetria.
- **Permissões:** nenhuma além desses arquivos e dessa porta local.

Relate uma vulnerabilidade em particular; consulte [SECURITY.md](SECURITY.md).

## Status

Versão 0.4.0, a primeira versão PyPI. É alfa: as interfaces e as formas de dados podem mudar sem aviso prévio. A cobertura dos próprios testes deste repositório é de 95%, com o limite em 90.

## Licença

MIT, © mcp-tool-shop. Consulte [LICENSE](LICENSE).

---

Criado por [MCP Tool Shop](https://mcp-tool-shop.github.io/)
