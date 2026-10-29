<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Un environnement de test axé sur les données pour les ensembles de données de puzzles, de quiz et d’éléments de vérification, où le code prouve la clé de réponse avant que tout modèle ne formule l’élément.** `rdlab` est la ligne de commande. Il planifie un ensemble de données sous forme de schéma préenregistré, crée des éléments dont les clés sont construites et prouvées, demande à un modèle local de simplement les formuler, et enregistre la provenance de chaque élément.

Une clé choisie par un modèle est une opinion. Une clé construite à partir d’un état caché par le code et prouvée par un solveur est un fait concernant l’élément. `rdlab` est conçu de telle sorte que la clé ne dépende jamais d’un juge.

## Installation

```bash
pip install reasoning-dataset-lab
```

Python 3.13 ou version ultérieure (3.15 est l’objectif principal ; les tests CI portent sur les versions 3.13, 3.14 et 3.15). Il dépend de [aspire-si-gates](https://pypi.org/project/aspire-si-gates/) et de [role-set](https://pypi.org/project/role-set/).

## Ce que vous pouvez faire sans aucun modèle

```bash
rdlab map check examples/t10-pilot-a.map.yaml
rdlab project fingerprint . --map examples/t10-pilot-a.map.yaml
rdlab --help
```

- `rdlab map check` valide un schéma d’ensemble de données (ses axes, la cible de chaque cellule par rapport à la capacité du générateur, les limites minimales et maximales partagées, les limites de variété, les critères d’acceptation) et affiche le schéma canonique SHA-256. Une valeur de sortie de 0 indique un schéma valide. Une valeur de sortie de 1 indique un refus, sous la forme d’un objet JSON `{code, message, hint}` sur stderr. Une valeur de sortie de 2 indique une erreur d’utilisation ou de fichier.
- `rdlab project fingerprint` calcule le hachage des cinq fichiers du projet, du schéma et de `canon/` dans un ordre fixe, afin qu’une compilation puisse indiquer exactement à partir de quoi elle a été créée.
- `rdlab models` liste les modèles qu’une compilation Pilot B utilise, et `rdlab variety` calcule le rapport de variété déterministe pour une compilation.

## Ce qui nécessite une GPU et des modèles locaux

`rdlab pilot-b` crée l’ensemble de test critique Pilot B : des triades d’affirmations validées, non validées et pour lesquelles il est impossible de se prononcer, issues de tableaux de faits typés. Il exécute de véritables modèles locaux via Ollama en boucle (un rédacteur et deux lecteurs aveugles de familles différentes), nécessite une GPU et nécessite un fichier de registre de licences indiquant que chaque modèle peut être utilisé pour l’évaluation. Il ne télécharge jamais un modèle pour vous et n’appelle jamais de point de terminaison cloud. Les indicateurs complets, le plan de mémoire et les paramètres `num_predict` se trouvent dans le [manuel](https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/handbook/).

## Le principe de la priorité aux données

1. **Prouver d’abord, formuler ensuite.** Le code construit un état formel et prouve la clé avant qu’un modèle ne rédige un texte. Un type d’élément qui ne peut pas être prouvé par construction est marqué `judged` et n’est accessible que via une porte d’unanimité multi-familiale.
2. **Cellules et capacité préenregistrées.** `map.yaml` déclare chaque cellule prévue, sa cible, la capacité du générateur pour cette cellule, les limites minimales et maximales partagées, et les limites de variété, avant que tout élément n’existe. Une cellule que le générateur ne peut pas encore créer reste dans le schéma avec `capacity: 0` et une raison indiquée, de sorte qu’une lacune de couverture soit visible et que le schéma ne soit jamais réduit pour la masquer.
3. **Une seule empreinte.** Un projet lie cinq fichiers, le schéma et son canon, et un hachage déterministe indique lesquels.

## Ce qui a été mesuré

Les ensembles de valeurs du générateur (les listes dans lesquelles les valeurs d’une affirmation sont permutées) sont certifiés avant leur utilisation. Lors de l’exécution v3, 44 des 50 ensembles ont été certifiés, contre 26 avant que les ensembles ne soient réparés. Ici, « certifié » signifie que deux lecteurs aveugles de familles différentes ont déclaré que chaque paire ordonnée était une contradiction, et qu’une clé aveugle indépendante avait déjà donné son accord. Il s’agit d’une preuve, et non d’une simple indication : les lecteurs modèles en ont fait partie. La formulation d’un élément rendu peut encore différer de sa clé, et la lecture manuelle d’une personne est la vérification de cela. Rien de ce que `rdlab` écrit n’est une donnée d’entraînement tant qu’une personne ne l’a pas vérifié. L’exécution et ses fichiers se trouvent dans [certification/v3](certification/v3).

## Sécurité et données

- **Lecture :** le schéma, les fichiers de projet et le canon que vous lui indiquez.
- **Écriture :** le répertoire de sortie que vous lui donnez.
- **Réseau :** uniquement Ollama en boucle (`127.0.0.1`), et uniquement pour `pilot-b` et `models --verify-live`. Tout autre élément est refusé. Pas de télémétrie.
- **Autorisations :** aucune autorisation au-delà de ces fichiers et de ce port local.

Signalez une vulnérabilité en privé ; consultez [SECURITY.md](SECURITY.md).

## État

Version 0.4.0, la première version PyPI. Il s’agit d’une version alpha : les interfaces et les formes de données peuvent changer sans préavis. La couverture des tests de ce dépôt est de 95 %, avec un seuil minimum de 90 %.

## Licence

MIT, © mcp-tool-shop. Consultez [LICENSE](LICENSE).

---

Créé par [MCP Tool Shop](https://mcp-tool-shop.github.io/)
