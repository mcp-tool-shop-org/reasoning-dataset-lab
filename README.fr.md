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

**Un laboratoire axé sur les cartes pour les ensembles de données de puzzles, de quiz et d’éléments de vérification, où le code prouve la clé de réponse avant que tout modèle ne formule l’élément.** `rdlab` est la ligne de commande. Il planifie un ensemble de données sous forme de carte préenregistrée, crée des éléments dont les clés sont construites et prouvées, demande à un modèle local de simplement les formuler, et enregistre la provenance de chaque élément.

Une clé choisie par un modèle est une opinion. Une clé construite à partir d’un état caché par un code, et prouvée par un solveur, est un fait concernant l’élément. `rdlab` est conçu de telle sorte que la clé ne dépende jamais d’un juge.

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

- `rdlab map check` valide une carte d’ensemble de données (ses axes, la cible de chaque cellule par rapport à la capacité du générateur, les limites minimales et maximales, les limites de variété, les critères d’acceptation) et affiche la carte canonique SHA-256. Une valeur de sortie de 0 indique une carte valide. Une valeur de sortie de 1 indique un refus, sous la forme d’un objet JSON `{code, message, hint}` sur stderr. Une valeur de sortie de 2 indique une erreur d’utilisation ou de fichier.
- `rdlab project fingerprint` calcule le hachage des cinq fichiers du projet, de la carte et de `canon/` dans un ordre fixe, afin qu’une compilation puisse indiquer exactement à partir de quoi elle a été créée.
- `rdlab models` liste les modèles qu’une compilation Pilot B utilise, et `rdlab variety` calcule le rapport de variété déterministe pour une compilation.

## Ce qui nécessite une GPU et des modèles locaux

`rdlab pilot-b` crée l’ensemble de tests du critique Pilot B : des triades d’affirmations validées, non validées et pour lesquelles il est impossible de se prononcer, tirées de tables de faits typées. Il exécute de véritables modèles locaux via Ollama en boucle (un rédacteur et deux lecteurs aveugles de familles différentes), nécessite une GPU et nécessite un fichier de registre de licences qui indique que chaque modèle peut être utilisé pour l’évaluation. Il ne télécharge jamais un modèle pour vous et n’appelle jamais de point de terminaison cloud. Les indicateurs complets, le plan de mémoire et les paramètres `num_predict` se trouvent dans le [manuel](https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/handbook/).

## Le concours des juges

Quel modèle local doit-il juger une affirmation ? `rdlab` y répond de la même manière qu’il crée des clés : par construction, avec les réponses scellées jusqu’à la fin du jugement.

```bash
rdlab bakeoff-part --output-dir <dir> ...      # build a sealed test part from certified value sets
rdlab bakeoff-probe --part-dir <dir> ...       # fail the part if its answers can be guessed from surface features
rdlab bakeoff-judge --part-dir <dir> ...       # run one judge without opening the sealed keys
rdlab bakeoff-judge-score --judge-dir <dir>    # only now join the judge's rows with the keys
```

La sélection du gagnant suit exactement le processus de pré-inscription, sans possibilité de dérogation. `scripts/colab/launch_judges.sh` exécute les cinq juges sur votre propre compte Google Colab, chacun dans son propre environnement protégé par un hachage, dans le respect d’une limite de dépenses définie à partir du taux Colab actuel de la fenêtre. Chaque fenêtre payante est associée aux fichiers de sa partie et vérifie chaque entrée au démarrage avant de procéder, et la vérification du verrou est effectuée dans deux fenêtres, chaque juge étant dans son propre environnement. La possibilité de répartir une exécution comptée sur plusieurs cartes est intégrée, mais n’a pas encore été testée en conditions réelles.

## Le principe de la carte en premier

1. **Prouvez d’abord, formulez ensuite.** Le code construit un état formel et prouve la clé avant qu’un modèle n’écrive un quelconque texte. Un type d’élément qui ne peut pas être prouvé par construction est marqué `judged` et n’y entre que par le biais d’une porte d’unanimité multi-familiale.
2. **Cellules et capacité préenregistrées.** `map.yaml` déclare chaque cellule prévue, sa cible, la capacité du générateur pour cette cellule, les limites minimales et maximales, et les limites de variété, avant que tout élément n’existe. Une cellule que le générateur ne peut pas encore créer reste dans la carte avec `capacity: 0` et une raison indiquée, de sorte qu’une lacune de couverture soit visible et que la carte ne soit jamais réduite pour la masquer.
3. **Une seule empreinte.** Un projet lie cinq fichiers, la carte et son canon, et un hachage déterministe indique lesquels.

## Ce qui a été mesuré

Les ensembles de valeurs du générateur (les listes dans lesquelles les valeurs d’une affirmation sont permutées) sont certifiés avant leur utilisation. Lors de l’exécution v3, 44 des 50 ensembles ont été certifiés, contre 26 avant la réparation des ensembles. Ici, « certifié » signifie que deux lecteurs aveugles de familles différentes ont déclaré que chaque paire ordonnée était une contradiction, et qu’une clé aveugle indépendante avait déjà été d’accord. Il s’agit d’une preuve, et non d’une simple indication : les lecteurs modèles en ont fait partie. La formulation d’un élément rendu peut encore différer de sa clé, et la lecture manuelle d’une personne est la vérification de cela. Rien de ce qu’écrit `rdlab` n’est une donnée d’entraînement tant qu’une personne ne l’a pas vérifié. L’exécution et ses fichiers se trouvent dans [certification/v3](certification/v3).

## Sécurité et données

- **Lectures :** la carte, les fichiers de projet et le canon que vous lui indiquez.
- **Écritures :** le répertoire de sortie que vous lui indiquez.
- **Réseau :** la commande `rdlab` n’atteint que Ollama en boucle (`127.0.0.1`), et uniquement pour `pilot-b` et `models --verify-live`. Tout le reste est refusé. Pas de télémétrie.
- **Colab avec consentement :** `scripts/colab/` télécharge les éléments et les juges que vous lui indiquez vers votre propre compte Colab et télécharge les résultats. Les clés de réponse ne quittent jamais votre machine ; le calcul des scores s’effectue localement.
- **Autorisations :** aucune autorisation au-delà de ces fichiers et de ce port local.

Signalez une vulnérabilité en privé ; voir [SECURITY.md](SECURITY.md).

## État

Version 0.6.0. Il s’agit d’une version alpha : les interfaces et les formats de données peuvent changer sans préavis. 1 316 tests réussissent (3 sont ignorés) ; l’intégration continue maintient une couverture de 90 % ou plus.

## Licence

MIT, © mcp-tool-shop. Voir [LICENSE](LICENSE).

---

Créé par [MCP Tool Shop](https://mcp-tool-shop.github.io/)
