<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Un laboratorio basado en el principio de "primero el mapa" para conjuntos de datos de acertijos, cuestionarios y elementos de verificación, donde el código demuestra la clave de respuesta antes de que cualquier modelo formule el elemento.** `rdlab` es la línea de comandos. Planifica un conjunto de datos como un mapa preregistrado, construye elementos cuyas claves se construyen y se demuestran, solicita a un modelo local que simplemente los exprese y registra la procedencia de cada elemento.

Una clave que un modelo elige es una opinión. Una clave que el código construye a partir de un estado oculto y que un solucionador demuestra, es un hecho sobre el elemento. `rdlab` está diseñado de tal manera que la clave nunca depende de un juez.

## Instalación

```bash
pip install reasoning-dataset-lab
```

Python 3.13 o posterior (3.15 es el objetivo principal; las pruebas de CI son para 3.13, 3.14 y 3.15). Depende de [aspire-si-gates](https://pypi.org/project/aspire-si-gates/) y [role-set](https://pypi.org/project/role-set/).

## Qué se puede hacer sin ningún modelo

```bash
rdlab map check examples/t10-pilot-a.map.yaml
rdlab project fingerprint . --map examples/t10-pilot-a.map.yaml
rdlab --help
```

- `rdlab map check` valida un mapa de conjunto de datos (sus ejes, el objetivo de cada celda en relación con la capacidad del generador, los límites superior e inferior compartidos, los límites de variedad, los criterios de aceptación) e imprime el SHA-256 canónico del mapa. La salida 0 indica un mapa válido. La salida 1 indica un rechazo, como un objeto JSON `{code, message, hint}` en stderr. La salida 2 indica un error de uso o de archivo.
- `rdlab project fingerprint` calcula el hash de los cinco archivos del proyecto, el mapa y `canon/` en un orden fijo, de modo que una compilación pueda indicar exactamente a partir de qué se construyó.
- `rdlab models` enumera los modelos que una compilación de Pilot B utiliza, y `rdlab variety` calcula el informe de variedad determinista para una compilación.

## Qué necesita una GPU y modelos locales

`rdlab pilot-b` construye el conjunto de pruebas del crítico de Pilot B: tríadas de afirmaciones admitidas, no admitidas y para las que no se puede determinar, creadas a partir de tablas de hechos tipadas. Ejecuta modelos locales reales a través de Ollama en bucle cerrado (un escritor y dos lectores "ciegos" de diferentes familias), necesita una GPU y necesita un archivo de registro de licencias que indique que cada modelo puede utilizarse para la evaluación. Nunca descarga un modelo por usted ni llama a ningún punto final en la nube. Las opciones completas, el plan de memoria y la configuración `num_predict` se encuentran en el [manual](https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/handbook/).

## La prueba de jueces

¿Qué modelo local debe juzgar una afirmación? `rdlab` responde de la misma manera que construye las claves: por construcción, con las respuestas selladas hasta que finalice el juicio.

```bash
rdlab bakeoff-part --output-dir <dir> ...      # build a sealed test part from certified value sets
rdlab bakeoff-probe --part-dir <dir> ...       # fail the part if its answers can be guessed from surface features
rdlab bakeoff-judge --part-dir <dir> ...       # run one judge without opening the sealed keys
rdlab bakeoff-judge-score --judge-dir <dir>    # only now join the judge's rows with the keys
```

La selección del ganador sigue exactamente el preregistro, sin excepciones indulgentes. `scripts/colab/launch_judges.sh` ejecuta los cinco jueces en su propia cuenta de Google Colab, cada uno en su propio entorno protegido por hash, bajo un límite de gasto que se lee de la tasa en vivo de Colab. Dividir una ejecución contada en varias tarjetas está implementado, pero aún no se ha completado una ejecución real.

## El principio de "primero el mapa"

1. **Demuestre primero, formule después.** El código construye un estado formal y demuestra la clave antes de que un modelo escriba cualquier texto. Un tipo de elemento que no se puede demostrar por construcción se marca como `judged` y solo entra a través de una puerta de unanimidad multifamiliar.
2. **Celdas preregistradas y capacidad.** `map.yaml` declara cada celda prevista, su objetivo, la capacidad del generador para ella, los límites superior e inferior compartidos y los límites de variedad, antes de que exista cualquier elemento. Una celda que el generador aún no puede construir permanece en el mapa con `capacity: 0` y una razón declarada, de modo que una brecha de cobertura sea visible y el mapa nunca se reduzca para ocultarla.
3. **Una huella digital.** Un proyecto vincula cinco archivos, el mapa y su canon, y un hash determinista indica cuál.

## Qué se ha medido

Los conjuntos de valores del generador (las listas dentro de las cuales se intercambian los valores de una afirmación) se certifican antes de su uso. En la ejecución v3, se certificaron 44 de 50 conjuntos, en comparación con los 26 antes de que se repararan los conjuntos. Aquí, "certificado" significa que dos lectores "ciegos" de diferentes familias calificaron cada par ordenado como una contradicción, y una clave "ciega" independiente ya había estado de acuerdo. Es evidencia, no prueba: los lectores del modelo formaron parte de ella. La redacción de un elemento renderizado aún puede diferir de su clave, y la lectura manual de una persona es la comprobación de esto. Nada de lo que escribe `rdlab` es un dato de entrenamiento hasta que una persona lo haya comprobado. La ejecución y sus archivos se encuentran en [certification/v3](certification/v3).

## Seguridad y datos

- **Lecturas:** el mapa, los archivos del proyecto y el canon que se le indican.
- **Escrituras:** el directorio de salida que se le proporciona.
- **Red:** el comando `rdlab` solo se comunica con Ollama en bucle cerrado (`127.0.0.1`), y solo para `pilot-b` y `models --verify-live`. Cualquier otra cosa se rechaza. No hay telemetría.
- **Colab opcional:** `scripts/colab/` carga los elementos y los jueces que se le indican a su propia cuenta de Colab y descarga los resultados. Las claves de respuesta nunca abandonan su máquina; la puntuación se ejecuta localmente.
- **Permisos:** ninguno más allá de esos archivos y ese puerto local.

Informe de una vulnerabilidad de forma privada; consulte [SECURITY.md](SECURITY.md).

## Estado

Versión 0.5.0. Es una versión alfa: las interfaces y las formas de los datos pueden cambiar sin previo aviso. 1.162 pruebas superadas (2 omitidas); CI mantiene la cobertura en el 90% o más.

## Licencia

MIT, © mcp-tool-shop. Consulte [LICENSE](LICENSE).

---

Creado por [MCP Tool Shop](https://mcp-tool-shop.github.io/)
