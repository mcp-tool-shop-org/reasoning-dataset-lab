<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Un approccio "mappa prima di tutto" per set di dati di enigmi, quiz ed elementi di verifica, in cui il codice dimostra la correttezza delle risposte prima che qualsiasi modello elabori l'elemento.** `rdlab` è la riga di comando. Definisce un set di dati come una mappa pre-registrata, crea elementi le cui chiavi vengono costruite e verificate, richiede a un modello locale di tradurli in linguaggio naturale e registra la provenienza di ogni elemento.

Una chiave scelta da un modello è un'opinione. Una chiave costruita dal codice a partire da uno stato nascosto e verificata da un risolutore è un dato di fatto relativo all'elemento. `rdlab` è progettato in modo che la chiave non dipenda mai da un valutatore.

## Installazione

```bash
pip install reasoning-dataset-lab
```

Python 3.13 o versione successiva (3.15 è l'obiettivo principale; i test CI utilizzano le versioni 3.13, 3.14 e 3.15). Dipende da [aspire-si-gates](https://pypi.org/project/aspire-si-gates/) e [role-set](https://pypi.org/project/role-set/).

## Cosa puoi fare senza alcun modello

```bash
rdlab map check examples/t10-pilot-a.map.yaml
rdlab project fingerprint . --map examples/t10-pilot-a.map.yaml
rdlab --help
```

- `rdlab map check` convalida una mappa del set di dati (i suoi assi, l'obiettivo di ogni cella rispetto alla capacità del generatore, i limiti inferiore e superiore, i limiti di varietà, i criteri di accettazione) e stampa la rappresentazione canonica SHA-256 della mappa. L'uscita 0 indica una mappa valida. L'uscita 1 indica un rifiuto, con un oggetto JSON `{code, message, hint}` su stderr. L'uscita 2 indica un errore di utilizzo o di file.
- `rdlab project fingerprint` calcola l'hash dei cinque file di progetto, della mappa e di `canon/` in un ordine fisso, in modo che una build possa indicare esattamente da cosa è stata creata.
- `rdlab models` elenca i modelli a cui fa riferimento una build Pilot B e `rdlab variety` calcola il report di varietà deterministico per una build.

## Cosa richiede una GPU e modelli locali

`rdlab pilot-b` crea il set di test critico Pilot B: triadi di affermazioni supportate, non supportate e per le quali non è possibile esprimersi, tratte da tabelle di fatti tipizzate. Esegue modelli locali reali tramite Ollama in loopback (un writer e due lettori "ciechi" provenienti da famiglie diverse), richiede una GPU e richiede un file di registro delle licenze che indichi che ogni modello può essere utilizzato per la valutazione. Non scarica mai un modello per te e non chiama mai un endpoint cloud. Tutte le opzioni, il piano di memoria e le impostazioni `num_predict` sono disponibili nel [manuale](https://mcp-tool-shop-org.github.io/reasoning-dataset-lab/handbook/).

## La "gara" dei valutatori

Quale modello locale dovrebbe valutare un'affermazione? `rdlab` risponde nello stesso modo in cui crea le chiavi: tramite la costruzione, con le risposte sigillate fino al termine della valutazione.

```bash
rdlab bakeoff-part --output-dir <dir> ...      # build a sealed test part from certified value sets
rdlab bakeoff-probe --part-dir <dir> ...       # fail the part if its answers can be guessed from surface features
rdlab bakeoff-judge --part-dir <dir> ...       # run one judge without opening the sealed keys
rdlab bakeoff-judge-score --judge-dir <dir>    # only now join the judge's rows with the keys
```

La selezione del vincitore segue esattamente le regole stabilite durante la preregistrazione, senza prevedere eccezioni o deroghe. `scripts/colab/launch_judges.sh` esegue
i cinque giudici sul proprio account Google Colab, ciascuno in un ambiente protetto tramite hash, nel rispetto di un limite di spesa definito in base al
tasso di utilizzo attuale di Colab, rilevato direttamente dall’ambiente. Ogni ambiente a pagamento è associato ai file specifici e verifica ogni input all’avvio
prima di procedere, e il controllo di sicurezza viene eseguito in due ambienti separati, con ciascun giudice nel proprio ambiente. La possibilità di suddividere un’esecuzione tra più schede è stata implementata, ma non è ancora stata testata in una situazione reale.

## Il principio "mappa prima di tutto"

1. **Dimostra prima, esprimi dopo.** Il codice costruisce uno stato formale e dimostra la chiave prima che un modello scriva qualsiasi testo. Un tipo di elemento che non può essere dimostrato tramite la costruzione è contrassegnato con `judged` e viene incluso solo tramite un meccanismo di unanimità multi-familiare.
2. **Celle e capacità pre-registrate.** `map.yaml` dichiara ogni cella prevista, il suo obiettivo, la capacità del generatore per quella cella, i limiti inferiore e superiore e i limiti di varietà, prima che esista qualsiasi elemento. Una cella che il generatore non può ancora creare rimane nella mappa con `capacity: 0` e una motivazione dichiarata, in modo che una lacuna di copertura sia visibile e la mappa non venga mai ridotta per nasconderla.
3. **Un'impronta digitale.** Un progetto associa cinque file, la mappa e la sua rappresentazione canonica, e un hash deterministico indica quali.

## Cosa è stato misurato

Gli insiemi di valori del generatore (gli elenchi all'interno dei quali vengono scambiati i valori di un'affermazione) sono certificati prima dell'uso. Nell'esecuzione v3, 44 insiemi su 50 sono stati certificati, rispetto ai 26 prima della correzione degli insiemi. Qui, "certificato" significa che due lettori "ciechi" provenienti da famiglie diverse hanno definito ogni coppia ordinata come una contraddizione e una chiave "cieca" indipendente ha già concordato. Si tratta di una prova, non di una dimostrazione: i lettori del modello hanno fatto parte del processo. La formulazione di un elemento renderizzato può ancora differire dalla sua chiave e la lettura manuale di una persona è il controllo per questo. Nulla di ciò che `rdlab` scrive è un dato di addestramento finché una persona non lo ha controllato. L'esecuzione e i suoi file sono disponibili in [certification/v3](certification/v3).

## Sicurezza e dati

- **Letture:** la mappa, i file di progetto e la rappresentazione canonica a cui lo si indirizza.
- **Scritture:** la directory di output che gli si fornisce.
- **Rete:** il comando `rdlab` raggiunge solo Ollama in loopback (`127.0.0.1`) e solo per `pilot-b` e `models --verify-live`. Qualsiasi altra cosa viene rifiutata. Nessuna telemetria.
- **Colab con consenso:** `scripts/colab/` carica gli elementi e i valutatori a cui lo si indirizza sul proprio account Colab e scarica i risultati. Le chiavi di risposta non lasciano mai la tua macchina; il punteggio viene eseguito localmente.
- **Autorizzazioni:** nessuna, oltre a tali file e a quella porta locale.

Segnala una vulnerabilità in privato; consulta [SECURITY.md](SECURITY.md).

## Stato

Versione 0.6.0. Si tratta di una versione alpha: le interfacce e le strutture dei dati possono cambiare senza preavviso. 1.316 test superati (3 saltati); il sistema di integrazione continua mantiene
la copertura dei test al 90% o superiore.

## Licenza

MIT, © mcp-tool-shop. Consulta [LICENSE](LICENSE).

---

Creato da [MCP Tool Shop](https://mcp-tool-shop.github.io/)
