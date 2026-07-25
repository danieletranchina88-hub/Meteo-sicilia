# Benchmark indipendente dei fronti

Questa cartella definisce il processo con cui misurare, calibrare e confrontare
l'algoritmo dei fronti. I campi ICON-2I e le sue versioni smussate **non sono
verità osservata**: servono a generare la previsione. Anche GFS può essere usato
come controllo indipendente in modalità shadow, ma non come etichetta.

## Cosa archiviare

Per ogni caso si conservano:

- il GeoJSON pubblicato dall'algoritmo e il relativo `front_qc.json`;
- run, scadenza, versione del codice e soglie utilizzate;
- un GeoJSON etichettato manualmente, disegnato senza vedere il risultato
  dell'algoritmo;
- ora valida e fonte sinottica consultata (per esempio carta frontale ufficiale
  DWD o Met Office), con URL e data di accesso;
- note sui casi ambigui e sul disaccordo tra analisti.

Le immagini delle carte ufficiali non vanno ridistribuite se la licenza non lo
permette. Il repository conserva il riferimento alla fonte e le linee GeoJSON
prodotte dal progetto.

Struttura consigliata:

```text
benchmarks/fronts/archive/
  predictions/YYYY/MM/<case>.geojson
  labels/YYYY/MM/<case>.geojson
  qc/YYYY/MM/<case>.json
  manifest.json
```

L'azione GitHub archivia inoltre `front_qc.json` per ogni esecuzione operativa
come artifact a rotazione. Per un archivio pluriennale, copiare periodicamente
gli artifact in uno storage versionato e immutabile.

## Separazione temporale

I casi vicini nel tempo sono fortemente correlati. La divisione deve quindi
essere per blocchi temporali, non casuale:

- `train`: sviluppo delle regole;
- `validation`: scelta di soglie e dell'eventuale mappa score→probabilità;
- `test`: confronto finale, congelato e mai usato per scegliere parametri.

Il test deve coprire stagioni e regimi diversi: fronti intensi e deboli, giornate
senza fronte, cut-off, brezze, linee convettive e zone orografiche. Le giornate
senza fronte sono etichette valide, non dati mancanti.

Come prerequisito minimo per abilitare soglie mensili servono 30 run indipendenti
su almeno 15 giorni distinti. È soltanto un controllo di numerosità: l'attivazione
richiede comunque un miglioramento sul benchmark di validation senza degrado
materiale sul test congelato.

## Formato delle etichette

Ogni label è un GeoJSON `FeatureCollection`. Le geometrie sono `LineString` e
la proprietà `frontType` usa `cold`, `warm`, `occluded` o `stationary`.

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {
      "type": "LineString",
      "coordinates": [[8.1, 41.2], [10.0, 42.0]]
    },
    "properties": {
      "frontType": "cold",
      "analyst": "initials",
      "certainty": "medium"
    }
  }]
}
```

Un secondo analista dovrebbe etichettare un sottoinsieme alla cieca. La
distanza tra analisti stabilisce il limite realistico di accuratezza e aiuta a
scegliere il corridoio spaziale del benchmark.

## Esecuzione

```bash
python scripts/front_benchmark.py benchmarks/fronts/manifest.json \
  --split test --strict --radius-km 100 --min-overlap 0.60 \
  --radius-grid-km 50,75,100,150 \
  --out benchmark-test.json
```

Il matching è uno-a-uno e richiede sovrapposizione bidirezionale delle linee.
Il report contiene precision, recall, F1, intervalli di confidenza, accuratezza
del tipo e errore nel numero di fronti, più le metriche standard della
verifica previsionale:

- **POD** (probability of detection) — identica alla recall: quota di fronti
  etichettati che l'algoritmo ha trovato;
- **success ratio** — identico alla precision: quota di fronti previsti che
  corrispondono a un fronte etichettato;
- **CSI** (critical success index) — `TP / (TP + FP + FN)`: un solo numero
  che paga sia i falsi allarmi sia i fronti mancati.

### Metriche di lunghezza, distanza, continuità e tipo (schema v2)

Oltre ai conteggi, il report espone:

- **contabilità delle lunghezze** (`lengthAccountingKm`): `reference`,
  `predicted`, `detectedReference` (quanta lunghezza etichettata è coperta da
  una previsione entro il raggio), `missed`, `falsePredicted`;
- **`lengthRecall`** = detectedReference / reference e **`falseLengthRatio`**
  = falsePredicted / predicted: catturano gli errori di posizione che i soli
  TP/FP/FN nascondono (un fronte agganciato solo su un tratto breve non è
  rilevato per intero);
- **distanze simmetriche** media / mediana / 95° percentile fra le linee
  accoppiate (`meanSymmetricDistanceKm`, `medianSymmetricDistanceKm`,
  `p95SymmetricDistanceKm`);
- **continuità** (`continuity`): `splitEvents` (un fronte di riferimento
  spezzato in più previsioni) e `mergeEvents` (più fronti fusi in uno);
- **matrice di confusione dei tipi** (`typeConfusionMatrix`) sulle coppie
  accoppiate; il matching della presenza è **indipendente dal tipo**, la
  correttezza del tipo è valutata a parte (`matchTypePolicy`);
- **stratificazione** (`stratification`): POD / success ratio / CSI raggruppati
  per `season`, `localHour`, `surface`, `orography`, `lifecycle` e banda di
  lead (`0-12h` / `12-36h` / `36-72h`), quando i casi del manifest portano
  quei campi.

### Detector vs forecast (sez. 14)

`--mode detector` valuta solo i casi che nel manifest hanno `mode: detector`
(algoritmo applicato a ICON-2I-ASSIM o al campo più vicino a un'analisi →
misura soprattutto l'errore del **rilevatore**); `--mode forecast` valuta i
casi di previsione deterministica (errore di previsione + rilevazione).
Nessun altro modello numerico è usato come verità: le etichette restano
carte frontali umane.

### Sensibilità al raggio spaziale

Con `--radius-grid-km 50,75,100,150` il report include la sezione
`radiusSensitivity`: le stesse metriche ricalcolate a più tolleranze
spaziali, dalla più severa alla più permissiva. Serve a smascherare i
risultati "buoni" ottenuti solo con un raggio molto ampio: se POD e CSI
crollano passando da 150 a 50 km, l'algoritmo trova i fronti ma li
posiziona male, e il numero singolo a raggio largo lo nasconderebbe. Le
metriche vanno sempre lette insieme al raggio a cui sono state calcolate.

`qualityScore` resta un indice diagnostico. Può essere chiamato probabilità solo
dopo aver congelato un calibratore sulla validation e averlo verificato su
almeno 200 casi e 500 previsioni del test indipendente.

## Uso del secondo modello

GFS 0.25° è il confronto operativo consigliato perché è indipendente da
ICON-2I e dispone di un archivio ampio. Va interpolato sulla stessa area e
confrontato alle stesse ore/quote. In prima fase produce soltanto diagnostiche
shadow (accordo di posizione, gradiente e tipo): non deve cancellare un fronte
ICON né promuoverne uno. Il suo peso potrà essere deciso solo dai risultati
separati per stagione del benchmark.

Fonti dati precise:

- [NOMADS GFS 0.25°](https://nomads.ncep.noaa.gov/gribfilter.php?ds=gfs_0p25)
  per il sottoinsieme operativo recente;
- [NOAA GFS su AWS](https://registry.opendata.aws/noaa-gfs-pds/) per la finestra
  cloud recente a 0.25°;
- [archivio GFS NCEI](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast)
  per la disponibilità storica ufficiale e i relativi metodi di accesso;
- [ERA5 su Copernicus CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-complete)
  per la ricostruzione oraria storica su livelli di pressione;
- [archivio carte Met Office](https://www.metoffice.gov.uk/research/library-and-archive/archive/charts)
  per le analisi superficiali ASXX con posizione dei fronti.

ERA5 è adatto alla ricostruzione storica dei campi atmosferici, ma non contiene
una verità pronta dei fronti. Le etichette geometriche restano quelle umane,
tracciate da fonti sinottiche indipendenti.

## Acquisizione ufficiale DWD

Il DWD pubblica carte di analisi al suolo con fronti alle 00 e 12 UTC. Le
carte correnti restano sull'Open Data circa 48 ore; il *European
Meteorological Bulletin* degli ultimi sei mesi è disponibile in archivi ZIP.
Le annate dal 2002 possono essere richieste alla Biblioteca DWD.

Raccogliere le carte correnti senza scaricarle:

```bash
python scripts/official_chart_archive.py collect-current --dry-run
```

Archiviarle con URL, ora valida, impronta SHA-256 e data di accesso:

```bash
python scripts/official_chart_archive.py collect-current
```

Elencare i pacchetti mensili ufficiali EMB disponibili:

```bash
python scripts/official_chart_archive.py list-emb
```

Archiviare anche l'ultima analisi Met Office ASXX:

```bash
python scripts/official_chart_archive.py collect-metoffice
```

Il workflow `archive_front_references.yml` esegue la raccolta due volte al
giorno e conserva un artifact immutabile per 90 giorni. Le immagini non
entrano in Git; bisogna rispettare i termini di ciascun ente e mantenerne
l'attribuzione.

Fonti ufficiali:

- analisi correnti: <https://opendata.dwd.de/weather/charts/analysis/>;
- bollettini mensili: <https://download.dwd.de/pub/EMB/>;
- descrizione e archivio: <https://www.dwd.de/DE/leistungen/pbfb_verlag_emb/emb.html>;
- metodologia delle carte: <https://www.dwd.de/DE/fachnutzer/hobbymet/wetter_europa/allgemeines_analyse_prognosekarten_europa_neu.html>.

## Protocollo di etichettatura

Una PNG non è ancora un'etichetta geografica. Per ogni carta, l'analista deve
digitalizzare i fronti in GeoJSON rispettando queste regole:

1. non vedere la previsione dell'algoritmo prima di completare l'etichetta;
2. distinguere `cold`, `warm`, `occluded` e `stationary`;
3. registrare zone non valutabili e certezza bassa/media/alta;
4. conservare URL e SHA-256 della carta sorgente;
5. far riesaminare almeno il 20% dei casi da un secondo analista cieco;
6. non correggere il test congelato dopo aver osservato il risultato.

Il disaccordo tra due analisti è una misura scientifica utile: stabilisce il
limite realistico della localizzazione e impedisce di premiare differenze più
piccole dell'incertezza umana. L'estrazione automatica dei colori dalla carta
può precompilare una linea, ma non può diventare verità senza controllo umano:
simboli, occlusioni, scritte e sovrapposizioni rendono l'operazione ambigua.

## Gerarchia delle fonti

Non tutte le carte ufficiali hanno lo stesso ruolo scientifico:

- **gold label primaria:** analisi DWD al suolo e analisi Met Office ASXX,
  perché incorporano il giudizio sinottico di meteorologi e osservazioni;
- **seconda opinione:** quando DWD e Met Office coprono la stessa ora, il loro
  disaccordo viene registrato come incertezza dell'etichetta, non risolto
  scegliendo la carta più simile all'algoritmo;
- **shadow benchmark:** ECMWF Cyclone Database, front animation e front-density
  ensemble. Sono prodotti ufficiali e molto utili, ma derivano da rilevamento
  oggettivo su output ECMWF/Met Office: non sono verità umana indipendente e
  non devono essere mescolati con il gold score;
- **campi ausiliari:** OpenCharts ECMWF θw 850 hPa, satellite, radar e
  precipitazione aiutano la revisione meteorologica, ma da soli non etichettano
  una superficie frontale.

Riferimenti ufficiali:

- Met Office surface pressure charts e archivio ASXX:
  <https://weather.metoffice.gov.uk/research/library-and-archive/archive/charts>;
- ECMWF extratropical cyclone/front products:
  <https://www.ecmwf.int/en/forecasts/charts/extra-tropical-cyclones>;
- ECMWF OpenCharts:
  <https://charts.ecmwf.int/>.
