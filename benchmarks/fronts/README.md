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
  --split test --radius-km 100 --min-overlap 0.60 \
  --out benchmark-test.json
```

Il matching è uno-a-uno e richiede sovrapposizione bidirezionale delle linee.
Il report contiene precision, recall, F1, intervalli di confidenza, accuratezza
del tipo e errore nel numero di fronti.

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
