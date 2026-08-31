# Rete di stazioni meteorologiche — stato, audit e architettura

Questo documento accompagna l'introduzione del package
`meteo_analysis/observations/` e riassume l'audit della rete di stazioni,
le cause del problema "molte stazioni senza dati" e l'architettura
multi-provider ora in produzione.

## 1. Architettura trovata inizialmente

Il progetto è un sito statico (GitHub Pages, `index.html` con MapLibre GL)
alimentato da una pipeline Python (`scripts/process_data.py`) eseguita da
GitHub Actions (`update_meteo.yml`, `collect_observations.yml`). Non esiste
un database relazionale/PostGIS né un backend applicativo: tutto lo stato
osservativo era un singolo file `data_weather/observations.json` rigenerato
ad ogni run.

**Prima di questo intervento, l'unica fonte di osservazioni al suolo era il
catalogo METAR di NOAA Aviation Weather Center** (`fetch_italy_metar_observations`
in `meteo_analysis/verification/observations.py`). METAR è una rete
aeroportuale: in Italia conta poche decine di stazioni, molte delle quali
militari o minori che trasmettono un report ogni 30-60 minuti o restano
silenti per ore. Questo — non un bug di parsing — è la causa principale del
sintomo "molte stazioni senza dati": **la rete monitorata semplicemente non
copriva il territorio nazionale**, non un guasto di singole stazioni.

Il modello ICON-2I era già acquisito da MeteoHub (Agenzia ItaliaMeteo) in
formato open-data non autenticato
(`https://meteohub.agenziaitaliameteo.it/api/datasets/.../opendata`), quindi
l'ente e l'infrastruttura erano già familiari al progetto per i *campi*
modello — non ancora per le *osservazioni* puntuali.

## 2. Cosa era già funzionante (non toccato/non degradato)

- Acquisizione e archiviazione GRIB ICON-2I, analisi frontale, algoritmi di
  pericolo, motore sinottico, meteogrammi, modello ML dei fronti: invariati.
- `StationForecastArchive`/`bilinear_sample` (campionamento ICON-2I nativo
  alle stazioni) e l'intero `meteo_analysis/verification/`: invariati,
  continuano a leggere lo stesso schema `stations`/`stationNetwork`.
- Fusione osservazione↔modello lato client (Cressman) in `index.html`:
  invariata nella logica, riceve dati aggiuntivi ma nello stesso formato.
- Clustering della mappa stazioni: **era già implementato** con i cluster
  nativi MapLibre (`cluster: true`, `clusterMaxZoom: 8`); non è stato
  necessario reinventarlo, solo mantenerlo funzionante con più stazioni.

## 3. Cosa è stato aggiunto in questo intervento

### 3.1 Architettura provider (`meteo_analysis/observations/`)

- `providers/base.py`: interfaccia `ObservationProvider` + `ProviderResult`.
  Ogni provider è isolato: un errore o l'assenza di credenziali di un
  provider non impedisce agli altri di pubblicare dati (`safe_fetch`).
- `providers/metar_provider.py`: adatta la rete METAR esistente allo schema
  canonico, **senza modificarne la logica di normalizzazione già verificata**.
- `providers/italiameteo_provider.py`: adapter per Agenzia ItaliaMeteo /
  MeteoHub (rete nazionale ~4.500 stazioni telemisurate). **Disattivato per
  default** finché non sono presenti le variabili d'ambiente
  `METEOHUB_BASE_URL` e `METEOHUB_API_TOKEN` (registrazione richiesta su
  MeteoHub; questo ambiente sandbox non ha accesso di rete per verificare lo
  schema JSON esatto, quindi l'estrazione dei campi è difensiva — prova più
  nomi di chiave plausibili — e va validata con un account reale).
- `providers/meteonetwork_provider.py`: adapter per MeteoNetwork (API REST
  v3, `https://api.meteonetwork.it/v3`, login con email/password →
  bearer token). **Disattivato per default** finché non sono presenti
  `METEONETWORK_EMAIL`+`METEONETWORK_PASSWORD` oppure `METEONETWORK_TOKEN`.
  Le stazioni MeteoNetwork restano sempre `source = "meteonetwork"`, con un
  peso di qualità di base inferiore a quello delle reti istituzionali.
- `registry.py`: registro canonico con deduplicazione **non distruttiva**:
  due stazioni vicine ricevono un *confidence score* di corrispondenza e solo
  sopra una soglia alta (0.85) la stazione di priorità inferiore viene
  marcata `duplicateOfStationId`, senza mai essere rimossa dal registro.
- `quality.py`: controllo qualità (physical bounds + consistenza temporale:
  spike futuri, dati troppo vecchi, sensore bloccato) con `qualityScore`.
- `health.py`: classificazione LIVE/DELAYED/STALE/OFFLINE/UNRELIABLE con
  soglie di latenza **specifiche per provider** (METAR 60 min attesi,
  ItaliaMeteo 40 min, MeteoNetwork 20 min) e diagnostica di rete
  (`stations_total`, `stations_reporting_15m/30m/60m/3h`, `stations_stale`,
  `stations_offline`, per fonte e per regione).
- `coverage.py`: copertura territoriale su griglia (0.1° di default,
  configurabile), distanza dalla stazione più vicina, celle coperte entro
  5/10/20/50 km, frazione di *coverage gap*, per l'intera rete e per
  parametro (temperatura, pioggia, vento, umidità, pressione).
- `pipeline.py`: orchestratore `collect_national_observations()`, unico
  punto d'ingresso. Ricostruisce lo schema legacy (`stations`,
  `stationNetwork`, `count`) per compatibilità totale con il codice
  esistente e aggiunge `registry`, `diagnostics`, `coverage`,
  `providerStatus`.

### 3.2 Pipeline di raccolta

- `scripts/process_data.py`: `collect_observations()`/`write_observations()`
  ora usano `collect_national_observations()`. Se **nessun** provider
  risponde, l'ultimo snapshot pubblicato in `data_weather/observations.json`
  **non viene cancellato né sovrascritto con un payload vuoto**
  (requisito "fallback provider"): resta online, chiaramente datato dal suo
  stesso `capturedAt`.
- `scripts/collect_observations.py` (job orario di archiviazione storica):
  usa la stessa pipeline e archivia uno snapshot multi-provider per run.

### 3.3 Frontend (`index.html`)

- `fetchObservations()` legge anche `payload.registry.stations` (se
  presente) in una nuova variabile `stationRegistry`, oltre al vecchio
  `stationNetwork` (retrocompatibilità totale con payload precedenti).
- Le stazioni duplicate (`duplicateOfStationId` valorizzato) non vengono
  disegnate due volte sulla mappa.
- Colore/stato marker ora riflette lo *station health* (LIVE verde, DELAYED
  ambra, STALE ocra, OFFLINE grigio, UNRELIABLE rosso) invece del solo
  binario live/delayed/offline calcolato lato client.
- La scheda stazione mostra **solo i parametri realmente disponibili**
  (niente righe "N/D" per sensori assenti) più fonte/rete, quality score ed
  età del dato in minuti.
- Nuovo filtro "Filtro stazioni" (select): Tutte, Solo LIVE, Istituzionali,
  MeteoNetwork, oppure per parametro disponibile (temperatura, pioggia,
  vento, pressione, umidità).

## 4. Variabili d'ambiente necessarie (non committate, da configurare)

| Provider | Variabili | Note |
|---|---|---|
| ItaliaMeteo/MeteoHub | `METEOHUB_BASE_URL`, `METEOHUB_API_TOKEN` (opz. `METEOHUB_STATIONS_PATH`, `METEOHUB_OBSERVATIONS_PATH`) | Richiede registrazione gratuita su MeteoHub; percorso REST esatto da confermare dopo login (questo ambiente non ha accesso di rete al portale). |
| MeteoNetwork | `METEONETWORK_EMAIL` + `METEONETWORK_PASSWORD`, oppure `METEONETWORK_TOKEN` | Account gratuito su `my.meteonetwork.it`; API v3 documentata via Swagger (`api.meteonetwork.it/v3/swagger`). |

Nessuna di queste variabili è obbligatoria: senza di esse la pipeline
continua a funzionare esattamente come prima (solo METAR), riportando
`providerStatus.<source>.configured = false` invece di fallire.

## 5. Cosa resta da fare (bloccato da credenziali/infrastruttura esterna, non da questo ambiente)

1. **Validare lo schema JSON reale di MeteoHub e MeteoNetwork** con un
   account attivo e adattare `_FIELD_ALIASES` se necessario (l'estrazione
   attuale è difensiva ma non verificata su una risposta live).
   2. Valutare l'accesso BUFR di MeteoHub se offre più stazioni/parametri del
   JSON REST.
3. Provider regionali (ARPA, centri funzionali) e METAR/SYNOP aggiuntivi:
   l'interfaccia `ObservationProvider` è pronta per riceverli; vanno aggiunti
   solo dove la copertura risultante da ItaliaMeteo+MeteoNetwork lascia un
   vuoto reale (misurabile con `coverage.py`).
4. Buddy-check spaziale e cross-provider completo: richiedono uno storico di
   più run per essere statisticamente solidi; il registro conserva già tutte
   le informazioni (coordinate, quota, duplicate candidates) necessarie.
5. Storage dedicato (PostGIS/TimescaleDB): l'architettura attuale è
   file-JSON+GitHub Actions; introdurre un database esterno richiede una
   decisione infrastrutturale (hosting, costi) che esula da questo ambiente
   sandbox. Il payload prodotto da `pipeline.py` è già nella forma tabellare
   (`stations`/`observations` per variabile) più adatta a una futura
   migrazione, senza dover riscrivere i provider.
6. Matching ICON↔osservazioni multi-provider e Verification Engine
   (BIAS/MAE/RMSE per lead time/regione/quota): oggi `StationForecastArchive`
   e `verify_station_forecasts` operano solo sulle stazioni METAR (le uniche
   con un ID stabile usato anche dal campionamento ICON). Estendere il
   campionamento ICON a tutte le stazioni del registro è il prossimo passo
   naturale una volta che ItaliaMeteo/MeteoNetwork saranno popolati da
   credenziali reali.

## 6. Metriche diagnostiche disponibili

Ogni payload pubblicato (`data_weather/observations.json`) contiene ora
`diagnostics`:

```
stationsTotal, stationsReporting15m/30m/60m/3h, stationsStale,
stationsOffline, byHealth, bySource, byRegion, duplicatesFlagged,
providerStatus
```

e `coverage.overall` / `coverage.byVariable.<temperature|precipitation|
windSpeed|relativeHumidity|pressureMsl>` con distanza media/massima dalla
stazione più vicina e frazione di celle senza copertura entro 20 km.

## 7. Test

`scripts/tests/test_observation_network.py` copre, con fixture locali
(nessuna chiamata di rete): isolamento dei provider, deduplicazione con
confidence score, controllo qualità (bounds fisici, sensore bloccato),
classificazione dello station health per-provider, diagnostica di rete,
copertura per parametro, e la disattivazione sicura dei provider senza
credenziali. Eseguito in CI insieme a `test_verification.py`.
