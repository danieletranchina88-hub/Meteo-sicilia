# Archivio previsione–osservazione per l'IA

## Perché viene prima del modello statistico

Una correzione automatica è scientificamente utile soltanto se migliora ICON-2I
su casi indipendenti. Il sistema conserva quindi, per ogni run e valid time, la
previsione realmente disponibile in quel momento e la collega in seguito a
osservazioni con timestamp proprio. Non si riusa un'analisi futura come
predittore e non si modifica retroattivamente la previsione archiviata.

Il primo dataset è destinato a verifiche e correzioni residue di temperatura,
umidità, vento e pressione. Le precipitazioni, la convezione e i fronti
richiederanno sorgenti osservative dedicate (radar, MTG-LI e analisi frontali
indipendenti) e non vengono etichettati artificialmente usando ICON stesso.

## Contenuti di ogni run

`data_weather/archive_manifest.json` dichiara:

- modello, run, dominio, risoluzioni nominali e scadenze;
- completezza oraria da +0 a +72 h;
- versione di ciascun algoritmo;
- URL, dimensione e SHA-256 di ogni GRIB accettato dalla pipeline;
- dimensione, ruolo e SHA-256 di ogni prodotto pubblicato;
- quali dati sono davvero conservati e quali soltanto identificati.

`data_weather/verification/forecast_samples.json.gz` contiene i valori ICON-2I
interpolati nelle coordinate delle stazioni METAR per tutte le 73 scadenze. Il
campionamento è bilineare sulla griglia superficiale nativa, non sulla griglia
diradata dei meteogrammi. Non vengono effettuate estrapolazioni né riempimenti
climatologici; una variabile mancante resta `null`.

Ogni ora la workflow `Collect Verification Observations` acquisisce un solo
snapshot METAR delimitato al dominio, rispettando la normale cadenza oraria
delle osservazioni. Ogni stazione conserva il proprio `obsTime`: l'ora più
recente del payload non viene assegnata alle altre stazioni. La documentazione
operativa della sorgente è disponibile presso
[NOAA Aviation Weather Center](https://aviationweather.gov/data/api/).

## Semantica della pressione

Nel JSON AWC sono conservate separatamente:

- `seaLevelPressureHpa`: vera codifica METAR `SLP`, confrontabile con la MSLP
  modellistica;
- `altimeterHpa`: altimeter setting/QNH, utile operativamente ma non trattato
  come se fosse la stessa grandezza della MSLP.

Il campo compatibile con l'interfaccia `pressHpa` replica esclusivamente
`seaLevelPressureHpa`. Se SLP manca, la pressione osservata resta mancante.
Questo riduce la copertura, ma impedisce che l'algoritmo apprenda un bias creato
dalla mescolanza di due variabili differenti.

## Verifica

`meteo_analysis.verification.metrics.verify_station_forecasts` associa una
previsione all'osservazione della stessa stazione più vicina nel tempo entro
45 minuti. L'errore è sempre:

\[
e = previsione - osservazione
\]

Per ogni variabile e fascia `0–6`, `7–24`, `25–48`, `49–72 h` vengono calcolati:

- numero di coppie valide;
- bias medio;
- MAE;
- RMSE.

Il vento è confrontato nelle componenti cartesiane e nella velocità, evitando
la discontinuità 359°/0°. L'umidità osservata viene derivata da temperatura e
punto di rugiada con la stessa formulazione dichiarata. Le metriche escludono
ogni coppia incompleta; uno zero non sostituisce mai un dato mancante.

Le stazioni sono misure puntuali mentre ICON rappresenta una cella di circa
2,2 km. I punteggi includono quindi anche differenze di quota, esposizione e
uso del suolo non risolte. Questi fattori verranno forniti al modello di
correzione come caratteristiche esplicite, non nascosti nel target.

Il comando riproducibile è:

```bash
python scripts/verify_archive.py forecast_samples.json.gz osservazioni/ \
  --output scorecard.json
```

## Rete nazionale delle stazioni

Il file `data_weather/observations.json` contiene due insiemi volutamente
separati, entrambi provenienti dal NOAA Aviation Weather Center:

- `stationNetwork`: il catalogo delle stazioni METAR italiane registrate
  dall'AWC, incluse stazioni costiere, montane e aeroportuali;
- `stations`: il solo sottoinsieme che ha un METAR ricevuto dal servizio al
  momento dell'acquisizione.

La mappa mostra la rete completa soltanto su richiesta dell'utente. Verde
significa report recente, ambra report ritardato e grigio assenza di report
corrente. La stazione non osservata resta visibile: non si sostituisce con un
valore ICON, non si interpola e non si presenta un dato vecchio come corrente.
Le osservazioni del sottoinsieme `stations` restano le uniche ammesse nella
fusione con il modello, e solo entro la finestra temporale dichiarata.

La rete METAR non equivale a una rete pluviometrica o ARPA ad alta densità. È
una rete nazionale coerente e con metadati verificabili, ma le correzioni locali
di precipitazione, suolo e microclima richiederanno in seguito fonti regionali
aperte con licenze e controlli di qualità espliciti.

## Conservazione a oggetti dei GRIB

La pipeline può ora archiviare tutti i GRIB ICON-2I realmente accettati —
superficie, livelli barici, campi frontali e diagnostiche convettive — in uno
storage privato compatibile S3. L'upload avviene immediatamente dopo ogni
download, prima della lettura e della rimozione dei file temporanei.

Ogni oggetto usa una chiave contenente run, ruolo e SHA-256 del contenuto. Se
la chiave esiste, dimensione e SHA-256 devono coincidere; in caso contrario la
workflow si interrompe senza sovrascrivere lo storico. Il manifest pubblico
espone soltanto modalità, chiave e checksum, mai endpoint, bucket o credenziali.

Per attivarlo in GitHub Actions servono questi *Repository secrets*:

| Secret | Significato |
| --- | --- |
| `ICON_ARCHIVE_S3_BUCKET` | nome del bucket privato |
| `ICON_ARCHIVE_S3_ENDPOINT` | endpoint S3 del provider |
| `ICON_ARCHIVE_S3_ACCESS_KEY_ID` | chiave privata per `PutObject` e `GetObject`/HEAD; se il provider separa le azioni multipart, anche quelle di upload multipart sul solo prefisso dedicato |
| `ICON_ARCHIVE_S3_SECRET_ACCESS_KEY` | segreto della chiave |
| `ICON_ARCHIVE_S3_REGION` | opzionale; per R2 normalmente `auto` |
| `ICON_ARCHIVE_S3_SESSION_TOKEN` | opzionale, solo credenziali temporanee |
| `ICON_ARCHIVE_S3_PREFIX` | opzionale; predefinito `runs/icon2i` |

La presenza del bucket attiva `ICON_ARCHIVE_MODE=raw`; se una credenziale o un
upload non funziona, il run fallisce invece di dichiarare falsamente i GRIB
come conservati. Lo stesso contratto funziona con AWS S3, Cloudflare R2 e
Backblaze B2. Il bucket deve restare privato, con versioning e una lifecycle
policy decisa in base a costo e periodo storico desiderato.

## Conservazione e stato operativo

La pipeline crea per ogni run un artifact `icon2i-verification-*`, con manifest,
campioni e controlli qualità, e conserva gli snapshot osservativi separati per
90 giorni. È sufficiente per avviare scorecard e primo MOS residuo superficiale.

Finché i secret S3 non sono configurati, l'archivio resta in modalità `off`: i
manifesti dichiarano `retainedInArchive: false` e nessun URL viene presentato
come se equivalesse al possesso del GRIB. Con storage attivo, il manifest
dichiara `rawGribAssetsRetained` e abilita onestamente la preparazione di
dataset storici a campo completo.

## Passo ML successivo

Il primo modello sarà una correzione residua controllata:

\[
T_{corretta} = T_{ICON} + \Delta T_{ML}
\]

con suddivisione temporale rigorosa train/validation/test, confronto contro la
baseline ICON non corretta, metriche per anticipo e stazione, controllo dei casi
fuori distribuzione e fallback automatico a ICON. Nessun modello verrà promosso
se il vantaggio non è stabile su dati mai usati nell'addestramento.
