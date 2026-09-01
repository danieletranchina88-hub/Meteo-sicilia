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

## Importazione degli archivi storici ufficiali ICON-2I

Non è necessario attendere che l'archivio operativo accumuli da solo tutti i
casi iniziali. MeteoHub cataloga cinque serie storiche ItaliaMeteo-Arpae-CINECA,
tutte con licenza CC BY 4.0:

| Dataset | Copertura | Dominio e contenuto | Ruolo corretto |
| --- | --- | --- | --- |
| `ICON_2I_ita2km` | 29/09/2024–26/05/2025 | Italia, superficie e livelli modello | addestramento deterministico principale |
| `ICON_2I_all2km` | 29/09/2024–26/05/2025 | dominio completo, superficie e livelli isobarici | contesto sinottico |
| `ICON_2I_ASSIM_ita2km` | 04/02/2025–26/05/2025 | Italia, background di assimilazione | stato modellistico iniziale, non verità osservativa |
| `ICON_2I_ASSIM_all2km` | 04/02/2025–26/05/2025 | dominio completo, background di assimilazione | diagnostica sinottica, non target osservativo |
| `ICON_2I_FCENS` | 18/06/2024–26/05/2025 | dominio completo, ensemble | incertezza e calibrazione probabilistica |

Il catalogo ufficiale e le attribuzioni sono pubblicati nella pagina
[MeteoHub — licenze](https://meteohub.agenziaitaliameteo.it/app/license). I
dataset marcati `NOT OPERATIONAL` sono serie storiche finite: non sostituiscono
la raccolta dei nuovi run.

Dal 17 giugno 2026 la catena ICON-2I usa inoltre una nuova configurazione del
ciclo di analisi del suolo. Ogni esempio storico importato viene pertanto
marcato come
`legacy-before-2026-06-17-soil-analysis-change`: mescolare silenziosamente
questi casi con i run correnti introdurrebbe un cambio di distribuzione, in
particolare per temperatura e umidità prossime al suolo.

### Importatore riprendibile

`scripts/import_icon2i_history.py` implementa il flusso API documentato da
MeteoHub:

1. crea una richiesta distinta per ciascun giorno/run;
2. conserva nel manifest il corpo esatto della richiesta e la sua impronta;
3. non reinvia richieste già registrate;
4. controlla lo stato asincrono del servizio;
5. scarica in streaming in un file `.part`;
6. verifica dimensione, formato contenitore e SHA-256;
7. archivia l'estratto nello storage S3 immutabile;
8. elimina il file temporaneo soltanto dopo la conferma dello storage.

Per vedere il catalogo incorporato e generare un piano senza effettuare
download:

```bash
python scripts/import_icon2i_history.py catalog
python scripts/import_icon2i_history.py plan \
  --dataset ICON_2I_ita2km \
  --start 2024-09-29 --end 2024-10-05
```

MeteoHub applica normalmente un limite di 1 GB per estrazione. Per questo
l'importatore rifiuta l'invio non filtrato. Dall'interfaccia MeteoHub si crea
una prima richiesta selezionando variabili, livelli e scadenze, quindi si usa
`Copy to clipboard` e si conserva quel JSON nel secret
`METEOHUB_REQUEST_TEMPLATE_JSON`. Per acquisire tutti i campi senza superare
la quota, il secret può contenere una lista JSON di richieste distinte, per
esempio superficie, dinamica/fronti, livelli modello e convezione. Il nome
della richiesta diventa l'identificatore del gruppo. L'importatore sostituisce
sempre dataset, data e nome operativo e rimuove qualunque pianificazione
contenuta nel template; i filtri scientifici restano identici per tutti i
giorni. Qualsiasi campo simile a password, token o autorizzazione viene
rifiutato prima della creazione del manifest.

La suddivisione prevista, senza riduzione della griglia orizzontale, è:

| Gruppo | Campi da includere |
| --- | --- |
| superficie | MSLP, T/Td/RH 2 m, U/V 10 m, precipitazioni, neve, nubi, visibilità e raffiche |
| fronti e bassa troposfera | T, QV/RH, U/V e geopotenziale a 925, 850 e 700 hPa |
| dinamica verticale | T, U/V, geopotenziale, omega e vorticità ai livelli isobarici disponibili |
| livelli modello | T, QV, U/V, pressione, geopotenziale e idrometeore sui livelli nativi disponibili |
| convezione e severità | CAPE/CIN, LPI, updraft helicity, shear, graupel, acqua integrata e raffica massima disponibili |

I nomi effettivi dei prodotti devono provenire dai filtri mostrati dal dataset
MeteoHub: non vengono inventati campi che l'archivio storico non contiene. La
separazione serve solo a rispettare le quote di estrazione; tutti i file
mantengono la risoluzione nativa e vengono ricongiunti mediante dataset, run,
gruppo e checksum.

L'autenticazione accetta `METEOHUB_TOKEN` oppure la coppia
`METEOHUB_USERNAME`/`METEOHUB_PASSWORD`. La workflow manuale
`Import Historical ICON-2I` ha due modalità:

- `plan`: convalida intervallo e manifest senza credenziali;
- `ingest`: invia al massimo il numero dichiarato di richieste, attende gli
  output, ne verifica i checksum e li conserva nello storage configurato.

Una singola esecuzione non invia più di dieci richieste e il valore consigliato
è cinque, coerentemente con il limite orario standard del servizio. Il manifest
della workflow rimane un artifact di provenienza; i file meteorologici non
vengono inseriti in `main`, `gh-pages` o nella cronologia Git.
