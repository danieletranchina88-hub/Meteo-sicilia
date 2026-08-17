# Fronti supervisionati e fusione fisica

## Cosa impara il modello

Il classificatore XGBoost apprende la probabilità che un punto sia entro
40 km da una linea frontale tracciata manualmente dal DWD. La verità terreno
è l'archivio CC BY 4.0 di S. Niebler, *Front polylines extracted from DWD
Maps*, DOI `10.5281/zenodo.5785816`. Le linee native sono densificate,
misurate sulla sfera e grigliate a 0,20° sull'intero dominio pubblico
ICON-2I/MeteoHub: 3–22°E, 33,7–48,9°N.

Il modello è binario: riconosce la presenza di una struttura frontale, non
decide se sia fredda, calda, stazionaria o occlusa. Il tipo resta una diagnosi
fisica basata su moto della linea, OFA/Hewson, avvezione e continuità della
traccia. L'occlusione continua a richiedere minimo barico, punto triplo e
avvolgimento geometrico.

## Predittori e formule

Le feature sono calcolate dallo stesso modulo sia su ERA5 storico sia su
ICON-2I operativo:

- temperatura potenziale:
  `theta = T (p0/p)^0.2854`;
- temperatura potenziale equivalente con LCL e formula di Bolton;
- gradienti metrici di theta a 850 hPa dopo smooth gaussiani di 20, 50 e
  100 km;
- avvezione termica: `-V · grad(theta)`, espressa in K/3 h;
- frontogenesi bidimensionale di Petterssen, dovuta a convergenza e
  deformazione del vento;
- gradienti di rugiada, UR e mixing ratio a 850/700 hPa;
- convergenza a 10 m, divergenza e vorticità a 850 hPa;
- tendenza PMSL in 3 h, avvezione di vorticità e geopotenziale a 500 hPa;
- media e deviazione su tre ore del gradiente di theta a 50 km;
- latitudine e longitudine sono calcolate per gli esperimenti, ma escluse
  dall'artefatto operativo corrente per impedire scorciatoie climatologiche.

Le tre feature non trasferibili in modo omogeneo dal dataset ERA5 pubblico
(shear 0–6 km, quota e ruggedness) sono escluse dall'artefatto operativo.
L'ordine delle colonne è salvato nei metadati e protetto da SHA-256.

## Addestramento riproducibile

```bash
python -m pip install -r requirements-ml-training.txt
python scripts/prepare_front_training_data.py \
  --start 2015-01-01 --end 2019-12-31 \
  --output training_data/front_features.parquet
python scripts/train_front_model.py \
  training_data/front_features.parquet \
  --model models/front_model.json

Per l'inferenza automatica il repository conserva lo stesso modello in
`models/front_model.json.gz.b64`: è una copia compressa e codificata Base64,
caricata direttamente da `FrontModel` senza alterare i valori predetti.
```

Il preparatore scarica l'archivio DWD verificandone l'MD5 e legge, senza
scaricare file globali, l'ERA5 pubblico Icechunk di Earthmover. I periodi sono
ordinati e separati cronologicamente 70/15/15: nessuna data della stessa
situazione può finire in train e test. `scale_pos_weight` gestisce la rarità
dei fronti; early stopping usa la precision-recall AUC. Una regressione
logistica di Platt calibra le probabilità dopo il riequilibrio.

La distanza geodetica dalla linea manuale viene conservata per ogni cella.
Le celle vicine al bordo artificiale dei 40 km ricevono un peso ridotto
(minimo 0,35), mentre nucleo frontale e fondo lontano mantengono peso pieno:
un errore di pochi chilometri dell'analista non viene trattato come una
contraddizione netta. L'accettazione fuori campione controlla anche recall
entro 20 km e falsi positivi oltre 80 km, oltre alle metriche di cella.

L'artefatto corrente è un modello iniziale addestrato su 89 analisi
sinottiche distribuite tra 2015 e 2019 (circa 658 mila celle). Le metriche
fuori campione sono registrate, senza arrotondamenti, in
`models/front_model.metadata.json`. Il profilo operativo senza coordinate è
stato scelto perché la prima ablation con lat/lon attribuiva troppo peso alla
posizione e trasferiva male le linee sul run ICON reale. Per una versione
definitiva è consigliato
rigenerare il dataset senza `--stride` (1.777 analisi 00Z) e accettare il nuovo
modello solo se migliora soprattutto precisione, recall, F1, IoU e Brier sul
test temporale.

## Inferenza ICON-2I

`scripts/ml_fronts.py` legge i GRIB run-wide T/Q/U/V a 850 e 700 hPa,
U/V/FI a 500 hPa, U/V a 10 m e PMSL. La griglia ridotta limita memoria e
tempo del runner. Il workflow calcola il ML ogni tre ore; le ore intermedie
sono interpolate esclusivamente per fornire una conferma al candidato fisico.
Le isolinee ML diagnostiche richiedono presenza in due delle tre scadenze
adiacenti e sono salvate in `data_weather/fronts_ml/`.

Valori mancanti restano `NaN`: XGBoost li gestisce esplicitamente. Non vengono
riempiti con climatologia o zeri plausibili. Temperature, umidità specifica e
pressione sono controllate nelle unità prima dell'inferenza. La riduzione a
0,20° e gli smooth multiscala limitano rumore convettivo e orografico.

## Regole della fusione

La fusione è intenzionalmente asimmetrica:

1. un candidato che supera la fisica viene mantenuto anche se il ML è basso;
2. una conferma ML può aggiungere al massimo 0,05 all'evidenza, quindi non
   sostituisce termodinamica, dinamica o persistenza;
3. il ML non assegna mai il tipo frontale;
4. cold pool, brezza, limite orografico, divergenza forte, diagnosi non
   frontale o assenza di supporto sinottico non sono recuperabili dal ML;
5. è recuperabile soltanto un *near-pass* termico: ogni ingrediente deve
   raggiungere almeno l'80% della propria soglia, il supporto sinottico deve
   già superare il gate, l'evidenza fisica deve essere almeno 0,40 e la
   probabilità ML mediana almeno 0,65 con il 55% della linea sopra soglia;
6. anche un near-pass resta `ml-assisted-continuation`: non è una rilevazione
   forte e può essere pubblicato soltanto dentro una traccia con un'ancora
   fisica forte, almeno tre ore di vita e classificazione coerente.

Ogni GeoJSON pubblicato espone probabilità mediana ML, frazione di supporto,
evidenza fisica originale, bonus e decisione di fusione. In questo modo un
fronte non è mai una scatola nera e ogni correzione può essere verificata.

## Limiti metodologici

Le analisi manuali hanno incertezza di posizione e stile del previsore; il
buffer di 40 km non è una probabilità fisica. ERA5 (0,25°) e ICON-2I (~2 km)
presentano un *domain shift*: per questo il modello non pubblica linee da
solo. Le coordinate possono apprendere una climatologia geografica: sono
disponibili nel dataset, ma il modello operativo corrente non le usa.
Nuvole e precipitazione non sono feature di esistenza del fronte: sono
effetti utili alla previsione, ma potrebbero far apprendere scorciatoie e
confondere fronti secchi o mascherati.
