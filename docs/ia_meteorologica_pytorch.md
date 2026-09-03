# IA meteorologica PyTorch: fronti e downscaling

## Stato operativo

Questi moduli costituiscono una pipeline di ricerca addestrabile e verificata
numericamente. Non contengono pesi preaddestrati e non modificano ancora i
prodotti pubblici. Un checkpoint privo di criteri di accettazione resta
esplicitamente `candidate-only`; il loader rifiuta per impostazione predefinita
un modello non promosso.

Questa separazione è necessaria: un'architettura corretta non è ancora una
previsione migliore. Servono target indipendenti, una separazione temporale
rigorosa e confronto contro baseline fisiche.

## 1. Segmentazione frontale

### Verità a terra

La tassonomia operativa è multiclasse:

0. fondo;
1. fronte freddo;
2. fronte caldo;
3. fronte occluso;
4. fronte stazionario.

Il preparatore usa le polilinee manuali DWD (CC BY 4.0, DOI
`10.5281/zenodo.5785816`) e predittori ERA5. L'archivio europeo supervisiona
solo fondo, freddo, caldo e occluso. Le carte DWD rappresentano infatti i
fronti stazionari come tratti alternati caldo/freddo e l'estrazione pubblicata
non permette di separarli in modo autorevole; il manifest e il checkpoint
dichiarano quindi `stationary` tra le classi non supportate. Non vengono creati
esempi fittizi né pesi di classe artificiali. La limitazione è documentata
nell'articolo metodologico DOI `10.5194/wcd-3-113-2022`.

La classe stazionaria rimane affidata alla diagnosi di moto, alla fisica e al
tracking temporale v19. Un futuro modello a cinque classi richiederà etichette
NWS/NOAA, che includono esplicitamente gli stazionari, seguite da verifica del
trasferimento di dominio sull'Europa. Le linee DWD vengono rasterizzate con
distanza geodetica; il bordo artificiale del buffer riceve peso minore. Tutte
le celle e patch della stessa data restano nello stesso split cronologico
70/15/15. Non è ammesso uno split casuale per pixel.

I target prodotti dall'algoritmo frontale fisico v19 possono essere usati per
pretraining debole, ma devono essere marcati come `pseudo-label`: non sono una
verità indipendente e non possono essere usati per dimostrare che la rete
migliori il v19.

### Input e rete

La U-Net residuale usa 40 canali fisici, senza latitudine/longitudine:

- theta e theta-e a 850 hPa;
- gradienti termici a 20, 50 e 100 km;
- avvezione termica e frontogenesi alle stesse scale;
- gradiente di rugiada, umidità relativa e mixing ratio a 850/700 hPa;
- convergenza a 10 m, divergenza e vorticità a 850 hPa;
- tendenza PMSL a 3 h;
- avvezione di vorticità e geopotenziale a 500 hPa;
- persistenza temporale del gradiente termico;
- theta-w e theta-e nella struttura 925–850–700 hPa;
- componenti `u/v` a 925, 850 e 700 hPa, vento a 10 m e omega a 700 hPa;
- PMSL assoluta, oltre alla sua tendenza.

Ogni canale è normalizzato con statistiche calcolate esclusivamente sul
training set. Le celle mancanti diventano zero dopo la standardizzazione, ma
la rete riceve anche una maschera di validità per ogni canale: un dato assente
non viene confuso con un valore medio reale. GroupNorm evita la fragilità di
BatchNorm con batch piccoli.

L'output comprende logits multiclasse e una testa gerarchica binaria
`frontness`. La loss combina cross-entropy pesata, Dice sulle sole classi
frontali, consistenza della testa binaria e una penalità debole facoltativa per
contraddizioni con il supporto fisico. La fisica non può sovrascrivere
un'etichetta manuale né trasformare un pseudo-target in osservazione.

Dopo l'early stopping, un unico parametro di temperature scaling viene
stimato soltanto sul validation set. Brier score ed Expected Calibration Error
sono poi calcolati sul test mai visto; la testa binaria resta ausiliaria e la
confidenza pubblicabile deriva dalla distribuzione multiclasse calibrata.

L'inferenza sull'intero dominio usa tile sovrapposte e finestra di Hann: niente
giunzioni visibili ai bordi delle patch. Le probabilità raster non devono
essere pubblicate direttamente come polilinee. Vanno fuse con il grafo v19,
che conserva posizione sulla cresta, persistenza temporale, punto triplo,
occlusione e controlli contro brezza, cold pool e orografia.

La rete riconosce il contesto sinottico sulla griglia di addestramento a
0,20°. Non si interpola ERA5 a 2,2 km fingendo di aver creato informazione. Il
tracciato finale viene invece rifinito sui campi ICON-2I nativi dal motore
fisico v19: classificazione appresa a grande scala, geometria alla massima
risoluzione realmente disponibile.

### Addestramento

```bash
# CPU; per CUDA installare prima la build PyTorch adatta al proprio driver.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ml-training.txt

python scripts/prepare_front_unet_data.py \
  --start 2015-01-01 --end 2019-12-31 \
  --output training_data/front_unet

python scripts/validate_deep_learning_data.py \
  training_data/front_unet/manifest.json

python scripts/train_front_unet.py \
  training_data/front_unet/manifest.json \
  --config configs/front_unet.json \
  --output models/candidates/front_unet.pt
```

Il workflow `Train Front U-Net Pilot` esegue automaticamente una prova reale
CPU su un massimo di 120 analisi distribuite nel 2015–2019. Serve a verificare
download,
costruzione dei 40 canali, training, calibrazione e checkpoint in un ambiente
riproducibile. Il risultato rimane `candidate-only` e l'artefatto GitHub scade
dopo 30 giorni: non sostituisce il training completo su GPU né uno storage
scientifico permanente.

La promozione richiede un file di criteri indipendenti passato con
`--acceptance-config`. Oltre a IoU e Dice per classe, la revisione operativa
deve misurare distanza media/95° percentile dalla linea manuale, continuità,
numero di componenti spurie, errori di tipo, stabilità fra ore e prestazioni
per stagione e forecast hour.

Il file di accettazione frontale deve dichiarare
`minimumMeanFrontalIoU`, `minimumMeanFrontalDice`, `maximumTestLoss`,
`maximumFrontBrier` e `maximumFrontCalibrationError`. I valori non sono
preimpostati: vanno fissati prima dell'esperimento rispetto alle prestazioni
del v19/XGBoost sul medesimo test, per evitare di scegliere la soglia dopo
aver visto il risultato.

## 2. Downscaling orografico

### Cosa può e non può imparare

ICON-2I e il DEM sono input, non target. Senza un campo indipendente a
risoluzione più alta, una SRGAN può soltanto imparare una texture plausibile.
Per l'Italia servono coppie temporalmente allineate con:

- temperatura: stazioni sottoposte a QC e, idealmente, analisi grigliata ad
  alta risoluzione costruita senza usare la previsione da correggere;
- precipitazione: composito radar quantitativo corretto con pluviometri, con
  maschere di qualità e copertura;
- vento: rete osservativa densa o analisi ad alta risoluzione, con esposizione
  e quota della stazione; `u` e `v` devono essere target separati.

Una stazione puntuale non va interpolata e dichiarata “verità 500 m” senza
incertezza. È preferibile una loss sparsa alle stazioni, oppure un'analisi
grigliata con relativa maschera di qualità.

### Architettura

Il modello predefinito porta 2,2 km a circa 550 m (`scale=4`) e usa due stadi:

1. correzione residuale alla scala ICON, supervisionata dalla media del target
   ad alta risoluzione;
2. redistribuzione sub-grid condizionata da elevazione, pendenza zonale e
   meridionale in m/m, frazione di terra e distanza geodetica dalla costa.
   L'area sferica di ogni cella entra soltanto nel vincolo numerico e non è
   mostrata alla rete come scorciatoia geografica.

I quattro output sono temperatura 2 m, rateo di precipitazione, vento `u10` e
vento `v10`. La precipitazione è non negativa. Per ogni cella ICON la media
dei 16 pixel figli coincide, entro precisione numerica, con il campo coarse
corretto. La correzione del bias può quindi cambiare la scala sinottica; la
super-risoluzione successiva può soltanto redistribuirla. Il vincolo vale per
quantità intensive (`K`, `m/s`, `mm/h`) e usa pesi areali sulla griglia
geografica. Per temperatura e vento è un vincolo di consistenza con la scala
coarse; per il rateo di precipitazione è anche conservazione areale. Eventuali
accumuli espressi come
totale per cella richiedono invece un vincolo area-pesato esplicito.

L'intero dominio italiano non viene inviato alla GPU in un unico blocco: la
funzione `tiled_downscaling` usa tile coarse sovrapposte, fonde con una finestra
di Hann costante dentro ogni cella figlia e riapplica infine il vincolo coarse.
Questo limita la memoria senza introdurre cuciture o perdere conservazione.

La loss usa Huber in unità fisiche per temperatura e componenti del vento,
Huber su `log1p` per la precipitazione, errore dei gradienti e supervisione
della correzione coarse. Non è presente una loss GAN predefinita: texture più
nitide non equivalgono a maggiore accuratezza meteorologica e possono
inventare estremi.

### Contratto dei dati

Il manifest JSON ha `task: "orographic-downscaling"` e tre split temporali.
Ogni record punta a un NPZ numerico, caricato con `allow_pickle=False`, con:

- `coarse`: `[C, H, W]` in unità fisiche;
- `static`: `[5, 4H, 4W]` nell'ordine descritto sopra, oppure un riferimento
  `sharedStatic` e una `fineWindow` esplicita nel record;
- `cell_area_m2`: `[4H, 4W]`, locale oppure nello stesso file statico
  condiviso, usata per la riaggregazione area-pesata;
- `target`: `[4, 4H, 4W]`;
- `valid_mask`: `[4, 4H, 4W]`, facoltativa.

Il manifest dichiara inoltre `coarseChannels`, `staticChannels`,
`outputChannels`, provenienza e versione dei target, QC, griglia/CRS e policy
di split. I primi quattro indici coarse usati dagli output sono configurabili.
La funzione `build_orographic_sample` ricava i cinque campi statici e l'area
di cella da DEM,
maschera terra-mare e coordinate regolari, senza fingere un allineamento tra
griglie sorgenti diverse: radar, DEM e ICON devono essere riproiettati a monte
con CRS, timestamp e metodo di interpolazione documentati.

`prepare_downscaling_data.py` riceve un manifest di sorgenti già allineate,
calcola una sola volta DEM derivato, distanza dalla costa e aree geodetiche,
crea patch sovrapposte e assegna tutti i campioni dello stesso `validTime` allo
stesso split. Non effettua deliberatamente una riproiezione implicita di radar
o stazioni. Gli shard e lo statico condiviso sono accompagnati da SHA-256 e
possono essere verificati prima del training:

```bash
python scripts/prepare_downscaling_data.py aligned_sources.json \
  --output training_data/downscale \
  --patch-size-coarse 64 --stride-coarse 48

python scripts/validate_deep_learning_data.py \
  training_data/downscale/manifest.json
```

```bash
python scripts/validate_deep_learning_data.py training_data/downscale/manifest.json

python scripts/train_downscaler.py \
  training_data/downscale/manifest.json \
  --config configs/orographic_downscaler.json \
  --output models/candidates/orographic_downscaler.pt
```

### Verifica necessaria

PSNR e SSIM non bastano. Il confronto deve includere:

- temperatura: MAE, bias, RMSE e verifica per fascia altimetrica/ora;
- vento: errore vettoriale, velocità, direzione solo sopra una soglia minima,
  valli, crinali e coste;
- precipitazione: bias, MAE su `log1p`, CSI/FSS a più soglie e scale,
  quantili estremi, spettro spaziale e affidabilità degli zeri;
- tutti i campi: errore di riaggregazione, stabilità temporale e confronto con
  interpolazione bilineare/conservativa e correzioni statistiche semplici.

Il test finale deve usare mesi o eventi mai visti e stazioni/radar esclusi
dalla costruzione dei target quando possibile. Il miglioramento deve essere
misurato contro ICON-2I nativo e contro una baseline interpolata, non soltanto
contro la rete precedente.

Lo script registra già MAE per canale, bias termico, errore vettoriale del
vento e CSI della precipitazione a 0,1/1/5/10 mm/h, insieme alle stesse MAE
della baseline a blocchi e allo skill relativo. La promozione richiede skill
positivo configurabile su ogni canale e vento vettoriale migliore della
baseline. FSS multiscala, quantili e spettro restano verifiche obbligatorie
nel rapporto scientifico prima del deploy, perché dipendono dalla risoluzione
e dalla maschera di qualità specifiche del dataset scelto.

Il file di accettazione richiede `maximumMaeByChannel`,
`minimumMaeSkillByChannel` e `maximumCoarseConsistencyError`. Anche questi
limiti devono essere preregistrati sul validation set e non adattati al test.

## Decisione operativa

I pesi entrano nel sito solo dopo quattro condizioni:

1. target e licenze documentati;
2. test temporale e spaziale indipendente superato;
3. beneficio rispetto alle baseline in più stagioni e forecast hour;
4. monitoraggio continuo di drift, bias e dati mancanti con rollback al
   prodotto fisico.

Fino a quel momento il v19 e i campi ICON-2I restano il prodotto autorevole;
il deep learning è un candidato sperimentale, non una correzione invisibile.

## Riferimenti metodologici

- Ronneberger, Fischer e Brox (2015), *U-Net: Convolutional Networks for
  Biomedical Image Segmentation*, DOI `10.1007/978-3-319-24574-4_28`.
- Biard e Kunkel (2019), *Automated detection of weather fronts using a deep
  learning neural network*, DOI `10.5194/ascmo-5-147-2019`.
- Vandal et al. (2017), *DeepSD: Generating High Resolution Climate Change
  Projections through Single Image Super-Resolution*, DOI
  `10.1145/3097983.3098004`.
- Harder et al. (2023), *Hard-Constrained Deep Learning for Climate
  Downscaling*, JMLR 24(365), pp. 1–40.
