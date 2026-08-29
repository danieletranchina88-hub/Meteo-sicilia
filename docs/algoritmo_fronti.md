# Analisi oggettiva dei fronti ICON-2I (v19-auditable)

## Scopo e limite fondamentale

Il sito pubblica una **stima oggettiva e conservativa** dei fronti sinottici
nel dominio ICON-2I. La geometria proviene esclusivamente da ICON-2I; ECMWF
resta un modello di superficie separato e non conferma, modifica o sostituisce
i fronti.

La linea non è un'osservazione né un'analisi manuale del Servizio
Meteorologico. `qualityScore` e `uncertaintyIndex` descrivono la coerenza
interna delle prove in un singolo run deterministico: non sono probabilità
calibrate e non misurano l'errore previsionale assoluto.

Il metodo operativo è `icon2i-ofa-physics-guided-v19-auditable`.

## Dati usati

Per ogni run 00/12 UTC e per tutte le 73 scadenze orarie:

- T, QV, U e V a 850 hPa: campi obbligatori e geometria primaria;
- PMSL: firma di saccatura e tendenza barica, opzionale;
- T, QV, U e V a 925 hPa: coerenza verticale e vento più vicino al suolo,
  opzionali;
- T, QV, U e V a 700 hPa: struttura verticale, inclinazione e vento in quota,
  opzionali;
- U e V e geopotenziale a 500 hPa: steering 700–500 hPa e contesto dell'onda
  sinottica, opzionali e mai usati come prova autonoma d'esistenza;
- U e V a 10 m su ore consecutive: controllo temporale del salto di vento,
  opzionali;
- OMEGA a 700 hPa: attività/ascesa frontale, opzionale;
- PS: maschera esatta delle superfici isobariche sotto il terreno, opzionale;
- HSURF: controllo orografico.

L'analisi frontale usa ora la griglia ICON-2I ricampionata a circa **4,4 km**
(`downsample=2`), anziché circa 8,8 km. Le scale fisiche di smoothing restano
45–100 km: la maggiore densità non crea dettagli mesoscalari artificiali, ma
descrive meglio curve, estremi e giunzioni della stessa struttura sinottica.

## Livelli disponibili nella mappa

La tendina della mappa ICON-2I consente di vedere **temperatura, vento e
umidità relativa** su tre quote reali: suolo (T 2 m, vento 10 m e RH
superficiale), 925 hPa e 850 hPa. Per 925/850 hPa il sito usa direttamente
T, QV, U e V dei file MeteoHub; l'umidità relativa è ricavata da QV, pressione
del livello e pressione di saturazione di Bolton. I livelli in quota vengono
esportati su una griglia di ispezione alleggerita (~19 km), sufficiente per la
lettura sinottica e molto più leggera sul cellulare. Se un run non contiene
un livello opzionale, l'interfaccia non lo sostituisce né lo interpola: torna
al suolo e avvisa l'utente. A 925 hPa il motore frontale confronta **PS** con
925 hPa e maschera ogni cella in cui la superficie isobarica cade nel terreno;
solo se PS manca usa il precedente ripiego orografico a 650 m.

Il livello 850 hPa riduce il rumore dello strato limite e i contrasti diurni.
Il livello 925 hPa interseca il terreno già attorno a 750 m. Dove almeno il
60% del confronto resta valido, la
coerenza termica a 925 hPa diventa una porta obbligatoria; altrove non viene
inventato un dato sostitutivo. Omega resta una prova morbida: un fronte maturo
o frontolitico può esistere senza forte ascesa istantanea.

La fonte ufficiale descrive ICON-2I come modello deterministico a circa
2,2 km, dominio 3–22°E / 33–49°N, orizzonte +72 h:
[MeteoHub — dataset ICON-2I](https://meteohub.agenziaitaliameteo.it/app/datasets).

### Revisione geometrica v18–v19

Il controllo del run operativo ha mostrato che la fisica poteva essere
corretta mentre la linea finale risultava incoerente. Il problema nasceva
dopo il tracking: quando due tracce condividevano un lungo tronco, il
deconflitto conservava ad ogni ora il frammento indipendente più lungo. Quel
frammento poteva essere l'estremità occidentale a un'ora e quella orientale
all'ora successiva, producendo un salto di centinaia di chilometri che non
esisteva nella traccia meteorologica originale.

La v18 tratta l'output come un insieme di **archi frontali esclusivi**; la v19
aggiunge la riparazione conservativa dell'identità temporale:

- il tronco condiviso ha un solo proprietario;
- un frammento rifilato deve mantenere continuità di centroide, orientamento o
  sovrapposizione diretta con il frammento precedente;
- se due frammenti rifilati implicano oltre 110 km/h senza almeno il 45% di
  nucleo comune, viene soppresso quello meno completo;
- un arco freddo o caldo può contenere un tratto stazionario, ma non il tipo
  di moto opposto;
- se dopo un taglio resta soltanto il tratto stazionario, anche `frontType`
  diventa `stationary`;
- un ramo occluso riceve simboli occlusi su tutta la propria estensione e il
  ramo freddo residuo rimappa le frazioni sul nuovo arco;
- una geometria interpolata viene rimossa se non è coerente con almeno uno dei
  due rilevamenti che dovrebbe collegare: non si pubblica una linea che il
  modello non ha mai risolto;
- se due rilevamenti diretti restano incompatibili, vengono conservati entrambi
  ma ricevono identità pubbliche distinte; `trackingSourceId` mantiene la
  precedente associazione per l'audit e il moto non viene stimato attraverso
  la discontinuità;
- `publishedMotionQc` misura i salti sulla geometria finale, non sul candidato
  precedente al raffinamento.

Il vento medio 700–500 hPa entra soltanto come prior debole (25% dello
spostamento passivo) quando una traccia ha ancora una sola osservazione. Il
moto geometrico misurato resta prioritario: un fronte non è una particella
trasportata integralmente dal vento in quota. FI500 descrive l'onda associata
e viene archiviato nei diagnostici, senza spostare la linea termica.

## 1. Variabili termiche indipendenti

Da pressione, temperatura e umidità specifica si calcolano:

- temperatura potenziale secca `theta`;
- temperatura potenziale equivalente con la formulazione di Bolton;
- temperatura potenziale di bulbo umido `theta_w` con l'approssimazione di
  Davies-Jones (2008).

La geometria viene cercata separatamente sia su `theta_w` sia sulla
temperatura potenziale secca. `theta_w` distingue bene le masse d'aria umide;
la seconda ricerca recupera intrusioni fredde secche che un compenso di
umidità può rendere poco visibili in `theta_w`. I due insiemi vengono uniti e
deduplicati, ma nessuno dei due può pubblicare una linea senza superare gli
stessi controlli incrociati di temperatura secca, `theta_w` e densità
virtuale. Un confine di sola umidità viene quindi respinto.

## 2. Localizzazione OFA

Il nucleo segue Hewson (1998) nella formulazione portabile verificata da
Sansom e Catto (2024):

1. smoothing gaussiano normalizzato in chilometri, prima delle derivate;
2. gradiente metrico di `theta_w`;
3. Thermal Front Locator
   `TFL = laplacian(|grad(theta_w)|)`;
4. estrazione delle sole isolinee `TFL = 0`;
5. mascheramento della linea con TFP e intensità della zona baroclina
   adiacente (ABZ).

In parallelo viene eseguito un secondo localizzatore: la derivata seconda di
`|grad(theta_w)|` lungo la normale termica locale. È una realizzazione
metrica dell'idea direzionale di Hewson, dichiaratamente non una riproduzione
letterale dei suoi assi medi a cinque punti. Il localizzatore isotropo resta
il riferimento; la distanza simmetrica fra le due geometrie produce
`positionUncertaintyKm` e il loro accordo è una conferma indipendente. Anche
la linea direzionale deve superare identici corridoi e filtri. Non fonda però
una seconda traccia autonoma: può seguire l'altro bordo della stessa fascia
frontale finita e, se pubblicata separatamente, produrrebbe due fronti per una
sola discontinuità fisica. Conferma quindi le geometrie `theta_w`/`theta`
principali e ne quantifica l'incertezza di posizione.

Il Thermal Front Parameter è scritto nella forma di Sansom e Catto:

```text
TFP = grad(|grad(theta_w)|) · grad(theta_w) / |grad(theta_w)| < K1,  K1 <= 0
```

Sul bordo caldo della zona baroclina è negativo. Hewson scrive la stessa
quantità con un meno davanti e la cita quindi positiva sul bordo caldo: la
linea individuata è identica, cambia solo il segno del numero stampato.

Le soglie di riferimento sono quelle pubblicate, convertite nelle unità qui
usate: `K1 = −1,6e-11 K m⁻²` diventa **−1,6e-5 K/km²** e
`K2 = 7,5e-6 K m⁻¹` diventa **0,75 K/100 km**.

### La zona baroclina adiacente si misura dov'è

L'ABZ non è il gradiente *sulla* linea. Il TFL mette la linea sul bordo
**caldo** della zona, dove per costruzione il gradiente non ha ancora
raggiunto il massimo: confrontare quel valore con una soglia calibrata sulla
zona sottostima la baroclinicità e scarta fronti veri.

Hewson stima la zona con un'estrapolazione al primo ordine su `m` passi di
griglia. Funziona quando il passo di griglia *è* la risoluzione del campo
analizzato. Qui il campo è già lisciato a 45 o 100 km, quindi il passo ICON
di 8 km è il metro sbagliato: misurato sul run reale quell'estrapolazione
spostava il campione di 6 km e recuperava 0,02 degli 0,9 K/100 km di soglia,
cioè nulla.

La zona viene quindi campionata dove si trova davvero: camminando dalla
linea verso l'aria fredda (contro `grad(theta_w)`) il gradiente sale fino al
centro della zona e ridiscende, e il massimo su un cammino limitato è la
zona baroclina adiacente, per definizione. La lunghezza del cammino è la
scala di analisi `sqrt(sigma_campo² + sigma_derivata²)`: per un dosso di
gradiente di larghezza sigma il TFL sta a un sigma dal picco, e sul run
reale il massimo è stato trovato a 40–50 km con scala di analisi 47 km —
teoria e modello concordano. Il cammino è limitato perché un secondo fronte
più a valle non possa essere preso in prestito come zona di questo.

La stima locale di Hewson resta come pavimento, per le zone più strette
della scala di analisi.

Effetto misurato su fronti reali del run ICON-2I del 21 agosto 2026 00 UTC:

| | prima | dopo | massimo vero entro 150 km |
|---|---:|---:|---:|
| +18h | 1,81 | **2,45** | 2,60 |
| +24h | 1,60 | **2,06** | 2,12 |
| +48h | 2,37 | **3,29** | 3,29 |
| +52h | 1,99 | **3,03** | 3,22 |

(K/100 km. La ZBA sottostimava del 30–45%; ora coincide con il massimo vero.)

Le derivate rispettano le distanze reali della griglia lon/lat; cambiare
risoluzione non cambia implicitamente le unità delle soglie.

**Aggiornamento numerico v17.** Il TFL usa direttamente le differenze seconde
centrali, con formule unilaterali di secondo ordine ai bordi, anziché applicare
due volte la derivata prima. È la correzione esplicitamente raccomandata da
Sansom e Catto (2024): evita la perdita di accuratezza nelle derivate superiori
che rende le isolinee più rumorose. Il test analitico su un campo quadratico
verifica tutto il dominio, bordi inclusi. Il filtro gaussiano usa convoluzione
normalizzata con spazio esterno mancante: non riflette artificialmente un
ciclone o una discontinuità oltre il limite di ICON-2I. Le convoluzioni sono
eseguite in codice compilato (`scipy.ndimage`) mantenendo sigma in chilometri e
la correzione `cos(lat)` riga per riga.

## 3. Soppressione del rumore ICON-2I

Si applica un rilevatore a due scale sullo stesso modello:

- scala sinottica: sigma 100 km, derivata 30 km, linea minima 350 km;
- scala raffinata: sigma 45 km, derivata 15 km, linea minima 220 km;
- almeno il 60% della linea raffinata deve stare entro 110 km da una linea
  sinottica;
- margine di 70 km dal bordo del dominio.

La scala forte decide se esiste una struttura sinottica; la scala raffinata
ne conserva la geometria ICON-2I. Un gradiente locale molto intenso non può
aggirare il requisito sinottico: è il meccanismo principale contro brezze,
outflow convettivi, canali vallivi e cold pool.

Soglie fuzzy OFA (valori di ripiego, in unità fisiche):

| Scala | TFP debole → pieno | ABZ debole → pieno |
|---|---:|---:|
| Sinottica | −1,5e-5 → −4,0e-5 K/km² | 0,65 → 1,10 K/100 km |
| Raffinata | −2,5e-5 → −9,0e-5 K/km² | 0,90 → 1,70 K/100 km |

**Calibrazione climatologica (documento sez. 17).** Non si copiano valori
esteri: `scripts/calibrate_thresholds.py` costruisce una **climatologia** dei
due campi decisionali (`|∇θw|` e TFP), calcolati alle *stesse* scale di
smoothing del rilevatore, su molti run ICON-2I, separata per mese e fascia
latitudinale. Da lì ricava le soglie base — ABZ debole = Q₅₀, ABZ pieno =
Q₈₅ di `|∇θw|`; TFP debole = Q₂₅, TFP pieno = Q₀₅ — salvate in
`scripts/climatology_thresholds.json`. L'analyzer carica le soglie del mese
del run (fascia intero dominio), con **ripiego** ai valori fissi se la
climatologia manca o non copre quel mese. Una climatologia stabile su molti
run evita che una singola mappa quieta o tempestosa sposti la soglia; per
luglio (bassa baroclinicità) la soglia di gradiente scende, recuperando i
fronti estivi deboli, mentre il TFP a piena forza è più severo.

I quantili del **singolo** run possono poi rendere le soglie ancora più
severe, mai più permissive — ma **entro un limite**. Sansom e Catto leggono i
quantili da una climatologia, quindi la soglia è una proprietà fissa del
dataset. In linea l'unica distribuzione disponibile è quella dell'ora in
esame, che è un'altra cosa: si muove con il tempo che fa. Misurato su un run
ICON-2I il quantile grezzo faceva girare il rilevatore raffinato a
−6,6e-5…−8,4e-5 K/km² — **quattro o cinque volte più severo del K1
pubblicato** — con oscillazioni del 27% fra ore consecutive. Una soglia che
si muove ora per ora fa passare lo stesso confine alle 03 UTC e lo respinge
alle 04 UTC: è così che un fronte finisce per lampeggiare sulla mappa.

Il quantile resta, perché sopprimere struttura di piccola scala è un
servizio reale, ma può irrigidire il valore configurato al massimo di un
fattore `ADAPTIVE_TIGHTENING_LIMIT = 1,5`. In pratica satura sul limite,
quindi il punto di lavoro diventa costante lungo tutta la corsa e il
tremolio sparisce con lui. Il quantile del gradiente si legge inoltre sul
`|∇θw|` semplice e non sull'ABZ: l'ABZ è il gradiente massimizzato lungo un
cammino, quindi la sua distribuzione è spostata verso l'alto per
costruzione, e calibrare la soglia ABZ sull'ABZ stesso alzerebbe l'asticella
esattamente quanto è migliorata la misura, annullandola in silenzio.

Vengono inoltre respinte anomalie quasi chiuse, hairpin molto sinuosi,
duplicati paralleli e linee prevalentemente sopra terreno elevato.

## 4. Controlli fisici indipendenti

Ogni linea viene orientata in modo univoco con l'aria calda a sinistra. Le
masse d'aria vengono campionate a **30, 55 e 85 km** su entrambi i lati: la
mediana evita che una singola coppia cada su una brezza, una valle o un
artefatto di griglia. Si misurano:

- salto di `theta_w`;
- salto di temperatura secca;
- salto di temperatura potenziale virtuale, proxy della densità, e di
  `theta_e` come controllo del contrasto umido;
- allineamento fra normale frontale e gradiente secco;
- rotazione del vento e convergenza normale.

Sulla linea si calcolano inoltre:

- vorticità e convergenza orizzontale;
- frontogenesi orizzontale di Petterssen/Keyser, in
  K/(100 km)/(3 h);
- avvezione termica;
- velocità frontale OFA;
- profondità della saccatura PMSL a ±100 km e tendenza isallobarica;
- supporto 925 hPa e omega 700 hPa, quando disponibili.
- contrasto e profilo trasversale a 700 hPa, coerenza 925–850–700 e
  inclinazione della superficie frontale;
- indice Parfitt-F a 925 hPa (fallback 850 hPa), con soglia di riferimento
  `F = 1`;
- variazione del vento a 10 m normalizzata a sei ore, incluso il classico
  passaggio da vento da SW a vento da NW descritto dal metodo WND.

Parfitt-F e WND sono campionati **solo lungo linee già termiche**: non
generano geometrie e non sono porte d'esistenza. Sono conferme limitate,
particolarmente utili quando i localizzatori geometrici concordano ma una
singola diagnostica è marginale.

I requisiti duri impongono un vero contrasto secco e di densità, persistenza
del contrasto alle tre distanze, allineamento termico plausibile e geometria
sinottica.
### Motore di ragionamento (diagnosi differenziale)

L'algoritmo ragiona come un meteorologo, non come una catena di soglie:
**individua un possibile fronte** (TFL/TFP/ABZ), **raccoglie tutte le prove
disponibili** (contrasto di θw, temperatura secca, densità virtuale, gradiente
secco, corridoio sinottico, lunghezza, geometria, vento, convergenza,
vorticità, frontogenesi, saccatura, 925/700 hPa, Parfitt-F, WND), **osserva
l'evoluzione nel tempo** (durata, copertura, coerenza del moto), **confronta le
spiegazioni alternative** e **sceglie l'ipotesi che spiega meglio l'intero
quadro**, rendendo trasparente il motivo.

Il motore (`front_physics.differential_diagnosis`) assegna un punteggio di
coerenza a **ogni** ipotesi in competizione — fronte sinottico, confine di
umidità/dryline, confine mesoscalare (brezza), outflow convettivo, segnale
orografico, rumore debole — a partire da *tutte* le caratteristiche osservate,
comprese quelle temporali: un fronte sinottico **persiste e si muove in modo
coerente** per molte ore, mentre una brezza o un outflow sono **transitori** e
spesso diurni. Vince l'ipotesi meglio supportata.

La decisione è **conservativa come quella di un analista**: un candidato già
localizzato che possiede il contrasto di massa d'aria è considerato un fronte,
a meno che una spiegazione concorrente non prevalga **chiaramente** — così un
fronte reale ma modesto non viene scartato da un punteggio di "rumore"
appena più alto. Se però manca del tutto il contrasto termico di massa d'aria
(solo umidità), l'ipotesi di dryline vince e la linea non entra nel
tracciamento. Ogni fronte pubblicato espone `reasoning` (verdetto, margine,
ipotesi alternativa e motivi) e `explanation`.

**Il margine di protezione è graduato, non commutato.** Prima svaniva di
colpo appena il contrasto di massa d'aria scendeva sotto una soglia netta:
a quel valore una differenza di un millesimo fra due ipotesi decideva il
verdetto, e un confine appoggiato sul gradino veniva letto come fronte alle
03 UTC, come linea mesoscalare alle 04 e di nuovo come fronte alle 05, senza
che nulla fosse cambiato nell'atmosfera. Ora la protezione svanisce insieme
al contrasto: nessun contrasto, nessuna protezione (la dryline perde
comunque); contrasto solido, protezione piena.

**Il contrasto di massa d'aria è un AND morbido.** Servono tutti e tre —
θw, temperatura secca e densità — ma un `min` secco lascia che il singolo
ingrediente più debole annulli il punteggio, e il più debole è quasi sempre
la densità. Sul run reale un confine da 900 km con un contrasto di θw di 2 K
e un contrasto secco pulito segnava 0,14 solo perché Δθv valeva 0,5 K. La
media geometrica va ancora a zero quando un ingrediente manca davvero — è
ciò che rende un fronte un fronte — ma degrada dolcemente quando uno è
soltanto modesto, che è il caso normale in una massa d'aria estiva.

Solo l'ipotesi sinottica può entrare nel tracciamento.

Le porte non compensabili sono volutamente poche:

- contrasto simultaneo di `theta_w`, temperatura secca e densità virtuale;
- orientamento corretto del gradiente secco per almeno metà della linea;
- firma sinottica a due scale e geometria aperta/coerente;
- rifiuto esplicito della divergenza forte non sostenuta dalla dinamica;
- rifiuto dei confini spiegati meglio da umidità sola, orografia o
  convergenza locale senza corridoio sinottico.

Vento, frontogenesi, saccatura barica, 925 hPa e omega a 700 hPa sono prove
indipendenti che aumentano o riducono la fiducia. Una cresta barica o una
temporanea incoerenza a 925 hPa non cancella da sola una linea termicamente
coerente: può impedirne la nascita come traccia forte e mantenerla soltanto
come continuazione già verificata. Il vento resta decisivo per etichettare il
tipo freddo/caldo, non per nascondere un confine di masse d'aria reale.

Le statistiche sono calcolate anche come frazioni della linea: un ottimo
segnale su pochi punti non può nascondere una maggioranza incoerente.

Soltanto dopo queste porte il punteggio fisico di base ordina le linee ammesse:

- 38% termodinamica;
- 24% dinamica;
- 10% pressione;
- 10% coerenza verticale;
- 4% attività a 700 hPa;
- 14% struttura.

Il consenso multi-metodo può aggiungere al massimo **0,04**. Non sottrae
evidenza e non compensa un gate fallito: serve a distinguere una linea
confermata da più metodi da una linea sostenuta da una sola geometria.

È un indice trasparente e diagnostico, non machine learning addestrato su
etichette e non una probabilità.

## 5. Tracciamento e tipo di fronte

I candidati orari sono associati globalmente con algoritmo ungherese usando
distanza simmetrica fra linee, orientamento, lunghezza, normale calda e
posizione prevista. Un'identità non può saltare più di due ore.

Il tipo non viene fissato una volta per l'intera vita della traccia: viene
ricalcolato in una finestra temporale locale per ogni scadenza. Lo stesso
confine può quindi passare in modo coerente da freddo a stazionario senza
sparire e senza conservare un simbolo ormai vecchio.

Per essere pubblicata una traccia deve avere:

- almeno 4 rilevamenti;
- durata di almeno 3 ore;
- copertura temporale almeno 72%;
- `qualityScore >= 0.61`;
- `uncertaintyIndex <= 0.39`;
- classificazione non conflittuale.

La classificazione confronta tre famiglie indipendenti:

1. spostamento geometrico orario verso il lato caldo/freddo;
2. velocità di fase `-d(theta_w)/dt / |grad(theta_w)|`;
3. famiglia del vento: velocità OFA
   `V · grad|grad(theta_w)| / |grad|grad(theta_w)||` e componente del vento
   normale al confine delle masse d'aria. Dove possibile si usa il vento a
   925 hPa, con fallback a 850 hPa sopra il terreno elevato.

Con soglia 1,5 m/s (5,4 km/h), moto verso l'aria calda = fronte freddo,
moto verso l'aria fredda = fronte caldo, altrimenti stazionario. Un tipo
mobile richiede almeno due famiglie concordi e nessun voto opposto. Una
contraddizione forte del vento rende la traccia `uncertain`, anche quando la
linea termica sembra muoversi correttamente.

**Stabilità temporale del tipo (Viterbi/HMM).** La classificazione oraria
locale coglie l'evoluzione reale di un fronte, ma il moto geometrico che
oscilla attorno allo zero — un fronte che rallenta — può far leggere per
un'ora il tipo opposto. Un fronte caldo **non** diventa freddo l'ora dopo:
per impedirlo, sulla sequenza dei tipi di ogni traccia si applica un passo di
**Viterbi** (documento sez. 13) con transizioni fisicamente plausibili — il
salto freddo↔caldo tra ore adiacenti è quasi proibito, il rallentamento verso
stazionario è ordinario. Il percorso più coerente elimina i cambi impossibili
(es. `…S C F S…` diventa `…S S S S…`) preservando le fasi reali e prolungate;
le ore corrette restano marcate (`typeSmoothed`, `rawFrontType`) e con
certezza ridotta.

### Occlusione

L'occlusione non è decisa dal gradiente di una singola ora, ma dal contesto
spaziale (documento sez. 14): sui fronti già pubblicati e sul campo PMSL si
cerca un **minimo barico reale** (prominenza ≥ 2 hPa sull'anello circostante)
a cui siano ancorati **sia un fronte freddo sia un fronte caldo** che si
incontrano in un **punto di tripla giunzione**. Ciò che distingue
l'occlusione dall'onda aperta è che il punto triplo è **spostato dal minimo**
e il fronte freddo **si avvolge** dal punto triplo fino al centro depressione:
quel ramo avvolto diventa un fronte separato di tipo `occluded` (simbolo viola,
triangolo e semicerchio alternati sullo stesso lato). Il criterio è
volutamente conservativo — un fronte freddo isolato o un'onda aperta (con
entrambi i fronti che partono dal minimo, senza ramo avvolto) **non** produce
un'occlusione.

Ogni candidato che raggiunge il tracciamento ha già superato le porte
fisiche dure (contrasto termico/densità, corridoio sinottico, diagnosi
`synoptic-front`): è un segmento frontale reale, non rumore. Può quindi
**dare origine a una traccia** anche quando è di grado "continuazione" e non
"forte" — altrimenti un fronte sinottico lento e quasi stazionario, che
raramente raggiunge il grado forte, non nascerebbe mai e verrebbe perso. La
persistenza spuria è comunque scartata dalle soglie di durata, numero di
rilevamenti e copertura. Non si impone più un limite alla lunghezza dei
tratti "continuazione" consecutivi: un confine coerente e ben coperto per
molte ore non viene cancellato solo perché di rado tocca il grado forte; il
`qualityScore` lo pondera senza eliminarlo. Una lunga perdita di segnale
(oltre la finestra di tracciamento) termina davvero l'identità.

L'incertezza diagnostica penalizza le dimensioni di **esistenza** deboli —
contrasto termico e coerenza strutturale — ma non la dinamica: un fronte
quasi stazionario ha per natura vento trasversale e convergenza deboli e non
deve per questo risultare "incerto" ed essere scartato. All'interno di una
traccia pubblicata, un'ora la cui classificazione locale è ambigua viene
mostrata con il tipo dominante della traccia, così la linea resta continua
invece di sparire per una singola ora.

**Recupero della fase debole (doppia soglia).** Un fronte reale spesso esiste
per ore sotto le porte dure di pubblicazione (nucleo termico debole all'inizio
della sua vita) prima di rafforzarsi in traccia pubblicata. Quelle ore non
sono perse: ogni candidato respinto ma diagnosticato `synoptic-front` dal
motore differenziale viene conservato per intero. Una traccia pubblicata
estende poi la propria identità all'indietro e in avanti attraverso i
candidati deboli consecutivi che la continuano fisicamente — è il
ragionamento del previsore: *il fronte solido di stasera era già lì stamane,
lo dimostra la continuità*. Il criterio di aggancio tollera i cambi di
estensione (i frammenti deboli di un confine lungo scorrono e si spezzano):
decide la **sovrapposizione** (≥40% della linea più corta entro ~65 km/h·gap
dalla più lunga), con orientamento, lato caldo e spostamento trasversale del
centroide limitati. Come nel doppio-soglia di Canny, i deboli possono solo
**estendere** una traccia affermata, mai fondarne una: il rumore non può
auto-promuoversi a fronte. Identità, tipo e qualità restano calcolati sul
solo nucleo forte (`coreHours`), così le ore recuperate allungano la linea
nel tempo senza gonfiarne la confidenza; le ore restano ispezionabili in
`recoveredHours`.

**Ricucitura delle tracce spezzate.** Il tracciamento è causale: elabora
un'ora alla volta e non può guardare avanti. Se un fronte reale scende sotto
la soglia di rilevamento per una singola ora, la predizione a due ore che lo
riaggancerebbe è troppo severa e il tracker lo spezza in due tracce
consecutive, con un buco di un'ora in mezzo — il fronte "sparisce e riappare".
Con l'intera sequenza in mano si rifà il controllo *a posteriori*: due tracce
separate da **esattamente un'ora mancante** vengono riunite in una sola se il
loro moto è fisicamente plausibile — spostamento del centroide sotto ~65 km/h
(guardia anti-teletrasporto, perché due linee lunghe parallele possono avere
distanza simmetrica piccola pur essendo fronti diversi), piccola distanza
linea-linea, orientamento e lato caldo compatibili. Il buco di un'ora così
ricucito è poi colmato dall'interpolazione descritta sopra, e la linea è
disegnata come un unico fronte continuo. Una perdita di segnale più lunga
resta invece una nuova identità.

**Risoluzione topologica dei tronchi comuni.** Il raffinamento finale aggancia
ogni traccia alla cresta continua del supporto fisico. Due identità temporali
possono quindi convergere sulla stessa cresta per un lungo tratto e poi
separarsi in rami: disegnarle entrambe in quel tratto crea la sovrapposizione
grafica e attribuisce due identità allo stesso fronte. Dopo il raffinamento,
un corridoio condiviso entro 25 km per almeno 140 km viene assegnato una sola
volta alla traccia con qualità oraria, rilevamento e tipo più solidi. Della
traccia più debole sopravvive il ramo indipendente più lungo, purché misuri
almeno 100 km; le frazioni dei tipi locali vengono rimappate sulla nuova
geometria. Un incrocio puntuale o una normale giunzione freddo/caldo non
raggiunge la lunghezza minima e rimane quindi intatto.

## 5b. Qualità oraria e qualità di traccia (v15)

Fino alla v14 `qualityScore`, diagnostica e spiegazione erano mediane
dell'intera traccia, ripetute identiche in ogni ora: un fronte che si
indeboliva a una scadenza non lo mostrava. Dalla v15 i due livelli sono
separati e coesistono nel GeoJSON:

- **`trackQualityScore` / `trackUncertaintyIndex` / `trackDiagnostics`** —
  il giudizio sulla traccia (persistenza, coerenza del moto, copertura):
  decide se la traccia è abbastanza affidabile da essere **pubblicata**;
- **`qualityScore` / `uncertaintyIndex` / `diagnostics`** — la vista
  dell'**ora corrente**: quanto i campi di QUELLA scadenza sostengono il
  fronte. Miscela diagnostica (non probabilità): 58% evidenza fisica
  dell'ora, 22% affidabilità del collegamento temporale, 10% supporto
  strutturale, 10% confidenza della classificazione;
- **`detectionQuality`** — l'evidenza del candidato dell'ora;
  **`trackingConfidence`** — quanto è solido il collegamento temporale in
  quell'ora (vicini stretti = alto; buco ricucito o recupero debole =
  ridotto); **`classificationConfidence`** — certezza locale del tipo.

Un'ora **interpolata** non simula mai un rilevamento: `detectionQuality` e
`trackingConfidence` sono `null`, `interpolated: true`, la diagnostica è
interpolata dalle due osservazioni vicine e il punteggio porta una
penalizzazione esplicita. Il tracking continua a impedire che un fronte
reale sparisca per una singola ora debole, ma la qualità di quell'ora resta
onestamente bassa. La spiegazione meteorologica descrive l'ora corrente; la
persistenza della traccia vi entra separatamente (`lifetimeH`).

## 5c. Sezioni trasversali e coerenza verticale (v15)

`front_sections.py` campiona θw lungo la normale fredda→calda a offset
metrici fissi (−85…+85 km) e ne deriva diagnostiche fisiche per candidato:
frazione valida del profilo (`profileValidFraction`), frazione della linea
con contrasto termico coerente (`profileThermalSupport`), gradiente massimo
del profilo (`profilePeakGradient`), posizione del massimo rispetto alla
linea (`frontOffsetKm`), **larghezza frontale** (`frontWidthKm`, zona in cui
il gradiente resta ≥50% del massimo) e omogeneità delle masse d'aria
(`airMassHomogeneity`). La larghezza resta una diagnostica: non è usata come
soglia finché non sarà calibrata su un archivio indipendente.

Con i dati a 925 hPa lo stesso profilo è confrontato fra le due quote:
`verticalCoherence` combina segno del contrasto, rapporto dei gradienti,
distanza fra i massimi (un fronte reale è inclinato: uno scarto modesto è
normale), somiglianza delle larghezze (`frontWidth925Km`) e frazione valida
a entrambe le quote. Entra nella componente verticale dell'evidenza con
peso prudente (20%); i dati 925 mancanti — incluse le zone dove la
superficie interseca il terreno, già mascherate — sono **neutri**, mai prova
contraria: una grave incoerenza riduce l'evidenza ma non cancella da sola un
fronte ben sostenuto a 850 hPa.

Dalla v16 lo stesso profilo è calcolato anche a 700 hPa. Le coerenze
925–850 e 850–700 vengono combinate in `verticalCoherence3Level`; la
differenza fra gli offset 700 e 925 hPa misura `frontalTiltKm`. L'inclinazione
è diagnostica, non un veto, perché orografia, occlusioni e deformazione
verticale possono produrre strutture reali non ideali.

## 6. Significato dell'incertezza

Il GeoJSON separa quattro giudizi che non vanno confusi:

- `existenceConfidence`: robustezza diagnostica dell'esistenza nell'ora;
- `typeConfidence`: certezza della classificazione freddo/caldo/stazionario;
- `positionUncertaintyKm`: disaccordo metrico fra localizzatori indipendenti;
- `methodAgreement`: quanti controlli indipendenti concordano e quanti erano
  disponibili.

Restano euristiche esplicitamente marcate
`heuristic-not-calibrated-probability`; `confidence` e `qualityScore` sono
mantenuti per compatibilità con l'interfaccia esistente.

Precipitazione e radar restano fuori dai gate. Lo script
`front_impact_validation.py` misura offline tasso di precipitazione vicino e
lontano dalle linee già rilevate: valida l'impatto meteorologico, ma non può
creare né cancellare un fronte secco.

L'indice aumenta con prove fisiche deboli, scarsa continuità, moto instabile,
geometria dubbia o classificazione discordante. Il sito pubblica soltanto la
classe interna bassa. Rimangono fuori dall'indice:

- errore del run ICON-2I e sensibilità alle condizioni iniziali;
- errore sistematico del modello;
- verifica contro osservazioni, radiosondaggi e analisi manuali;
- probabilità ensemble.

Perciò “incertezza interna bassa” significa che il fronte è coerente con i
campi del run, non che la posizione reale sia garantita. Per una probabilità
meteorologica occorrerebbero ensemble e verifica retrospettiva con un archivio
di analisi ufficiali.

## 6b. Output spiegabile

Il ragionamento non è nascosto dietro un solo numero. Ogni fronte pubblicato
espone, oltre a `qualityScore`/`uncertaintyIndex` e ai `diagnostics` numerici:

- `diagnosis`: l'ipotesi vincente della diagnosi differenziale (fronte
  sinottico, confine di umidità, orografico, mesoscalare, promontorio);
- `reasoning`: il ragionamento del motore — verdetto, margine sul secondo
  classificato, ipotesi alternativa con il suo supporto e i motivi della
  scelta (compreso l'argomento temporale: persistenza e moto coerente);
- `explanation`: le ragioni in linguaggio umano che verbalizzano le prove
  numeriche già calcolate — contrasto di masse d'aria (Δθw), zona baroclina
  adiacente, contrasto secco/densità, rotazione e convergenza del vento,
  vorticità/frontogenesi, saccatura barica, coerenza a 925 hPa, moto verticale
  a 700 hPa, persistenza e concordanza di classificazione.

La spiegazione non introduce nuova fisica: rende leggibile ciò che i punteggi
codificano, così l'utente vede *perché* una linea è un fronte.

## 6c. Campo continuo di supporto (Fase B, ispirato a Biard & Kunkel 2019)

Biard e Kunkel (2019) lasciano che una rete neurale produca un campo di
probabilità per-pixel da cui poi estraggono le linee. `front_support.py` ne
adotta l'**idea architetturale** — un campo continuo che separa l'**esistenza**
di una struttura frontale dal suo **tipo** — ma lo costruisce
**analiticamente dalla fisica**, non da una rete addestrata: nessun machine
learning. Ogni componente è una membership normalizzata in [0, 1] di una
diagnostica già affidabile:

- `thermal`, `abz`, `tfp` (nucleo termodinamico); `dry_thermal`, `moisture`;
- `dynamic` (convergenza, vorticità, frontogenesi); `pressure` (saccatura);
- `vertical` (925/700 hPa); `synoptic` (gradiente alla scala 100 km).

Con penalità esplicite: `terrain`, `edge` (bordo dominio), `missing_data`
(NaN gestiti esplicitamente, mai azzerati in silenzio), `moisture_boundary`
(gradiente di sola umidità), `local_scale` (struttura solo mesoscalare senza
supporto sinottico). La combinazione pesata è `any_front_support` ∈ [0, 1],
un supporto fisico **non** una probabilità: dice *quanto* i campi sostengono
la presenza di un fronte, indipendentemente da freddo/caldo/stazionario.
Dalla v17 è separato da `geometry_support`, che contiene solo le prove adatte
a **posizionare** la linea: gradiente θw, ABZ, TFP sul bordo caldo, contrasto
secco e scala sinottica. Vento, frontogenesi e saccatura possono confermare
l'esistenza, ma i loro massimi sono spesso spostati rispetto alla
discontinuità termica e non devono trascinare il disegno fuori dal confine fra
le masse d'aria.

Nessun candidato scartato sparisce in silenzio: `rejected_candidates(hour)`
esporta le linee respinte con `rejectedAs` e i motivi, e
`analysis_summary.rejectedByReason` ne conta le cause.

### Geometria a cresta (Fase C/E, `front_ridge.py`)

La linea pubblicata non è più il solo contorno TFL: segue la **cresta termica**
di `geometry_support`. Dato un candidato TFL–TFP–ABZ, si apre un **corridoio**
di ±120 km attorno alla linea e si cerca il **percorso a costo minimo**
(Dijkstra su griglia 8-connessa, `scipy`; nessuna nuova dipendenza, nessun ML)
che resta sul supporto più alto. Il costo per cella è
`log((1+ε)/(supporto+ε))`, più penalità esplicite di terreno/bordo e una
penalità morbida della distanza dal contorno TFL originario: una cresta più
forte può correggere la posizione, ma il percorso non può saltare liberamente
su un fronte vicino. Ogni passo è pesato con i suoi **chilometri reali**
(`dx·cos(lat)`, `dy`, diagonale metrica), non con la distanza fra pixel.

Gli estremi vengono spostati solo localmente (massimo 45 km) verso una cella
sostenuta, evitando code artificiali che partono da terreno o da un buco. Il
percorso ottimo su griglia è a gradini: uno smoothing di **Chaikin** rimuove la
scala di griglia e un piccolo RDP declutter. Prima di pubblicarlo, una guardia
obbligatoria respinge auto-intersezioni, scorciatoie con lunghezza implausibile,
uscite dal corridoio e qualunque linea che perda supporto termico rispetto al
TFL. In tutti questi casi torna automaticamente il contorno originale.

L'estrazione è **subordinata alla fisica**: non inventa una linea dove il
supporto è alto, rifinisce *dove disegnare* un fronte già rilevato, dentro un
corridoio limitato. È **completamente protetta** — qualsiasi errore o risultato
degenere ripiega sul contorno originale, quindi la pubblicazione non può mai
regredire a una linea vuota o rotta. Il numero di fronti, i tipi e i segmenti
restano invariati: cambia solo la posizione della linea.

L'attivazione originaria (`REFINE_PUBLISHED_GEOMETRY = True`) è avvenuta dopo
il **benchmark Fase E** su 3 run reali (24 linee, metrica equa a passo uniforme):
supporto medio lungo la linea 0.43 → 0.50, frazione su supporto forte 0.50 →
0.67, tortuosità 7.27 → 6.77 °/20 km. La cresta non regredisce su nessun
criterio e migliora nettamente il supporto fisico; l'interruttore resta
reversibile (`False` torna ai contorni). La v17 aggiunge le guardie numeriche e
topologiche sopra descritte; la misura dell'errore meteorologico assoluto resta
comunque subordinata a un archivio indipendente di analisi frontali ufficiali.

**Il limite di posizione trovato durante l'audit e' stato corretto in v17.**
Misurato su un run reale, il raffinamento spostava la linea disegnata rispetto
al contorno TFL con mediana −4 km ma dispersione ampia, da −43 a +62 km, e
tendenza verso l'aria fredda nel 61% dei casi: il campo di supporto era
dominato da termini che culminano al *centro* della zona baroclina, mentre
solo il TFP culmina sul bordo caldo dove Hewson definisce la linea. La
separazione di `geometry_support` risolve esattamente questo: la geometria
usa solo le prove termodinamiche di bordo caldo, con il TFP al peso dei
termini di gradiente, e le firme dinamiche e bariche restano prove di
*esistenza* senza tirare la linea via dal confine di massa d'aria.

### L'aria calda deve stare a sinistra della linea pubblicata

Da questa convenzione dipende il lato dei simboli: il renderer mette i
triangoli del fronte freddo a sinistra del verso di percorrenza e i
semicerchi del caldo a destra, quindi se l'aria calda finisse dal lato
sbagliato il fronte verrebbe disegnato come se avanzasse all'indietro.

`_orient_warm_left` stabilisce la convenzione sul **candidato**, ma la linea
pubblicata non è il candidato: viene agganciata alla cresta del supporto e
può essere tagliata dall'occlusione, e la normale calda memorizzata non
viene ricalcolata. Misurato su un run, **2 linee su 80** uscivano con l'aria
calda a destra — entrambe confini i cui due lati differiscono di pochi
decimi di kelvin, dove l'orientamento ereditato è meno affidabile.

Il lato viene quindi rimisurato contro il campo `theta_w` sulla geometria
finale (`_orient_published_line`): invertire l'ordine dei punti sposta i
simboli dall'altra parte della linea, e l'eventuale carattere per-segmento
viene specchiato con essa. Dopo la correzione: 0 linee su 80.

**Classificazione per-segmento (`segmentTypes`).** La stessa struttura può
essere freddo attivo su un tratto e quasi stazionaria su un altro **alla
stessa ora**: la geometria resta *una* linea continua (l'esistenza è tracciata
indipendentemente dal tipo), ma il carattere può variare lungo di essa. Ogni
punto è classificato dal suo **moto geometrico locale**, poi si uniscono
tratti adiacenti coerenti in segmenti espressi come frazioni di arco
`[start, end]` con tipo e certezza.

Vincolo fisico: **una singola identità frontale non diventa il tipo opposto**
lungo la sua lunghezza — un fronte freddo che diventasse caldo sarebbe un
altro fronte, o un'occlusione (gestita a parte). Il carattere può solo
indebolirsi verso *stazionario*. Poiché il moto per-punto di una sola ora è
rumoroso, ogni lettura del tipo opposto viene declassata a stazionario e le
ore incerte si ancorano al tipo dominante della traccia. Restano quindi le
variazioni reali (es. freddo→stazionario) senza falsi tratti opposti.

Il campo `segmentTypes` è **additivo**: il singolo `frontType` dominante resta
per retro-compatibilità. Il viewer disegna ogni segmento con i propri simboli
(il tratto di linea è ritagliato per frazione d'arco `[start, end]`), con
ripiego al tipo unico quando `segmentTypes` è assente — così una stessa linea
mostra il fronte freddo che sfuma in stazionario, senza spezzare la geometria.

## 7. Test automatici

La workflow blocca la pubblicazione se falliscono i test sintetici:

- segno TFP e posizione sul bordo caldo;
- invarianza a orientamento della griglia e risoluzione;
- accordo e casi nulli del localizzatore direzionale;
- Parfitt-F nullo in traslazione uniforme e positivo con vorticità ciclonica;
- WND positivo nel passaggio SW→NW e nullo con vento invariato;
- distanza metrica fra localizzatori e coerenza verticale 925–850–700;
- rimozione di patch locali e anomalie chiuse;
- traslazione rigida senza falsa frontogenesi;
- convergenza con frontogenesi positiva;
- rotazione rigida con vorticità ma senza falsa deformazione;
- rifiuto dei confini di sola umidità;
- rifiuto di gradiente termico con vento divergente/contrario;
- promontorio barico che indebolisce una linea ma non cancella un fronte
  termicamente coerente già tracciato;
- tracciamento, classificazione e separazione delle identità;
- consenso obbligatorio fra moto geometrico, fase termica e vento;
- continuità di una traccia lunga a prevalenza "continuazione" e persistenza
  visibile di un fronte quasi stazionario passato ogni porta fisica;
- riconoscimento dell'onda in occlusione e rifiuto di onda aperta / fronte
  freddo isolato come falsa occlusione;
- presenza e buona forma dell'array `explanation` su ogni fronte pubblicato;
- caricamento della climatologia delle soglie (otto valori ordinati, ripiego
  fra mesi) e loro applicazione nel rilevatore;
- il motore differenziale valuta tutte le ipotesi e sceglie il verdetto;
- stabilità temporale del tipo (Viterbi): rimozione del flip caldo↔freddo,
  conservazione di una fase reale prolungata;
- esclusività degli archi: nessun segmento caldo dentro un fronte freddo,
  nessun doppio segno su uno stazionario e simboli occlusi sull'intero ramo;
- stabilità post-deconflitto: soppressione del frammento che salta fra due
  estremità, conservando invece una normale crescita lungo lo stesso asse;
- maschera 925 hPa basata sulla pressione al suolo reale;
- estrazione a cresta (`front_ridge`): la linea rifinita segue il crinale del
  supporto staccandosi da una guess storta, una banda larga dà una sola linea,
  il percorso evita una penalità di terreno a parità di supporto, resta sulla
  cresta termica quando un massimo dinamico è spostato e rifiuta geometrie
  auto-intersecanti;
- Laplaciano esplicito: valore analitico esatto su un campo quadratico anche
  sui quattro bordi del dominio;
- la ZBA supera il gradiente sulla linea, non supera il massimo reale del
  campo e raggiunge la zona;
- la calibrazione adattiva non supera il limite di irrigidimento;
- il verdetto non oscilla lungo il contrasto di densità e il margine di
  protezione è continuo (il salto si riduce infittendo il campionamento,
  cosa che un gradino non farebbe);
- una linea con l'aria calda a destra viene riorientata e i suoi segmenti
  specchiati; una già corretta resta intatta;
- convenzione WMO dei simboli nel renderer (lato dei triangoli e dei
  semicerchi per freddo, caldo, occluso e stazionario) e scala cartografica
  che lega dimensione, spessore e passo.

## Riferimenti primari

- Hewson, 1998, *Objective fronts*:
  [Meteorological Applications](https://www.cambridge.org/core/journals/meteorological-applications/article/objective-fronts/DEF4E8845B0B5DBC102560C658FB4B6B)
- Sansom e Catto, 2024, *Objective identification of meteorological fronts and
  climatologies from ERA-Interim and ERA5*:
  [Geoscientific Model Development](https://gmd.copernicus.org/articles/17/6137/2024/gmd-17-6137-2024.html)
- Codice ufficiale collegato all'articolo:
  [phil-sansom/front_id](https://github.com/phil-sansom/front_id)
- Beckert et al., 2023, rilevamento tridimensionale e filtri fuzzy:
  [Geoscientific Model Development](https://gmd.copernicus.org/articles/16/4427/2023/gmd-16-4427-2023.html)
- Jenkner et al., 2010, fronti ad alta risoluzione e orografia alpina:
  [Meteorological Applications](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/met.142)
- Bitsa et al., 2021, criteri termo-dinamici nel Mediterraneo:
  [International Journal of Climatology](https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.7208)
- Davies-Jones, 2008, temperatura potenziale di bulbo umido:
  [Monthly Weather Review](https://journals.ametsoc.org/view/journals/mwre/136/7/2007mwr2224_1.xml)
- Parfitt, Czaja e Kwon, 2017, indice dinamico-termico F:
  [Journal of Climate](https://doi.org/10.1175/JCLI-D-16-0904.1)
- Schemm, Rudeva e Simmonds, 2015, rilevamento e WND temporale:
  [Journal of Climate](https://doi.org/10.1175/JCLI-D-14-00718.1)
- Niebler et al., 2022, classificazione multivariata e valutazione a oggetti:
  [Weather and Climate Dynamics](https://doi.org/10.5194/wcd-3-113-2022)
- Berry, Reeder e Jakob, 2011, climatologia globale dei fronti:
  [Geophysical Research Letters](https://doi.org/10.1029/2010GL046451)
- Biard e Kunkel, 2019, rilevamento supervisionato a campo continuo:
  [Advances in Statistical Climatology, Meteorology and Oceanography](https://doi.org/10.5194/ascmo-5-147-2019)
- Dagon et al., 2022, fronti ML e associazione con precipitazioni estreme:
  [JGR Atmospheres](https://doi.org/10.1029/2022JD037038)
