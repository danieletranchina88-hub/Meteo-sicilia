# Diagnostiche meteorologiche automatiche

## Principio operativo

I prodotti di `meteo_analysis` sono diagnostiche deterministiche derivate da
ICON-2I. Non sono probabilità calibrate su un ensemble e non sostituiscono
allerte o bollettini ufficiali. Un campo mancante viene pubblicato come
`null`/`unavailable`: non sono ammessi valori climatologici costanti o variabili
ricavate artificialmente dalla temperatura a 2 metri.

## Innesco convettivo

Metodo corrente: `icon2i-mlcape-cin-convergence-front-v3`.

Input:

- ML-CAPE e ML-CIN nativi ICON-2I;
- convergenza `-(du/dx + dv/dy)` del vento a 10 m, con derivate metriche su
  griglia latitudine/longitudine e smooth gaussiano di circa 10 km;
- distanza dalla linea frontale OFA più vicina;
- omega a 700 hPa, quando disponibile;
- umidità relativa superficiale come modulatore secondario.

La disponibilità e il formato sono verificati sul GRIB prima del calcolo:
`CAPE_ML` e `CIN_ML` devono avere unità energetiche `J kg-1`, 73 scadenze
orarie e valori fisicamente plausibili. Nel prodotto MeteoHub ICON-2I il CIN
è prevalentemente una magnitudine positiva; viene convertito nella convenzione
firmata negativa. Il sentinel `-999,9` viene trasformato in dato mancante e non
entra né nella probabilità né nelle statistiche QC.

La classe alta è possibile soltanto con:

`CAPE > 800 J/kg`, `CIN > -50 J/kg`, convergenza `> 1e-4 s-1` e fronte entro
50 km. Senza questa combinazione il risultato è limitato sotto il 70%. CAPE
debole, CIN forte o divergenza impongono ulteriori limiti. L'output massimo è
95%, per ricordare che il prodotto è una regola esperta e non una probabilità
ensemble calibrata.

Il file `hazard_qc.json` salva per ogni ora massimo, 95° percentile, celle
valide e frazione di area sopra 40%, sopra 70% ed esattamente all'80%. Il
deploy viene interrotto e mantiene il run precedente se l'area sopra 70%
supera il 15% del dominio oppure se un valore fisso all'80% ricopre oltre il
5%. Il test `test_convection.py` contiene una regressione specifica contro il
precedente campo nazionale fisso all'80%.

## Probabilità di temporale entro 10 km

Metodo corrente: `icon2i-physical-evidence-fusion-v2`.

L'evento dichiarato è «un temporale entro 10 km dal punto». Il valore di base
deriva dall'LPI nativo di ICON-2I con un vicinato circolare e uno smoothing di
25 km: questa scala rappresenta l'incertezza di posizione di una cella in un
modello convection-permitting, non trasforma la previsione deterministica in
un ensemble.

Come per i fronti, l'algoritmo formula un'ipotesi e cerca prove indipendenti,
coerenza e spiegazioni concorrenti. Le famiglie di evidenza sono:

- **segnale diretto:** LPI e corrente ascensionale convettiva, massimo fra
  omega a 850, 700 e 500 hPa;
- **ambiente:** ML-CAPE, MU-CAPE, CIN, umidità superficiale e a 700 hPa,
  altezza della base delle nubi;
- **innesco:** convergenza, brezza di mare o lago, risalita orografica, omega
  a 700 hPa e vicinanza a un fronte fisicamente diagnosticato;
- **corroborazione:** presenza dell'LPI nell'ora precedente o successiva,
  shear 0–6 km ed elicità dell'updraft.

I meccanismi alternativi di sollevamento usano il più forte, non la somma: una
brezza debole non diventa intensa solo perché cade vicino a un pendio o a un
fronte. Gli ingredienti necessari interagiscono invece in modo moltiplicativo,
così CAPE elevata senza umidità o innesco non genera da sola un'alta
probabilità; lo stesso vale per shear ed elicità.

L'algoritmo prova anche l'ipotesi opposta. Un LPI isolato, non persistente,
senza updraft e in ambiente ostile riceve una penalità di contraddizione. Una
cella confermata contemporaneamente da LPI e updraft non viene invece scartata
per CAPE locale bassa: nel nucleo maturo la CAPE può essere già stata consumata
e la cold pool può avere stabilizzato il suolo.

L'ambiente favorevole, in assenza di una cella esplicitamente risolta, è
limitato alla fascia di possibilità d'innesco e non supera il 42%. La fascia
alta cresce soltanto con supporto indipendente. Insieme alla probabilità sono
pubblicati `directEvidence`, `environmentSupport`, `instabilitySupport`,
`moistureSupport`, `liftSupport`, `temporalSupport`,
`organisationSupport`, `contradiction` e `confidence`: il valore finale resta
quindi spiegabile e controllabile punto per punto.

Questa è una probabilità diagnostica deterministica fisicamente vincolata, non
una probabilità statisticamente calibrata. La calibrazione assoluta richiede
un archivio di previsioni storiche e fulminazioni osservate, separato per
stagione, area geografica e lead time; fino ad allora il metodo dichiara
esplicitamente la propria confidenza e non attribuisce ai numeri un'affidabilità
che i dati disponibili non possono dimostrare.

## Bollettino automatico

Metodo corrente: `icon2i-conditional-nlg-v2`.

Il testo usa soltanto:

- tipi di fronte presenti nel GeoJSON dell'ora;
- distribuzione spaziale della probabilità convettiva;
- copertura delle precipitazioni;
- percentili di temperatura e vento;
- tendenza media di temperatura e pressione rispetto a tre ore prima.

La temperatura media non determina mai il tipo di fronte. Un massimo isolato
non viene descritto come fenomeno diffuso: il generatore considera anche il
95° percentile e la percentuale di celle sopra soglia. Nel JSON sono pubblicati
sia il testo compatibile (`nlg_bulletin`) sia la struttura verificabile
`nlg_bulletin_details`.

## Gelicidio

Metodo corrente: `icon2i-warm-nose-925-850-700-v1`.

Il rischio richiede contemporaneamente:

- temperatura a 2 m sotto 0 °C;
- uno strato caldo sopra 0 °C fra 925, 850 e 700 hPa;
- precipitazione realmente in arrivo nel passo di previsione.

La classe elevata richiede un warm nose di almeno 1 °C e T925 sopra zero,
segnale coerente con uno strato freddo superficiale relativamente sottile.

## Foehn alpino

Metodo corrente: `icon2i-cross-alpine-700-pmsl-rh-v1`.

Il campo combina vento perpendicolare all'asse alpino superiore a 15 kt a
700 hPa, differenza PMSL nord-sud di almeno 2 hPa e umidità sottovento sotto
40%. Una maschera geografica conservativa impedisce di classificare come
foehn normali correnti secche lontane dall'arco alpino.

## Prodotti ancora non disponibili

SHIP/SCP restano non disponibili finché il run operativo non fornisce tutti
gli input verticali indipendenti: lapse rate 700–500 hPa, temperatura a
500 hPa, quota dello zero termico, SRH reale e bulk shear 0–6 km.

Questa scelta evita che una mappa visivamente completa comunichi un rischio
fisicamente inesistente.

## Dominio e distribuzione

I GRIB MeteoHub ICON-2I usati dal sistema hanno una griglia regolare
761 × 761, da 3°E a 22°E e da 33,7°N a 48,9°N. Il processore non applica più
il precedente ritaglio sull'Italia: conserva l'intero dominio disponibile.
I file orari vengono pubblicati come JSON compresso gzip e trasformati in
array numerici compatti nel browser, così l'estensione geografica non richiede
un degrado della risoluzione.
