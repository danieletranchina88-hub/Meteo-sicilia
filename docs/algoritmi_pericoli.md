# Diagnostiche meteorologiche automatiche

## Principio operativo

I prodotti di `meteo_analysis` sono diagnostiche deterministiche derivate da
ICON-2I. Non sono probabilità calibrate su un ensemble e non sostituiscono
allerte o bollettini ufficiali. Un campo mancante viene pubblicato come
`null`/`unavailable`: non sono ammessi valori climatologici costanti o variabili
ricavate artificialmente dalla temperatura a 2 metri.

## Innesco convettivo

Metodo corrente: `icon2i-mlcape-cin-convergence-front-v2`.

Input:

- ML-CAPE e ML-CIN nativi ICON-2I;
- convergenza `-(du/dx + dv/dy)` del vento a 10 m, con derivate metriche su
  griglia latitudine/longitudine e smooth gaussiano di circa 10 km;
- distanza dalla linea frontale OFA più vicina;
- omega a 700 hPa, quando disponibile;
- umidità relativa superficiale come modulatore secondario.

La classe alta è possibile soltanto con:

`CAPE > 800 J/kg`, `CIN > -50 J/kg`, convergenza `> 1e-4 s-1` e fronte entro
50 km. Senza questa combinazione il risultato è limitato sotto il 70%. CAPE
debole, CIN forte o divergenza impongono ulteriori limiti. L'output massimo è
95%, per ricordare che il prodotto è una regola esperta e non una probabilità
ensemble calibrata.

Il file `hazard_qc.json` salva per ogni ora massimo, 95° percentile e frazione
di area sopra 40% e 70%. Il test `test_convection.py` contiene una regressione
specifica contro il precedente campo nazionale fisso all'80%.

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

## Prodotti tridimensionali

SHIP/SCP, gelicidio e foehn richiedono profili e diagnostiche indipendenti. Le
funzioni fisiche sono presenti nel pacchetto, ma il processore operativo
pubblica questi layer come non disponibili finché il run non fornisce tutti
gli input richiesti:

- SHIP: CAPE, rapporto di mescolanza, lapse rate 700–500 hPa, temperatura a
  500 hPa e quota dello zero termico;
- SCP: CAPE, SRH e bulk shear 0–6 km reali;
- gelicidio: profilo termico 925/850/700 hPa, temperatura a 2 m e
  precipitazione;
- foehn: vento perpendicolare alle Alpi a 700 hPa, differenza di pressione
  nord-sud e umidità sottovento.

Questa scelta evita che una mappa visivamente completa comunichi un rischio
fisicamente inesistente.
