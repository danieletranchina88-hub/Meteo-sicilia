# Rilevamento oggettivo dei fronti — fisica e algoritmo

Questo documento spiega la fisica del rilevatore di fronti. Dalla **v11**
il sito usa l'analisi frontale oggettiva (OFA) con nucleo Hewson
(`scripts/front_analysis_v10.py`, metodo
`thetaw-850-icon2i-ecmwf-consensus-v11`),
che riusa l'infrastruttura dati validata della v9 (lettura GRIB,
validazione, griglia, campionamento, export 850 hPa) e ne sostituisce il
nucleo scientifico. La v9 (`scripts/front_analysis.py`) resta come
baseline euristica di confronto. La parte inferiore di questo documento
descrive la fisica della v9, in gran parte condivisa dalla v10.

## Architettura v11 (in produzione)

Quattro livelli separati, con test sintetici dedicati:

1. **Termodinamica** (`thermodynamics.py`) — campo primario θw
   (temperatura potenziale di bulbo umido, Davies-Jones 2008) a 850 hPa.
   θw è quasi-conservativa e, a differenza di θe, meno sensibile ai soli
   confini di umidità: riduce alla radice i "fronti" da dryline. Validata
   contro MetPy (polinomio 0.0000 mK, catena completa 0.023 K).
2. **Rilevamento a due scale** (`front_detection.py`) — un campo θw
   fortemente smussato (~150 km) fornisce il **corridoio sinottico**
   (prior strutturale); un campo leggermente smussato (~50 km) dà la
   **geometria rifinita**. Il localizzatore è il **Thermal Front Locator**
   di Sansom–Catto (TFL = ∇²|∇θw|, contorno zero) mascherato dal **TFP**
   (segno = lato caldo) e dalla **zona baroclina adiacente (ABZ)**. Un
   candidato ICON-2I di mesoscala isolato, privo di supporto sinottico,
   **non** viene pubblicato come fronte: è così che si eliminano i falsi
   fronti da alta risoluzione.
3. **Tracking geometrico globale** (`front_tracking.py`) — associazione
   **Hungarian** su tutta la previsione (nascita/morte; split e merge
   terminano una identità e ne iniziano un'altra), moto
   geometrico della linea lungo la normale verso il caldo come segnale
   **primario** di classificazione, con avvezione del vento e velocità OFA
   (Hewson) come riscontri. Classificazione per **consenso**: se i tre
   segnali concordano → tipo netto; se sono in conflitto reale →
   `uncertain` (meglio onesto che sbagliato). Il punteggio pubblicato è un
   `qualityScore` (euristica di supporto fisico, **non** una probabilità)
   con componenti termiche, dinamiche, temporali e strutturali. La voce
   `modelAgreement` viene valorizzata solo dal modello indipendente.
4. **Conferma indipendente ECMWF IFS (obbligatoria)** — ICON-2I rifinisce
   la geometria oraria, ma non decide da solo se un fronte esiste. Una
   traccia viene pubblicata soltanto se almeno il 60% delle ore è coerente
   con un fronte ECMWF già filtrato e tracciato, con almeno il 55% della
   linea in accordo entro 110–165 km. I candidati ECMWF grezzi non vengono
   usati come conferma. Se la guida manca, il sistema non pubblica fronti
   ICON non verificati.

Ogni candidato deve inoltre mostrare un gradiente reale di T a 850 hPa,
una rotazione o convergenza del vento, una forma aperta e sinottica e non
può seguire prevalentemente terreno oltre 900 m. Un gradiente intenso non
può bypassare questi filtri: brezze, dryline e outflow possono essere più
netti di un fronte vero.

Il numero di tracce non è prefissato: in condizioni senza strutture
sinottiche coerenti l'output corretto può essere vuoto. Metodo storico v9:
`theta-e-850-ofa-v9`.

**Robustezza numerica e consenso v11.** I campioni fuori
dal dominio restituiscono NaN (non il valore della cella di bordo): un
candidato con meno del 75% di punti interni viene scartato, per non
generare falsi fronti ai margini della mappa. Il confronto tra linee usa
la distanza punto-**segmento** (robusto anche con riferimenti a pochi
vertici) ed è simmetrico. Il collegamento temporale ha un raggio con
tetto fisso a 200 km. La copertura è contata sulle scadenze realmente
disponibili nell'arco della traccia. Quando la guida ECMWF è del tutto
assente la traccia resta marcata `corroborated: null` e la sua confidenza
è penalizzata, così l'assenza di conferma è esplicita nell'output.

## Stato: v9 baseline congelata

Questa è la **baseline euristica v9**, chiusa e congelata. È un
prototipo avanzato basato su θe + TFP=0 + gate multipli: fisicamente
sensato e temporalmente stabile, ma con due limiti concettuali noti
rispetto al metodo OFA completo — il localizzatore è `TFP=0` (asse del
massimo gradiente) invece di TFL+ABZ (bordo caldo della zona baroclina),
e il campo primario è θe (più sensibile ai confini di umidità) invece di
θw. La **v10** con nucleo Hewson completo (θw primaria, TFL+TFP+ABZ,
smoothing calibrato in km, rilevamento a due scale, tracking geometrico
globale) è **completata e in produzione** (vedi *Architettura v10* in
apertura); la v9 resta come riferimento di confronto.

**Chiusura v9 (robustezza finale, nessun nuovo criterio):**
- guardia di finitezza estesa a *tutte* le metriche (pressione,
  vorticità, frontogenesi, ΔTv, moto): un candidato con una firma NaN
  viene scartato, non fatto passare da un confronto con NaN;
- `_box_smooth` NaN-aware (media di finestra solo sui punti finiti): un
  NaN in ingresso non contamina più l'intero quadrante — protezione per
  GRIB con valori mancanti (i dati ICON-2I attuali non ne hanno);
- validazione esplicita dei GRIB all'apertura: livello 850 hPa, unità di
  T (kelvin), q (kg/kg, non g/kg), orografia (metri, non geopotenziale),
  e corrispondenza delle scadenze fra T/Q/U/V. Un input incoerente
  produce un errore chiaro invece di un risultato silenziosamente errato.

## Obiettivo dichiarato del prodotto

*Generare una rappresentazione automatica, fisicamente coerente e
temporalmente stabile delle principali strutture frontali sinottiche
previste dai modelli* — **non** riprodurre esattamente le linee
soggettive di un meteorologo (analisti esperti collocano diversamente
fronti deboli, occlusioni, split e sistemi mediterranei). Sul sito la
dicitura corretta è "fronti sinottici stimati automaticamente", non
"fronti ufficiali".

L'architettura è **multi-evidenza**: nessun singolo campo può "creare" un
fronte. Un candidato deve superare simultaneamente prove indipendenti su
**temperatura, umidità, vento e pressione**, più la coerenza nel tempo e
tra modelli. È l'approccio della letteratura operativa (Hewson 1998;
Parfitt et al. 2017; Thomas & Schultz 2019) adattato a un dominio
mediterraneo, dove i falsi positivi da contrasti terra-mare e da
convezione orografica sono il problema principale (Schemm et al. 2015).

## Cos'è un fronte

Un fronte sinottico è il bordo della zona di transizione tra **due masse
d'aria di origine diversa**. Non è un semplice bordo di gradiente: è una
struttura dinamica coerente, che possiede *insieme* tutte le firme che
seguono — ed è per questo che ogni firma è un criterio dell'algoritmo.

## Individuazione dei candidati

Si lavora a **850 hPa** (~1.5 km, sopra il rumore dello strato limite)
sulla **temperatura potenziale equivalente θe** (Bolton), doppiamente
smussata alla scala sinottica. I candidati sono gli zeri del **Thermal
Front Parameter** (TFP, Renard & Clarke 1965; Hewson 1998) — il bordo
caldo della zona baroclina — estratti come linea di livello zero del TFP
con Marching Squares (`contourpy`) e mascherati dove |∇θe| supera la
**soglia OFA standard di 4 K/100 km** (adattiva verso l'alto, percentile
85, per non tracciare troppi rami nelle scene molto barocline) e fuori
dall'orografia elevata.

## Le prove che ogni candidato deve superare

### Gate 0 — Geometria sinottica
Un fronte separa due masse d'aria: quasi-lineare o dolcemente arcuato,
lungo ≥ 150 km. **Sinuosità** (lunghezza/distanza estremi) ≤ 1.8 e
**rotazione netta** della direzione ≤ 150°: una forcina gira di ~180°
netti, un anello si richiude — sono bordi di anomalie locali (sacche
d'aria marina, cold pool convettivi), mai fronti. La rotazione è *netta*
(direzione finale vs iniziale) e non accumulata, per non punire le linee
vere ma frastagliate.

### Gate 1 — Baroclinicità reale
θe mescola temperatura e umidità: un bordo con θe forte ma **senza
gradiente di temperatura a 850 hPa** (mediana |∇T| < 2 K/100 km) è un
confine di sola umidità — brezza, aria marina notturna, outflow — il cui
contrasto vive nello strato limite sottostante. Scartato.

### Gate 1b — Coerenza termodinamica firmata (temperatura virtuale)
Un fronte è un contrasto di **densità**: "la massa d'aria più calda, e
meno densa, sale sopra quella più fredda". A pressione fissata la densità
dipende dalla **temperatura virtuale** Tv (l'aria umida è più leggera):
il lato caldo secondo θe deve risultare più leggero anche in Tv. Se i
segni si oppongono nettamente (ΔTv < −0.3 K — aria secca e calda contro
aria umida e fresca) il bordo è una **dryline** o un confine di umidità,
non una superficie frontale: nel Mediterraneo estivo è il caso tipico del
bordo tra aria sahariana (calda, secca, θe più bassa) e aria marittima
(fresca, umidissima, θe più alta), che i metodi su sola θe disegnano come
"fronte" con orientazione termica invertita. Il ΔTv firmato è esportato
(`deltaT`).

### Gate 2 — Risposta del vento
Un fronte è una discontinuità dinamica: attraversandolo il vento **ruota**
e **converge**. Salto vettoriale ≥ 2 m/s oppure convergenza ≥ 0.2 m/s
(campionati a ±45 km dalla linea); un bordo termico senza risposta del
vento è scartato.

### Gate 3 — Saccatura di pressione
Un fronte giace **dentro una saccatura**: la PMSL a ±75 km dalla linea
deve superare quella sulla linea (mediana ≥ −0.3 hPa). Una linea su un
promontorio anticiclonico è fisicamente impossibile e viene scartata.
La profondità della saccatura (`pressureTrough`) alimenta la confidenza.
Se la PMSL manca il criterio resta neutro.

### Coppia isallobarica (peso, non gate)
Il passaggio frontale ha una sequenza barica precisa: la pressione
*decresce prima, tocca il minimo durante, aumenta dopo*. Quindi, per un
fronte in moto, la **tendenza barica sul lato che il fronte sta
lasciando deve superare quella sul lato d'avanzamento**. La differenza
firmata (in hPa/3 h, esportata come `isallobaric3h`) pesa sulla
confidenza dei fronti freddi e caldi; è neutra per gli stazionari.

### Gate 4 — Vorticità ciclonica
Un fronte è una **striscia di shear ciclonico** (è il fondamento del
parametro F di Parfitt et al. 2017): vorticità relativa mediana
chiaramente anticiclonica (< −2×10⁻⁵ s⁻¹) lungo la linea è incompatibile
con un fronte. La vorticità (`vorticity1e5`) alimenta la confidenza.

### Frontogenesi (peso, non gate)
La **frontogenesi cinematica di Petterssen** su θe misura se il flusso
sta *stringendo* il gradiente (fronte attivo, in intensificazione) o
allentandolo (frontolisi). Non è un gate — anche un fronte in decadimento
è reale — ma pesa sulla confidenza ed è esportata (`frontogenesis`,
K/100 km/3 h).

### Gate 5 — Tracciamento temporale a oggetti
Un fronte non è una sequenza di decisioni orarie indipendenti: è **un
oggetto che vive nel tempo**. I candidati delle singole ore vengono
collegati in **tracce** (associazione per sovrapposizione geometrica tra
ore consecutive, raggio proporzionale al tempo trascorso) e l'accettazione
avviene a livello di traccia:

- **vita minima**: due rilevamenti su un arco di almeno 6 h (ICON-2I) /
  12 h (ECMWF);
- **copertura ≥ 50%**: un fronte vero è rilevato quasi sempre durante la
  sua vita; un bordo diurno (convezione pomeridiana, brezze) riappare a
  grappoli con lunghi vuoti notturni e viene respinto anche se ogni sua
  singola apparizione sembrava valida;
- **fiducia mediana ≥ 0.55** sull'intera vita;
- **soglie piene come mediane di traccia**: baroclinicità ≥ 2 K/100 km e
  risposta del vento (≥ 2 m/s o convergenza ≥ 0.2) sono richieste sulla
  mediana della vita del fronte, con pavimenti per-ora più bassi: un
  fronte debole ma reale non viene spezzato dalle ore in cui oscilla
  appena sotto soglia (caso verificato: il fronte freddo francese del
  run 12z 20/07, visto a tratti per-ora, ora traccia continua 16-35 h).

La pubblicazione è **continua per costruzione**: i buchi brevi (≤ una
finestra) vengono colmati interpolando la linea tra le ore adiacenti
(`interpolated: true` nel GeoJSON), e il tipo deriva dal **moto smussato
sulla vita della traccia** — un fronte non può sparire per un'ora e
ricomparire, né cambiare tipo per rumore. La vita della traccia è
esportata (`lifetimeH`).

### Gate 6 — Conferma tra modelli (solo ICON-2I)
Un fronte vero è su larga scala, quindi appare anche in un modello
indipendente e più rado. La conferma è valutata **sull'insieme della
traccia**: almeno la metà delle sue ore con guida disponibile deve
trovare il 50% della linea entro il raggio (180 km a +0 h, +1.5 km/h di
scadenza, max 320 km) da un riferimento ECMWF alla stessa validità.
Come riferimento si usano i **candidati di rilevamento** ECMWF (pre
tracciamento, campo `candidates` del catalogo): i fronti pubblicati da
ECMWF sono volutamente pochissimi, e una guida troppo rada renderebbe la
conferma quasi sempre non applicabile. Valutarla ora per ora creerebbe
sfarfallio quando la guida (passo 6 h) cambia scadenza. Fail-open se la
guida manca.

## Confidenza e classificazione

I sopravvissuti ricevono una confidenza che pesa: intensità di ∇θe,
∇T secco, salto di vento, convergenza, vorticità, frontogenesi,
saccatura, lunghezza, velocità di moto, penalità per orografia. Sotto
0.55 si scarta; al massimo 4 fronti per scadenza, deduplicati.

Il tipo segue il moto del fronte lungo la sua normale (che punta verso
l'aria calda), combinando due misure indipendenti con lo stesso segno:
la **propagazione** dell'isolinea θe (dalla tendenza temporale — il moto
reale della linea, valido anche quando il vento a 850 hPa scorre
parallelo al fronte) e l'**avvezione termica cinematica** (metodo OFA —
la componente del vento lungo la normale). Segno positivo (verso il
caldo) → **freddo** (≥ +5 km/h); negativo → **caldo** (≤ −5 km/h);
≈ nullo → **stazionario**. La propagazione è la misura principale
(l'avvezione da sola, come segnalato in revisione, classifica male i
fronti che avanzano non paralleli al flusso locale). I fronti occlusi
non sono distinti e ricadono su freddo/stazionario.

## Verifica visiva sul sito

La mappa offre i campi di ispezione **θe 850 hPa** (il campo su cui i
fronti sono individuati: i bordi netti tra i colori *sono* i contrasti
tra masse d'aria) e **T 850 hPa** (per distinguere fronti veri da confini
di sola umidità), su file `upper_*.json` caricati solo su richiesta.
Nel pannello dei fronti c'è la **tabella del passaggio** (prima/durante/
dopo per fronte freddo e caldo: temperatura, pressione, vento, nubi,
fenomeni), e toccando la mappa il dettaglio del punto riporta il **fronte
più vicino** con distanza, moto rispetto al punto (in avvicinamento o in
allontanamento) e vita della traccia.

## Validazione

Suite sintetica (vecchio vs nuovo rilevatore):

| Scenario | Vecchio | Nuovo |
|---|---|---|
| Fronte freddo vero (8 K, vento che ruota, 40 km/h) | rilevato 0.85 | rilevato 0.85 (0.88 con saccatura) |
| Sacca di umidità marina (T e vento uniformi) | falso fronte 705 km | scartato |
| Bordo su promontorio anticiclonico | — | scartato (Gate 3) |
| Artefatto presente 1 ora sola | — | scartato (Gate 5) |
| Dryline (θe da umidità, densità invertita) | — | scartato (Gate 1b) |
| Firme sul fronte vero | — | ζ = +5.8×10⁻⁵ s⁻¹, frontogenesi +2.9 K/100km/3h; con coppia isallobarica coerente la confidenza sale a 0.91 |

Dati reali (run ICON-2I 00z 20/07/2026 scaricato da MeteoHub, tutte le
73 ore analizzate con entrambe le versioni):

- il rilevatore originale produceva **sempre 4 fronti** (limite saturato)
  e alle 18 UTC disegnava fronti sulla convezione pomeridiana appenninica;
- la versione per-ora corretta produceva 0-3 fronti giusti ma con **7
  eventi di sfarfallio** in 73 ore (fronte che sparisce e ricompare);
- la versione a tracciamento mantiene gli stessi fronti (boundary alpina
  a +06h, zero fronti sui massimi convettivi pomeridiani, fronte caldo
  sull'intrusione fresca a +72h) con **1 solo evento residuo di
  discontinuità**, 9 ore colmate per interpolazione e vite delle tracce
  di 6-31 h — tempi sinottici plausibili. Il criterio di copertura ha
  inoltre respinto le pseudo-tracce diurne (bordi convettivi che
  rientravano ogni pomeriggio simulando 72 h di "vita").

## Limiti dichiarati

Stima automatica (`estimated: true`), non un'analisi manuale: i fronti
occlusi non sono etichettati come tali, i fronti in quota senza riscontro
a 850 hPa non sono rilevati, e in estate mediterranea è normale vedere
pochi o nessun fronte per giorni.

## Riferimenti

- Renard, R.J., Clarke, L.C. (1965): il Thermal Front Parameter.
- Hewson, T.D. (1998): *Objective fronts*, Met. Apps 5, 37-65 — TFP,
  mascheramenti, classificazione per velocità.
- Petterssen, S. (1936/1956): frontogenesi cinematica.
- Parfitt, R., Czaja, A., Seo, H. (2017): parametro F (vorticità ×
  gradiente) per l'identificazione dei fronti.
- Berry, G., Reeder, M.J., Jakob, C. (2011): metodo del salto di vento.
- Schemm, S., et al. (2015): metodi termici vs dinamici; sovrastima dei
  "fronti di umidità" nel Mediterraneo.
- Thomas, C.M., Schultz, D.M. (2019): scelta della variabile
  termodinamica e della funzione di localizzazione.
