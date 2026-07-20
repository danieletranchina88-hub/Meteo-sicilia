# Rilevamento oggettivo dei fronti — fisica e algoritmo

Questo documento spiega la fisica su cui si basa il rilevatore di fronti
(`scripts/front_analysis.py`) e il perché di ogni filtro. Metodo:
`theta-e-850-tfp-wind-v3-physgate`.

## Cos'è un fronte (e cosa non lo è)

Un fronte sinottico è il bordo della zona di transizione tra **due masse
d'aria di origine diversa** (es. aria polare vs subtropicale). Non è una
semplice linea di contrasto termico: è una struttura dinamica con firme
precise e simultanee.

Il rilevamento parte dal metodo oggettivo classico (Hewson 1998, usato in
varie forme dai servizi meteo europei): si lavora a **850 hPa** (~1.5 km,
sopra il rumore dello strato limite) sulla **temperatura potenziale
equivalente θe** (che combina temperatura e umidità), e si cercano i punti
in cui il gradiente di θe "parte" — lo zero del **Thermal Front Parameter**
(TFP), mascherato dove l'intensità del gradiente supera una soglia.

Il TFP da solo però trova *qualunque* bordo di gradiente. Un fronte vero ha
**quattro firme contemporanee**, e ognuna è un filtro dell'algoritmo:

### 1. Geometria sinottica (Gate 0)
Un fronte separa due masse d'aria: è quasi rettilineo o dolcemente arcuato,
lungo centinaia di km. Una linea che si ripiega su sé stessa (forcina) o
quasi si richiude (anello) è per costruzione il bordo di un'anomalia locale
— una sacca d'aria calda marina, un cold pool convettivo — mai un fronte.

Criteri: **sinuosità** (lunghezza / distanza tra gli estremi) ≤ 1.8 e
**rotazione netta** della direzione (heading finale vs iniziale) ≤ 150°.
Si usa la rotazione *netta* e non quella accumulata: una linea lunga e
frastagliata ma dritta accumula centinaia di gradi di piccole svolte senza
mai ripiegarsi, mentre una forcina gira di ~180° netti comunque la si
campioni. Lunghezza minima: 200 km.

### 2. Contrasto termico reale in quota (Gate 1)
θe mescola temperatura e umidità: un bordo visibile in θe ma **privo di
gradiente di temperatura a 850 hPa** è un confine di sola umidità — brezza,
aria marina notturna, outflow temporalesco — il cui contrasto vive tutto
nello strato limite sottostante. Un fronte sinottico ha sempre
baroclinicità anche a 850 hPa.

Criterio: mediana di |∇T₈₅₀| lungo la linea ≥ 2 K/100 km.

Questo è il filtro che elimina la classe di falsi più insidiosa per il
Mediterraneo: i bordi termici terra-mare all'alba (es. l'anello di aria
umida sull'Adriatico), dove la temperatura a 850 hPa è uniforme.

### 3. Firma dinamica (Gate 2)
Un fronte giace in una saccatura di pressione: attraversandolo il vento
**ruota** (ciclonicamente) e **converge**. Un bordo termico senza alcuna
risposta del vento non è un fronte.

Criterio: salto del vento attraverso la linea ≥ 2 m/s **oppure**
convergenza ≥ 0.2 m/s (misurati ±45 km dalla linea, mediane).

### 4. Moto coerente (classificazione)
Il fronte si muove con la componente del vento perpendicolare a sé:
il tipo deriva dalla velocità di propagazione (stimata dalla tendenza
temporale di θe combinata col vento normale): ≥ +5 km/h verso l'aria
calda → **freddo**; ≤ −5 km/h → **caldo**; altrimenti **stazionario**
(soglia coerente col ±1.5 m/s di Hewson).

## Conferma incrociata tra modelli (ICON-2I + ECMWF)

ICON-2I a 2.2 km risolve strutture di mesoscala che un rilevatore sinottico
non deve mostrare. Un fronte vero è su larga scala, quindi appare — smussato
e spostato al più di poche decine di km — anche in un modello indipendente e
più rado (ECMWF IFS 0.25°). Ogni fronte ICON-2I deve avere **almeno il 50%
della linea** entro un raggio (180 km a +0 h, +1.5 km/h di scadenza, max
320 km) da un fronte ECMWF della stessa validità. La frazione di linea è
essenziale: la vicinanza di un solo punto farebbe "confermare" un artefatto
da un fronte vero che gli passa accanto. Se la guida ECMWF non è
disponibile, il filtro non blocca (fail-open).

## Punteggio di fiducia

I fronti sopravvissuti ai gate ricevono una confidenza (0-1) che pesa:
intensità del gradiente θe, gradiente termico secco, salto di vento,
convergenza, lunghezza, velocità di moto, penalità per terreno elevato.
Sotto 0.55 il candidato è scartato; sulla mappa la trasparenza riflette la
confidenza. Massimo 4 fronti per scadenza, deduplicati per centroide.

## Limiti dichiarati

È una **stima automatica** (`estimated: true` nei metadati), non un'analisi
manuale da meteorologo. I fronti occlusi non sono distinti (ricadono in
freddo/stazionario); i fronti in quota senza riscontro a 850 hPa non sono
rilevati; in estate mediterranea è normale vedere pochi o nessun fronte per
giorni.

## Validazione

- Scenario sintetico "fronte freddo vero" (contrasto 8 K, rotazione del
  vento, avanzamento 40 km/h): rilevato, tipo corretto, confidenza 0.84.
- Scenario sintetico "sacca di umidità marina" (θe forte, T uniforme, vento
  uniforme, stazionaria): l'algoritmo precedente disegnava un falso fronte
  da 705 km; quello attuale la scarta.
- Regressione sul caso reale del 20/07/2026 05 UTC (falso fronte a ferro di
  cavallo sull'Adriatico, run ICON-2I 12z del 19/07): sinuosità 3.03 e
  rotazione netta 175° → scartato da due criteri indipendenti; i due fronti
  veri contemporanei sui Balcani (sinuosità 1.20/1.32) restano.

## Riferimenti

- Hewson, T.D. (1998): *Objective fronts*, Met. Apps 5, 37-65 — TFP,
  mascheramenti, classificazione per velocità.
- Berry, G., Reeder, M.J., Jakob, C. (2011): climatologia globale dei
  fronti con metodo del salto di vento.
- Schemm, S., et al. (2015): confronto metodi termici vs dinamici;
  i metodi termici puri sovrastimano i "fronti di umidità" nel
  Mediterraneo — motivazione diretta del Gate 1.
