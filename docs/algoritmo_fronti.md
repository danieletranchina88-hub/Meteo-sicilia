# Rilevamento oggettivo dei fronti — fisica e algoritmo

Questo documento spiega la fisica su cui si basa il rilevatore di fronti
(`scripts/front_analysis.py`) e il perché di ogni criterio. Metodo:
`theta-e-850-multievidence-v5` (ICON-2I: `theta-e-850-icon2i-multievidence-v7`).

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
caldo della zona baroclina — mascherati dove |∇θe| supera una soglia
adattiva (percentile 88 della scena, minimo 6 K/100 km) e fuori
dall'orografia elevata.

## Le prove che ogni candidato deve superare

### Gate 0 — Geometria sinottica
Un fronte separa due masse d'aria: quasi-lineare o dolcemente arcuato,
lungo ≥ 200 km. **Sinuosità** (lunghezza/distanza estremi) ≤ 1.8 e
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

### Gate 5 — Coerenza temporale
Un fronte reale **persiste per molte ore e trasla con continuità**; gli
artefatti (brezze diurne, outflow, rumore di griglia) compaiono e
scompaiono. Ogni candidato deve ritrovare almeno il 40% della propria
linea — entro un raggio di 50 km per ora di finestra (150 km a ±3 h per
ICON-2I, 300 km a ±6 h per ECMWF) — tra i candidati di almeno una delle
ore adiacenti dello stesso run. I rilevamenti orari sono in cache, quindi
il costo aggiuntivo è nullo.

### Gate 6 — Conferma tra modelli (solo ICON-2I)
Un fronte vero è su larga scala, quindi appare anche in un modello
indipendente e più rado. Almeno il **50% della linea** ICON-2I deve
trovarsi entro un raggio (180 km a +0 h, +1.5 km/h di scadenza, max
320 km) da un fronte della corsa ECMWF alla stessa validità. La frazione
di linea è essenziale: la vicinanza di un solo punto farebbe "confermare"
un artefatto da un fronte vero che gli passa accanto. Fail-open se la
guida manca.

## Confidenza e classificazione

I sopravvissuti ricevono una confidenza che pesa: intensità di ∇θe,
∇T secco, salto di vento, convergenza, vorticità, frontogenesi,
saccatura, lunghezza, velocità di moto, penalità per orografia. Sotto
0.55 si scarta; al massimo 4 fronti per scadenza, deduplicati.

Il tipo segue il moto lungo la normale (tendenza di θe combinata col
vento normale): ≥ +5 km/h verso l'aria calda → **freddo**; ≤ −5 km/h →
**caldo**; altrimenti **stazionario** (soglia ≈ ±1.5 m/s di Hewson).
I fronti occlusi non sono distinti e ricadono su freddo/stazionario.

## Verifica visiva sul sito

La mappa offre i campi di ispezione **θe 850 hPa** (il campo su cui i
fronti sono individuati: i bordi netti tra i colori *sono* i contrasti
tra masse d'aria) e **T 850 hPa** (per distinguere fronti veri da confini
di sola umidità), su file `upper_*.json` caricati solo su richiesta.

## Validazione

Suite sintetica (vecchio vs nuovo rilevatore):

| Scenario | Vecchio | Nuovo |
|---|---|---|
| Fronte freddo vero (8 K, vento che ruota, 40 km/h) | rilevato 0.85 | rilevato 0.85 (0.88 con saccatura) |
| Sacca di umidità marina (T e vento uniformi) | falso fronte 705 km | scartato |
| Bordo su promontorio anticiclonico | — | scartato (Gate 3) |
| Artefatto presente 1 ora sola | — | scartato (Gate 5) |
| Firme sul fronte vero | — | ζ = +5.8×10⁻⁵ s⁻¹, frontogenesi +2.9 K/100km/3h |

Dati reali (run ICON-2I 00z 20/07/2026 scaricato da MeteoHub, 8 scadenze
ispezionate su mappa):

- il vecchio rilevatore produceva **sempre 4 fronti** (limite saturato) —
  a luglio, sovra-rilevamento evidente; alle 18 UTC disegnava fronti
  sulle strisce di θe da convezione pomeridiana lungo l'Appennino;
- il nuovo ne produce 0-2, tutti su bordi di massa d'aria visibili in θe
  e dentro saccature: la boundary quasi-stazionaria lungo le Alpi/
  Dinariche a +06h, nessun fronte sui massimi convettivi pomeridiani a
  +18h/+36h, e a +72h il fronte caldo sul bordo dell'intrusione fresca
  adriatica. Regressione confermata anche sul falso "ferro di cavallo"
  adriatico del 20/07 (sinuosità 3.03 → scartato).

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
