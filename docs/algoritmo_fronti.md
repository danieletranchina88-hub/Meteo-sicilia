# Analisi oggettiva dei fronti ICON-2I (v14)

## Scopo e limite fondamentale

Il sito pubblica una **stima oggettiva e conservativa** dei fronti sinottici
nel dominio ICON-2I. La geometria proviene esclusivamente da ICON-2I; ECMWF
resta un modello di superficie separato e non conferma, modifica o sostituisce
i fronti.

La linea non è un'osservazione né un'analisi manuale del Servizio
Meteorologico. `qualityScore` e `uncertaintyIndex` descrivono la coerenza
interna delle prove in un singolo run deterministico: non sono probabilità
calibrate e non misurano l'errore previsionale assoluto.

Il metodo operativo è `icon2i-ofa-physics-guided-v14`.

## Dati usati

Per ogni run 00/12 UTC e per tutte le 73 scadenze orarie:

- T, QV, U e V a 850 hPa: campi obbligatori e geometria primaria;
- PMSL: firma di saccatura e tendenza barica, opzionale;
- T, QV, U e V a 925 hPa: coerenza verticale e vento più vicino al suolo,
  opzionali;
- OMEGA a 700 hPa: attività/ascesa frontale, opzionale;
- HSURF: controllo orografico.

## Livelli disponibili nella mappa

La tendina della mappa ICON-2I consente di vedere **temperatura, vento e
umidità relativa** su tre quote reali: suolo (T 2 m, vento 10 m e RH
superficiale), 925 hPa e 850 hPa. Per 925/850 hPa il sito usa direttamente
T, QV, U e V dei file MeteoHub; l'umidità relativa è ricavata da QV, pressione
del livello e pressione di saturazione di Bolton. I livelli in quota vengono
esportati su una griglia di ispezione alleggerita (~19 km), sufficiente per la
lettura sinottica e molto più leggera sul cellulare. Se un run non contiene
un livello opzionale, l'interfaccia non lo sostituisce né lo interpola: torna
al suolo e avvisa l'utente. A 925 hPa le celle sopra 650 m di orografia sono
mascherate: a quella quota la superficie di pressione può trovarsi dentro il
rilievo, quindi un valore visualizzato sarebbe fisicamente fuorviante.

Il livello 850 hPa riduce il rumore dello strato limite e i contrasti diurni.
Il livello 925 hPa interseca il terreno già attorno a 750 m: sopra 650 m i
campioni vengono esclusi. Dove almeno il 60% del confronto resta valido, la
coerenza termica a 925 hPa diventa una porta obbligatoria; altrove non viene
inventato un dato sostitutivo. Omega resta una prova morbida: un fronte maturo
o frontolitico può esistere senza forte ascesa istantanea.

La fonte ufficiale descrive ICON-2I come modello deterministico a circa
2,2 km, dominio 3–22°E / 33–49°N, orizzonte +72 h:
[MeteoHub — dataset ICON-2I](https://meteohub.agenziaitaliameteo.it/app/datasets).

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

1. smoothing gaussiano in chilometri, prima delle derivate;
2. gradiente metrico di `theta_w`;
3. Thermal Front Locator
   `TFL = laplacian(|grad(theta_w)|)`;
4. estrazione delle sole isolinee `TFL = 0`;
5. mascheramento della linea con TFP e intensità della zona baroclina
   adiacente (ABZ).

Il Thermal Front Parameter conserva il segno standard:

```text
TFP = grad(|grad(theta_w)|) · grad(theta_w) / |grad(theta_w)|
```

Sul bordo caldo della zona baroclina è negativo. L'ABZ è la stima locale
pubblicata, non un campione arbitrario preso lontano dalla linea:

```text
ABZ = |grad(theta_w)|
      + grid_length / sqrt(2) * |grad(|grad(theta_w)|)|
```

Le derivate rispettano le distanze reali della griglia lon/lat; cambiare
risoluzione non cambia implicitamente le unità delle soglie.

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

Soglie fuzzy OFA, espresse in unità fisiche:

| Scala | TFP debole → pieno | ABZ debole → pieno |
|---|---:|---:|
| Sinottica | −0,15 → −0,40 K/(100 km)² | 0,65 → 1,10 K/100 km |
| Raffinata | −0,25 → −0,90 K/(100 km)² | 0,90 → 1,70 K/100 km |

I quantili del dominio possono rendere le soglie più severe, mai più
permissive. Vengono inoltre respinte anomalie quasi chiuse, hairpin molto
sinuosi, duplicati paralleli e linee prevalentemente sopra terreno elevato.

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

I requisiti duri impongono un vero contrasto secco e di densità, persistenza
del contrasto alle tre distanze, allineamento termico plausibile e geometria
sinottica.
### Motore di ragionamento (diagnosi differenziale)

L'algoritmo ragiona come un meteorologo, non come una catena di soglie:
**individua un possibile fronte** (TFL/TFP/ABZ), **raccoglie tutte le prove
disponibili** (contrasto di θw, temperatura secca, densità virtuale, gradiente
secco, corridoio sinottico, lunghezza, geometria, vento, convergenza,
vorticità, frontogenesi, saccatura, 925 hPa, omega 700 hPa), **osserva
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

Soltanto dopo queste porte il punteggio di evidenza ordina le linee ammesse:

- 38% termodinamica;
- 24% dinamica;
- 10% pressione;
- 10% coerenza verticale;
- 4% attività a 700 hPa;
- 14% struttura.

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

## 6. Significato dell'incertezza

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

## 7. Test automatici

La workflow blocca la pubblicazione se falliscono i test sintetici:

- segno TFP e posizione sul bordo caldo;
- invarianza a orientamento della griglia e risoluzione;
- rimozione di patch locali e anomalie chiuse;
- traslazione rigida senza falsa frontogenesi;
- convergenza con frontogenesi positiva;
- rotazione rigida con vorticità ma senza falsa deformazione;
- rifiuto dei confini di sola umidità;
- rifiuto di gradiente termico con vento divergente/contrario;
- promontorio barico che indebolisce una linea ma non cancella un fronte
  termicamente coerente già tracciato;
- tracciamento, classificazione e separazione delle identità.
- consenso obbligatorio fra moto geometrico, fase termica e vento;
- continuità di una traccia lunga a prevalenza "continuazione" e persistenza
  visibile di un fronte quasi stazionario passato ogni porta fisica;
- riconoscimento dell'onda in occlusione e rifiuto di onda aperta / fronte
  freddo isolato come falsa occlusione;
- presenza e buona forma dell'array `explanation` su ogni fronte pubblicato.

## Riferimenti primari

- Hewson, 1998, *Objective fronts*:
  [Meteorological Applications](https://www.cambridge.org/core/journals/meteorological-applications/article/objective-fronts/DEF4E8845B0B5DBC102560C658FB4B6B)
- Sansom e Catto, 2024, *A portable objective front identification method*:
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
