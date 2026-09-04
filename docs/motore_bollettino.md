# Motore del bollettino meteorologico esperto

## Scopo e natura del prodotto

Il bollettino tecnico è una diagnosi automatica del singolo run deterministico
ICON-2I. Non è un'allerta, non è un'analisi osservativa e non misura la
dispersione di un ensemble. Il metodo pubblicato nel JSON è
`icon2i-multifield-synoptic-engine-v1`.

Il motore non deduce un fenomeno da un solo indice. Prima riduce i campi
grigliati in prove regionali verificabili, poi cerca la concordanza fra
termodinamica, dinamica, umidità, sollevamento, vento e sequenza temporale. Un
campo mancante resta mancante: non viene sostituito con climatologia, soglie
stagionali o valori costanti.

## Identificazione del dato

Ogni prodotto dichiara:

- modello e run UTC;
- valid time e forecast hour;
- risoluzione nominale del modello (circa 2,2 km);
- passo temporale (1 ora);
- dominio analizzato;
- orizzonte disponibile;
- natura qualitativa della confidence.

La confidence esprime soltanto completezza e concordanza interna delle prove
nel run. Non è una percentuale di accuratezza e non sostituisce una verifica
contro osservazioni o analisi manuali.

## Campi utilizzati

| Livello | Campi e diagnostiche |
|---|---|
| Superficie | MSLP, T2m, UR2m, vento e raffica 10 m, convergenza 10 m, copertura nuvolosa, precipitazione oraria |
| 925 hPa | temperatura, umidità relativa derivata da T e q, theta-e, vento; valori sottoterra mascherati |
| 850 hPa | temperatura, theta-e, theta-w, gradiente di theta-e, advezione di theta-e, umidità e vento |
| 700 hPa | temperatura, umidità, vento e omega |
| 500 hPa | geopotenziale, temperatura, vento, vorticità relativa, avvezione di vorticità e omega |
| 300 hPa | vento e divergenza orizzontale filtrata su scala sinottica |
| Colonna/convezione | ML-CAPE, ML-CIN, MU-CAPE, PWAT/TQV, WSHEAR 0–6 km, LPI e UH_MAX |
| Pericoli condizionati | indici grandine e downburst, zero termico, gelicidio, nebbia/visibilità e foehn |
| Innesco locale/orografia | risalita sul terreno, convergenza di brezza, rapporto di Bowen e indice d'innesco |
| Fronti | linee OFA multilivello, tipo, supporto d'esistenza, incertezza, frontogenesi, convergenza e struttura 925–700 hPa |

Le derivate orizzontali sono metriche: tengono conto della convergenza dei
meridiani. Prima delle derivate del vento vengono applicati filtri fisici
espressi in chilometri, non in numero fisso di celle. La PVA è presentata come
diagnostica di avvezione su tre ore; non viene trasformata in un valore
assoluto di sollevamento.

Il gradiente termico 700–500 hPa usa uno spessore medio diagnostico di 3 km.
È utile per confrontare la stabilità regionale, ma non sostituisce il calcolo
su altezze geopotenziali complete della colonna.

## Catene fisiche

### Sinottica e fronti

Un minimo di MSLP non viene chiamato automaticamente ciclone attivo. La
diagnosi considera contemporaneamente gradiente barico, vento, fronti,
geopotenziale a 500 hPa, PVA, omega e divergenza in quota. I fronti sono
accettati dal prodotto frontale dedicato soltanto dopo i controlli
multilivello e temporali descritti in [algoritmo_fronti.md](algoritmo_fronti.md).

### Convezione

Il motore scompone il quadro in:

1. instabilità;
2. umidità;
3. innesco;
4. forcing dinamico;
5. shear e organizzazione;
6. inibizione;
7. contraddizioni fra le prove.

ML-CAPE elevata non basta. Per classificare il segnale come sufficientemente
corroborato servono anche CIN compatibile con l'innesco, umidità e almeno due
forzanti concordi fra convergenza, moto ascendente e fronte. L'indice
temporalesco proprietario resta uno **score deterministico non calibrato**.

`UH_MAX` è updraft helicity della cella modellata, non storm-relative helicity.
LPI, UH_MAX e gli indici di grandine/downburst sono quindi diagnostiche del
singolo run e non osservazioni né probabilità.

### Precipitazioni

La precipitazione del passo viene trasformata in accumuli mobili di 3, 6, 12
e 24 ore. Il massimo modellistico non viene tradotto automaticamente in rischio
idrogeologico: mancano esposizione, vulnerabilità, stato del suolo e una
valutazione idrologica dedicata.

### Evoluzione

Ogni scadenza è letta nel run intero. Il bollettino confronta gli orizzonti
+6, +12, +24, +48 e +72 ore, quando esistono, riportando evoluzione del minimo
barico, numero di fronti e segnale convettivo. Se l'orizzonte cade oltre la
fine del run viene dichiarato non disponibile.

## Diagnostiche non disponibili

La versione corrente dichiara esplicitamente come non disponibili:

- PV, superfici isentropiche e tropopausa dinamica 2 PVU;
- SRH 0–1/0–3 km e hodograph completo;
- DCAPE;
- effective bulk shear e storm-relative flow;
- separazione della precipitazione convettiva e stratiforme;
- profilo verticale completo per neve, wet-bulb zero e warm nose;
- altezza PBL e contenuto di acqua di nube.

Queste grandezze non vengono ricostruite con proxy deboli. Potranno entrare
solo quando il dataset operativo fornirà profili o campi nativi sufficienti.

## Output verificabile

`data_weather/expert_bulletin.json.gz` contiene:

- metadati e semantica del prodotto;
- un'analisi per ogni forecast hour;
- sezioni tecniche e sintesi operativa;
- componenti convettive e fattori limitanti;
- catene causali “Perché?”;
- fenomeni significativi con area, periodo, motivazione e confidence;
- disponibilità dei campi e statistiche regionali usate come evidenza.

L'interfaccia seleziona automaticamente l'analisi corrispondente all'ora della
timeline. Il tab “Punto selezionato” rimane un prodotto distinto, costruito
sulla serie locale di 72 ore.

Quando i due segreti API sono disponibili, il workflow produce anche
`ai_expert_bulletin.json.gz`. È una sintesi sovrapposta e non una sostituzione:
ogni claim deve citare prove del presente prodotto, superare il controllo
numerico locale ed essere approvato da un revisore indipendente. Il dettaglio
del protocollo e del fallback è documentato in
[agente_meteorologico.md](agente_meteorologico.md).

## Riferimenti essenziali

- Bolton, 1980: *The Computation of Equivalent Potential Temperature*,
  Monthly Weather Review, DOI
  [10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1980)108%3C1046:TCOEPT%3E2.0.CO;2).
- Davies-Jones, 2008: *An Efficient and Accurate Method for Computing the Wet-Bulb Temperature along Pseudoadiabats*, DOI
  [10.1175/2007MWR2224.1](https://doi.org/10.1175/2007MWR2224.1).
- Hewson, 1998: *Objective fronts*, Meteorological Applications, DOI
  [10.1017/S1350482798000553](https://doi.org/10.1017/S1350482798000553).
- Sansom e Catto, 2024: confronto e implementazione riproducibile dei metodi
  oggettivi di identificazione frontale,
  [Geoscientific Model Development](https://gmd.copernicus.org/articles/17/6137/2024/).
- Doswell, Brooks e Maddox, 1996: approccio ingredients-based ai fenomeni
  intensi e alle alluvioni improvvise, DOI
  [10.1175/1520-0434(1996)011<0560:FFFAIB>2.0.CO;2](https://doi.org/10.1175/1520-0434(1996)011%3C0560:FFFAIB%3E2.0.CO;2).
- [Agenzia ItaliaMeteo – MeteoHub](https://meteohub.agenziaitaliameteo.it/app/datasets): fonte operativa dei campi ICON-2I.

L'architettura di verifica e la distinzione fra diagnosi del run e correzione
statistica addestrata sono documentate in
[archivio_verifica_ia.md](archivio_verifica_ia.md).
