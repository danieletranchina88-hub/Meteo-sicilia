# MeteoHub, ricalcolo della mappa e costi AI

La raccolta oraria e la pipeline ICON acquisiscono le osservazioni pubbliche di
MeteoHub su **tutto il dominio del modello**, separatamente dai METAR NOAA AWC.
Non serve trasferire il login dell'utente al sito né pubblicare chiavi API.

MeteoHub aggrega le reti regionali di tutta Italia. Chiedendo la sola rete
`dpcn-sicilia` su un riquadro siciliano, com'era prima, rispondevano 421
stazioni; chiedendo tutte le reti sul dominio ICON-2I ne rispondono 3459 su 27
reti. Il modello copre tutta l'Italia, e un'analisi che corregge solo la Sicilia
lascia scoperto il resto della carta.

Contratto verificato sul codice ufficiale MeteoHub (`maps_observed.py`,
`services/dballe.py`) e su un export reale JSONL: POST `/api/observations`,
`output_format=JSON` (maiuscolo), `reliabilityCheck=true`, nessun filtro
`networks`, riquadro 3–22 °E / 33,7–48,9 °N, e `q` con `reftime`,
`license:CCBY_COMPLIANT` e la lista dei prodotti separati da `or`. Le variabili
sono temperatura e rugiada a 2 m, umidità relativa, direzione, intensità e
raffica del vento, pressione ridotta al livello del mare e pressione di
stazione. Poiché il feed generale non include sempre la pressione mostrata dalla
mappa ufficiale, due richieste mirate a `B10004` e `B10051` vengono fuse al
risultato.

Due scelte tengono il download ragionevole, entrambe misurate sul dominio
intero:

- si chiedono i prodotti che servono invece di `allStationProducts=true`, che
  portava anche pioggia, radiazione e suolo: 18 MB invece di 78;
- la finestra è di **75 minuti** invece di due ore. Serve solo l'ultimo dato di
  ogni stazione, e 75 minuti ne raccolgono 3606 contro le 3669 di due ore,
  dimezzando i byte. A 45 minuti se ne perderebbe il 42%.

Una richiesta ogni ora, circa 10 secondi. È un download pubblico, non una
richiesta di estrazione persistente dell'account.

**Livello della pressione di stazione.** Le reti dichiarano `B10004` a livelli
diversi: la Sicilia come quota in millimetri (tipo 102), molte altre come
"superficie del suolo" (tipo 1) con valore nullo. Pretendere lo zero, com'era
prima, scartava la pressione di 928 record su 27 reti — un difetto invisibile
finché si guardava una regione sola.

Fonte: reti regionali italiane tramite MeteoHub, Agenzia ItaliaMeteo / CINECA.
Portale e attribuzioni: https://meteohub.agenziaitaliameteo.it/app/license
Codice del servizio: https://gitlab.hpc.cineca.it/mistral/meteo-hub
Guida: https://meteohub.agenziaitaliameteo.it/ui/user-guide

Il parser riunisce i record per stazione senza perdere i prodotti acquisiti a
orari differenti e pubblica anche l'orario di ogni singolo campo. Accetta solo
livelli e intervalli temporali compatibili con ciascuna
variabile, converte kelvin in °C, m/s in km/h e pascal in hPa, mantiene
l'orario di ciascuna stazione, scarta tempi futuri e dati più vecchi di due
ore e conserva l'ultima misura per coordinate. La pressione di stazione resta
separata e non viene mai usata come MSLP. Quote zero o mancanti restano nulle:
la stazione è visibile ma non entra nella correzione termica.
Per la pressione sono ammessi il livello medio del mare DB-All.e 101 per
`B10051` e i livelli di superficie/quota/altezza 1, 102 e 103 per `B10004`.
Il catalogo MeteoHub visualizzato è il sottoinsieme delle stazioni ricevute,
non l'intera rete. I dati restano provvisori; il filtro del fornitore non è
una validazione climatologica. `sourceStatus` registra ogni fonte anche
quando l'altra non risponde. Una fonte indisponibile non cancella l'altra.

## Pulsante Ricalcola mappa · Downscaling

È un'analisi oggettiva sperimentale sulla griglia esistente, non una nuova
integrazione dinamica a risoluzione più fine. Usa osservazioni entro ±45
minuti dalla scadenza della mappa e comunque non più vecchie di due ore.
Le scadenze future senza osservazioni compatibili rimangono ICON.

Temperatura: raggio 40 km, correzione massima 3 °C, quota del modello
necessaria; normalizzazione con gradiente standard assunto −6,5 K/km.
Pressione: raggio 150 km e correzione massima 3 hPa. La vera SLP viene
confrontata direttamente con MSLP; la pressione di stazione resta distinta e
viene confrontata con ICON ridotto idrostaticamente alla quota della stazione.
In questo modo lo scarto corregge MSLP senza fingere che pressione locale e
pressione al livello del mare siano la stessa variabile. QNH/altimeter METAR
non viene usato. Vento: raggio 40 km, componenti U/V in m/s,
correzione massima 4 m/s per componente. Umidità relativa: raggio 40 km,
massimo 20 punti percentuali e risultato limitato a 0–100%.

Ogni cella richiede almeno tre stazioni non collocate entro 1 km,
scarti limitati e dispersione contenuta. Un peso assegnato al modello e
pesi spaziali decrescenti impediscono di estendere integralmente uno scarto
fino al bordo del raggio. Fuori dalle zone supportate resta il modello.
Il pulsante riporta la percentuale di celle corrette. Raster, letture
puntuali, pressione e vettori al suolo usano i campi ricalcolati.

Queste soglie sono conservative ma non calibrate sullo storico: non si
dichiara un miglioramento previsionale dimostrato. Pioggia, temporali,
fronti e livelli isobarici non sono modificati da questa analisi superficiale.
La stima puntuale di temperatura fino a sei ore resta un prodotto distinto,
con il proprio controllo spaziale e attenuazione temporale.
