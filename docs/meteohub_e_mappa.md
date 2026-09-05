# MeteoHub, ricalcolo della mappa e costi AI

La raccolta oraria e la pipeline ICON acquisiscono le osservazioni pubbliche
della rete `dpcn-sicilia` da MeteoHub, separatamente dai METAR NOAA AWC.
Non serve trasferire il login dell'utente al sito né pubblicare chiavi API.

Contratto verificato sul codice ufficiale MeteoHub (`maps_observed.py`,
`services/dballe.py`) e su un export reale JSONL del 5 settembre 2026:
POST `/api/observations`, parametri query `networks=dpcn-sicilia`,
`output_format=JSON` (maiuscolo), `reliabilityCheck=true`,
`allStationProducts=true` e `q` con `reftime`, `license:CCBY_COMPLIANT`. Le
variabili utilizzabili sono
temperatura e rugiada a 2 m, umidità relativa, direzione, intensità e raffica
del vento, pressione ridotta al livello del mare e pressione di stazione.
La richiesta riguarda solo le ultime due ore e il dominio Sicilia. È un download pubblico, non una
richiesta di estrazione persistente dell'account.

Fonte: Regione Siciliana tramite MeteoHub, Agenzia ItaliaMeteo / CINECA.
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

## Qualità e costo del meteorologo AI

La mappa non chiama gli LLM. Il bollettino viene generato centralmente e
condiviso tra i visitatori, con un autore AI, un revisore distinto e controlli
numerici e temporali. Le riscritture dei claim non validi sono limitate a una
per claim e otto complessive. I test automatici sui push non duplicano la
verifica live della pipeline; la verifica API dedicata resta avviabile a mano.

Una cache riusa solo bollettini validati con identica impronta delle evidenze,
metodo e modelli. Un errore 429 produce una pausa di un'ora nei tentativi della
pipeline. Non si attiva alcun abbonamento o modifica della fatturazione.
Se la quota del provider resta esaurita, il sito mostra esplicitamente il
bollettino deterministico di riserva: non viene presentato come bollettino AI.

Configurazione dal 5 settembre 2026: GPT-OSS 120B su Groq come meteorologo
autore e GPT-OSS 20B come revisore con modello distinto. Se Groq non autorizza
la richiesta, il percorso passa a Gemini 3.5 Flash per la redazione e Gemini
3.5 Flash-Lite per la revisione. Il file di stato pubblicato riporta sempre i
modelli realmente usati. Il prodotto resta pubblicabile soltanto dopo i
controlli deterministici su riferimenti, numeri e finestre temporali. Prezzi
ufficiali verificati su https://ai.google.dev/gemini-api/docs/pricing e
https://console.groq.com/docs/models; nessuna stima mensile è affidabile senza
misurare token e frequenza reali.
