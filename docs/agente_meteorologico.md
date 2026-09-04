# Agente meteorologico verificato

## Scopo

L'agente aggiunge una sintesi ragionata al bollettino deterministico ICON-2I.
Non sostituisce il motore numerico, non modifica griglie, fronti, probabilità o
downscaling e non costituisce un sistema di assimilazione dati. Il prodotto è
pubblicato soltanto quando supera tre controlli indipendenti:

1. schema strutturato di Gemini 3.8 Flash;
2. validazione deterministica locale;
3. revisione dei claim con GPT-OSS 120B su Groq.

Se un controllo fallisce, il sito utilizza automaticamente
`expert_bulletin.json.gz`, prodotto dal motore fisico esistente.

## Flusso dei dati

`scripts/process_data.py` genera tutte le 73 scadenze e il bollettino
deterministico. `scripts/generate_ai_bulletin.py` seleziona intervalli di sei
ore per costruire le prove e li sintetizza in periodi previsionali di circa
dodici ore. I GRIB e le matrici complete non vengono inviati agli LLM.

Il catalogo trasmesso contiene soltanto:

- modello, run, valid time, risoluzione e orizzonte;
- paragrafi diagnostici già derivati da campi reali;
- disponibilità e assenza esplicita delle grandezze;
- catene causali e limiti del motore deterministico;
- identificatore, forecast hour e famiglia fisica di ogni prova.

## Regole di pubblicazione

Ogni claim deve:

- citare da due a cinque prove esistenti;
- usare almeno due famiglie diagnostiche differenti;
- utilizzare solo prove appartenenti al periodo descritto;
- copiare eventuali numeri esattamente dalle prove citate;
- impiegare una confidence qualitativa prevista dal vocabolario del sito;
- restare entro 420 caratteri e non contenere markup eseguibile.

Il validatore respinge identificatori inesistenti, numeri non documentati,
sezioni mancanti o duplicate, periodi alterati e claim fondati su una sola
famiglia. GPT-OSS rilegge separatamente sintesi generale e ciascun periodo per
rimanere entro la quota gratuita; può chiedere un downgrade della confidence
oppure bloccare l'intero prodotto.

## Output e tracciabilità

Quando tutti i controlli passano viene scritto
`data_weather/ai_expert_bulletin.json.gz`, contenente:

- metodi e modelli utilizzati;
- conteggio dei token restituito dai provider;
- sintesi generale e periodi previsionali;
- claim, confidence ed evidenceIds;
- catalogo delle sole prove effettivamente citate;
- esito dei controlli deterministici e della revisione.

`data_weather/ai_agent_status.json` viene scritto sempre. In caso di fallback
riporta una motivazione tecnica sanitizzata senza contenere chiavi API. Il
manifest del run viene poi rigenerato affinché anche questi file ricevano
dimensione, ruolo e checksum SHA-256.

## Segreti e costi

Le credenziali `GEMINI_API_KEY` e `GROQ_API_KEY` sono lette esclusivamente dal
runner GitHub Actions. Non entrano nel repository, nel branch `gh-pages`, nei
prompt pubblicati o nei file di stato. Il frontend scarica soltanto il JSON già
validato.

L'assenza delle chiavi, il superamento di una quota o un errore dei provider
non interrompono l'elaborazione ICON-2I e non cancellano il bollettino
deterministico.

## Limiti scientifici

L'agente interpreta un singolo run deterministico. Non apprende dai run
precedenti, non calibra probabilità e non misura la dispersione di un ensemble.
Radar, satellite e stazioni potranno essere aggiunti al catalogo soltanto con
provenienza, allineamento temporale e controlli di qualità espliciti. Fino ad
allora l'agente deve descrivere esclusivamente l'evidenza prognostica ICON-2I.
