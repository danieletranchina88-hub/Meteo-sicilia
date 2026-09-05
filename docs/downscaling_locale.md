# Downscaling locale con osservazioni e strumenti fisici

## Cosa viene pubblicato

La scheda **AI Downscaling · temperatura locale**, dentro il punto selezionato
sulla mappa, confronta ICON originale con una stima locale sperimentale. La
quota locale arriva dal DEM della vista 3D o da un valore inserito dall'utente.
La quota ICON è campionata dalla griglia HSURF dello stesso run.

Il coordinamento è una pipeline di strumenti deterministici, senza pesi neurali.
L'agente LLM dei bollettini riceve esempi METAR–ICON controllati e contemporanei
come prove distinte dalle previsioni. Il LLM non produce né decide liberamente
correzioni numeriche; non può cambiare i limiti di questo strumento.

Questa versione riguarda esclusivamente la temperatura puntuale. Non crea una
nuova simulazione atmosferica, non modifica le griglie originali, pioggia, vento,
fronti o meteogrammi. La U-Net e il downscaler neurale restano progetti separati.

## Metodo e limiti

1. Campionamento bilineare sulla griglia ICON nativa alle stazioni registrate.
2. Interpolazione temporale alla misura, senza estrapolazione e solo tra passi
   distanti al massimo 90 minuti. Misure future, vecchie oltre due ore, mancanti,
   coordinate discordanti e scarti oltre 5 °C sono esclusi.
3. Gradiente **assunto** −0,0065 K/m per normalizzare le quote, limitato a
   dislivelli di 300 m. Non è un gradiente misurato e non risolve inversioni,
   ristagno d'aria fredda, brezze o strati limite disaccoppiati.
4. Scarto normalizzato: osservazione − (ICON + gradiente × dislivello).
5. Combinazione locale di almeno tre stazioni entro 40 km e con quota entro
   250 m dal punto. Il peso è `(1 − (d/R)²)² × exp(−(Δz/150)²)`.
   Un peso unitario assegnato allo scarto nullo del modello attenua le correzioni
   quando il supporto è debole. Dispersione superiore a 2 °C blocca il contributo.
6. Contributo osservato attenuato secondo `exp(−h/2) × (1−h/6)` per 0 ≤ h ≤ 6,
   con h calcolato dalla singola osservazione alla validità richiesta. Nessuna
   osservazione successiva alla validità viene usata. Nessuna persistenza a 72 h.
7. La correzione complessiva osservazioni + quota non può superare 3 °C.

Questi limiti sono ipotesi prudenziali di progetto, **non parametri calibrati**.
La distanza orizzontale e la quota non garantiscono la stessa massa d'aria:
barriere orografiche, costa e fronti possono limitare ulteriormente il metodo.

## Controllo e astensione

Il backend esclude ogni stazione dalla previsione usata per valutarla
(leave-one-station-out). Con almeno cinque punti valutabili, il MAE degli scarti
normalizzati dopo correzione deve essere inferiore al MAE della medesima baseline
sui medesimi punti. Altrimenti il contributo osservato resta sospeso.

Questo è un controllo spaziale sullo snapshot corrente, non una verifica futura
indipendente, non prova di miglioramento rispetto all'ICON grezzo, né una misura
calibrata di incertezza. Il sito può mostrare separatamente la sola stima con
gradiente standard, dichiarandola esplicitamente. Dove mancano anche le quote,
non viene inventato un risultato. Zero è un valore valido; null resta mancante.

Prima di estendere il prodotto servono un archivio di errori su periodi futuri,
confronti con ICON grezzo e baseline orografica e una rete di osservazioni più
densa, soprattutto nelle aree montane. I criteri non vanno allentati per far
apparire attiva una correzione senza supporto.

## Dati e aggiornamenti

- `data_weather/downscaling.json`: stato, policy, misure e scarti del run.
- `data_weather/downscaling_terrain.json.gz`: quota ICON con geometria e run.
- `data_weather/live/observations.json` e `live/downscaling.json`: aggiornamento
  orario attraverso il collector già esistente, con timestamp e run propri.

La cartella live non modifica lo snapshot osservativo originario o il suo
manifest archivistico. Il frontend verifica il run, l'età del prodotto e delle
singole misure; conserva ICON originale come riferimento. La pubblicazione
oraria ricostruisce il prodotto dai campioni del branch pubblico corrente dopo
ogni conflitto, senza forzare il push. Il collector conserva anche gli snapshot
immutabili negli artifact di Actions.

## Riferimento per il metodo orografico

Dutra et al. (2020), *Environmental Lapse Rate for High-Resolution Land Surface
Downscaling: An Application to ERA5*, https://doi.org/10.1029/2019EA000984.
Il metodo qui implementato usa ancora un gradiente standard, non il gradiente
variabile da profili proposto nello studio: non eredita i risultati dello studio.
