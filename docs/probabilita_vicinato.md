# Probabilità di vicinato

## La domanda giusta

Un modello a 2,2 km risolve i singoli rovesci. Sa che nel pomeriggio ci sarà un
temporale sull'Appennino; non sa su quale paese scaricherà. Chiedergli «quanti
millimetri cadono in questo punto» è una domanda mal posta: la risposta esiste
nel file, ma è una scelta arbitraria fra decine di posizioni ugualmente
plausibili, e sistematicamente sbagliata per il punto che interessa a chi legge.

La domanda a cui il modello sa rispondere è: **che probabilità c'è che entro R
chilometri da qui si superi la soglia T?** È anche l'unica delle due che si
possa verificare contro le osservazioni, perché non pretende che il modello
azzecchi una posizione che non è in grado di prevedere.

È il metodo standard per i modelli che risolvono la convezione: Theis, Hense e
Damrath (2005), *Probabilistic precipitation forecasts from a deterministic
model: a pragmatic approach*, DOI
[10.1017/S1350482705001763](https://doi.org/10.1017/S1350482705001763);
Schwartz e Sobash (2017), *Generating Probabilistic Forecasts from
Convection-Allowing Ensembles Using Neighborhood Approaches*, DOI
[10.1175/MWR-D-16-0400.1](https://doi.org/10.1175/MWR-D-16-0400.1).

## Cosa viene calcolato

Nessun addestramento, nessun parametro libero, nessun archivio: una frazione di
punti in un cerchio.

**Primo stadio — l'evento diventa areale.** `E(x) = 1` se il campo raggiunge la
soglia in almeno un punto entro 10 km da `x`. Qui non c'è ancora probabilità:
è solo la definizione di cosa conta come «è successo». Serve perché la frazione
d'area interessata è una quantità corretta ma piccola per costruzione quando il
fenomeno è a chiazze: sul run del 6 settembre 2026 la pioggia oraria sopra 1 mm
copriva lo 0,013% dei punti, e la frazione su un disco da 15 km non superava
0,33 nemmeno dentro il rovescio. Letta come «probabilità che piova» mentirebbe
verso il basso.

**Secondo stadio — la posizione è incerta.** La probabilità è la frazione di
punti entro 25 km dove `E` vale 1. Un modello a 2,2 km sbaglia dove mette la
cella, non se la cella esiste: 25 km è la scala su cui quella posizione è
incerta.

Sullo stesso run, la pioggia sopra 1 mm passa da 0,33 a 0,65 e la raffica sopra
75 km/h da 0,12 a 0,83.

## Il vicinato è un cerchio sul terreno

La griglia ICON-2I è regolare in gradi, non in chilometri. Il passo in
latitudine resta 2,226 km ovunque; quello in longitudine passa da 2,315 km a
33,7 °N a 1,829 km a 48,9 °N, perché i meridiani si stringono verso il polo.

Un disco con lo stesso numero di celle nelle due direzioni sarebbe quindi
un'ellisse sul terreno. Con un passo medio unico, un raggio nominale di 25 km al
bordo nord del dominio diventa 22,6 km in longitudine e 27,5 in latitudine:
schiacciato del 18%. Per questo il raggio in celle lungo x è ricavato dalla
latitudine di ciascuna riga.

Due convenzioni, dichiarate perché non sono le uniche possibili:

- le distanze est-ovest usano il passo del punto **centrale**, non del vicino.
  Su 25 km la latitudine cambia al massimo di 0,22° e il coseno con essa dello
  0,45%: trascurabile accanto al 18% che si sta correggendo, e in cambio il
  vicinato resta un insieme ben definito attorno al punto;
- al bordo del dominio il vicinato **si restringe** invece di replicare l'ultima
  riga. Replicare farebbe pesare su una probabilità un dato che non esiste; qui
  il mezzo disco disponibile sta sia al numeratore sia al denominatore.

I punti mancanti non contano né a favore né contro: escono da entrambi gli
insiemi.

## Soglie

Non sono numeri tondi scelti per simmetria: sono i gradini a cui cambia quello
che succede.

| Campo | Soglie | Perché |
|---|---|---|
| Pioggia (accumulo orario) | 1, 5, 10, 20 mm/h | 1 separa bagnato da asciutto; 5 è un rovescio vero; 10 è quando i tombini non ce la fanno; 20 è la soglia a cui rispondono i piccoli bacini |
| Raffiche a 10 m | 50, 75 km/h | I gradi Beaufort già usati dalla scala colore: 7 (vento forte) e 9 (burrasca forte) |

## Costo e risoluzione

Il disco è scomposto in segmenti orizzontali e ogni segmento si legge in tempo
costante da una somma cumulata: il costo cresce con il raggio in righe, non in
celle. Circa 0,26 s per campo su 761×761. Il conto multi-soglia riusa le due
somme che non dipendono dalla soglia, perché dipendono solo da dove il campo
esiste.

Il prodotto viene pubblicato su griglia diradata di due (≈4,4 km). Il campo esce
da una media su 25 km: sotto i 4 km non ha più struttura da rappresentare.
Misurato sul run del 6 settembre, diradare di 2 costa in media 0,12 punti di
probabilità e al massimo 3,9; diradare di 5 arriverebbe a 15, troppo. Il blocco
pesa circa 35 KB compressi per scadenza, l'1,1% del file del passo.

## Limiti

Questa è una **frequenza geometrica su una singola corsa**, non una probabilità
calibrata. In particolare:

- non c'è un archivio che la verifichi. Finché non esiste una serie di coppie
  previsione–osservazione lunga abbastanza, non si può dire se «60%» significhi
  davvero sei volte su dieci;
- il raggio di dispersione di 25 km è una **scelta dichiarata, non tarata**.
  Cambiarlo cambia quanto la probabilità si spalma, non dove sta il fenomeno;
- il raggio non cresce con la scadenza, benché l'incertezza di posizione cresca.
  Farlo crescere richiederebbe una legge, e una legge inventata sarebbe peggio
  di un raggio costante dichiarato;
- viene da una corsa sola. Non c'è nessun ensemble dietro: l'incertezza
  rappresentata è solo quella di posizione, non quella delle condizioni
  iniziali né quella del modello.

Il sito lo dice nella scheda scientifica del campo, e il JSON pubblicato lo
dichiara nel campo `semantics`.

## Dati

`data_weather/step_N.json.gz`, blocco `prob`:

```
prob.method          neighbourhood-exceedance-deterministic-v1
prob.semantics       frequency-of-occurrence-within-eventRadiusKm-not-calibrated
prob.eventRadiusKm   10
prob.spreadRadiusKm  25
prob.nx, ny, lo1, la1, dx, dy    geometria della griglia diradata
prob.fields.rain_1 … gust_75     percentuali intere 0-100, null dove ignoto
```

Codice in `meteo_analysis/core/neighbourhood.py`, prove in
`scripts/tests/test_neighbourhood.py`. Le prove più importanti confrontano
l'implementazione veloce con la definizione applicata per forza bruta, così
controllano la formula e non se stesse.
