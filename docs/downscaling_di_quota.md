# Downscaling di quota

## Il problema

Un modello a 2,2 km non sa che le montagne esistono: le smussa. Misurato sul
run del 6 settembre 2026, confrontando l'orografia ICON-2I con la quota reale:

| cima | reale | ICON-2I | errore | in temperatura |
|---|---|---|---|---|
| Gran Sasso | 2912 m | 2078 m | −834 m | +5,4 °C |
| Monte Bianco | 4808 m | 4000 m | −808 m | +5,3 °C |
| Marmolada | 3343 m | 2616 m | −727 m | +4,7 °C |
| Monte Rosa | 4634 m | 3936 m | −698 m | +4,5 °C |
| Etna | 3357 m | 2866 m | −491 m | +3,2 °C |
| Vesuvio | 1281 m | 886 m | −395 m | +2,6 °C |
| Pizzo Carbonara | 1979 m | 1733 m | −246 m | +1,6 °C |

La temperatura a 2 m che il modello scrive in quei punti è la temperatura di
una montagna più bassa. È un errore sistematico, grande, presente ogni giorno,
e non serve nessuna stazione per correggerlo: serve sapere quanto è alta la
montagna e come varia la temperatura con la quota.

**Da non confondere con l'altro "downscaling".** Il sito ha anche un'analisi
oggettiva alla Cressman che tira i valori verso le stazioni osservate. Quella
corregge *cosa* dice il modello, non *quanto fine* lo dice: resta sulla griglia
a 2,2 km e funziona solo dove ci sono stazioni. Questa invece cambia la
risoluzione, e vale ovunque. Le due si compongono e stanno sotto lo stesso
interruttore.

## Il metodo

    T(z_vero) = T(z_modello) + [ Tprofilo(z_vero) − Tprofilo(z_modello) ]

Due ingredienti.

**La quota vera** viene dalle tessere Terrarium che già alimentano la vista 3D
(Mapzen / AWS `elevation-tiles-prod`, fino a zoom 15). Qui vengono lette per
numero invece che per rilievo: `h = R·256 + G + B/256 − 32768`. Lo zoom si
sceglie in modo che il passo del DEM somigli a quello del raster, con un tetto
di 48 tessere per vista; il raster meteo è già costruito a risoluzione schermo,
quindi il dettaglio compare allo zoom in cui si sta guardando.

La batimetria è negativa e viene azzerata: senza questo, la correzione
inventerebbe gradi in mezzo al Tirreno, dove il modello ha giustamente quota
zero.

**Il profilo** non si assume, si legge dal modello. Assumere il gradiente
dell'atmosfera standard, −6,5 °C/km, è comodo e in inverno è **sbagliato di
segno**: sotto un'inversione di fondovalle la temperatura cresce con la quota, e
una cima sopra l'inversione è più calda del paese sotto, non più fredda. Perciò
la pipeline pubblica il profilo termico che il modello ha davvero calcolato, ai
livelli di 925, 850 e 700 hPa.

Le quote di quei livelli non sono costanti — l'aria calda è più spessa — e si
ricavano dall'equazione ipsometrica:

    dz = (R_d / g) · T̄ · ln(p_basso / p_alto),   R_d/g = 29,27 m/K

esatta a meno dell'umidità: usare la temperatura invece di quella virtuale
sottostima lo spessore di meno dell'1%. Fra i livelli si interpola linearmente
in quota; oltre si prolunga il gradiente del segmento estremo, perché inventare
un gradiente diverso dove non c'è dato sarebbe peggio.

Solo la **differenza** conta, quindi lo scarto fra la temperatura a 2 m e quella
dell'aria libera alla stessa quota si elide, e non serve assumere quanto valga.

## Guardie

- sotto mezzo metro di dislivello non si corregge: non c'è niente da correggere;
- sopra 2500 m di scarto non si corregge: non è più orografia, è un errore;
- una tessera che non si decodifica (CORS assente, rete caduta) annulla la
  correzione invece di falsarla;
- se manca uno dei tre livelli barici il profilo non viene pubblicato affatto:
  un profilo con un buco produrrebbe correzioni peggiori dell'assenza di
  correzione.

## Dove si vede

Mappa e lettura del punto applicano la stessa correzione. Se la applicasse solo
la mappa, la carta direbbe una temperatura e il punto cliccato un'altra.

L'interruttore è quello già presente, «Ricalcola mappa · Downscaling». Non si
spegne più quando mancano le osservazioni: la correzione di quota non ne ha
bisogno, e spegnersi lascerebbe scoperta proprio la montagna.

## Limiti

- il DEM è smussato quanto lo zoom scelto. A zoom 8 il passo è ~600 m e il Gran
  Sasso risulta 2571 m invece di 2912: la correzione è reale ma parziale, e si
  affina zoomando. Non è un difetto da nascondere — è il motivo per cui la
  correzione cresce mentre si entra nella carta;
- il profilo ha tre livelli. Un'inversione più sottile dello strato 925–850
  (circa 750–1450 m) non viene risolta;
- è una correzione **di quota**, non un modello. Non aggiunge fenomeni che
  ICON-2I non ha: non inventa brezze di valle, ristagni d'aria fredda,
  incanalamenti, né l'effetto dell'esposizione dei versanti. Quelli sono altri
  pezzi di fisica, e restano da fare;
- si applica alla temperatura a 2 m al suolo. Gli altri campi restano com'erano.

## Dati e codice

`data_weather/step_N.json.gz`, blocco `profile`:

```
profile.method       hypsometric-925-850-700
profile.semantics    model-own-profile-not-standard-lapse-rate
profile.levelsHpa    [925, 850, 700]
profile.nx, ny, lo1, la1, dx, dy    griglia diradata di 8 (~17 km)
profile.z[0..2]      quote dei livelli, in metri sul mare
profile.t[0..2]      temperature dei livelli, in gradi
```

Il campo è sinottico: diradarlo di 8 costa in media 0,003 °C sulla correzione e
al massimo 0,06, e pesa 51 KB compressi per scadenza (l'1,6% del file del passo).

Fisica in `meteo_analysis/core/vertical_profile.py`, prove in
`scripts/tests/test_vertical_profile.py`; lettura del DEM e applicazione in
`index.html`. Le prove ritrovano le quote dell'atmosfera standard partendo
dall'equazione ipsometrica, e verificano i casi in cui un gradiente assunto
sbaglierebbe: profilo isotermo (nessuna correzione) e inversione (correzione di
segno opposto).
