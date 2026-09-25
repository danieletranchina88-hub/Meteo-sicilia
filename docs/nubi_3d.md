# Ricostruzione delle nubi in volume

Il volume non rappresenta una tomografia della nube. La maschera CLM definisce
la copertura, CTH la quota della cima e l'IR MTG il rilievo locale; GeoColour
aiuta con le nubi basse di giorno. Le basi, lo spessore e la struttura interna
sono stime continue vincolate a questi dati. Il radar e le scariche vengono
associati solo a fotogrammi sufficientemente vicini nel tempo. In assenza di
un sensore opzionale, la sua evidenza vale zero e non viene simulata.

## Forme usate

| Genere | Evidenza | Morfologia del volume |
| --- | --- | --- |
| Cumulonembo | Cima fredda e irregolare con nucleo locale; RDT oppure rovescio radar e scariche concordi lo sostengono | Torre a cupole con base bassa nel nucleo; attorno, sommità quasi piatta e ampia incudine ghiacciata limitata alla copertura osservata |
| Nembostrato | Banco continuo e otticamente spesso, sostenuto da precipitazione diffusa nel radar | Manto ampio, sommità poco scolpita e base sfumata dalla precipitazione; senza radar non si inventano veli di pioggia |
| Cirro | Cima alta e otticamente sottile, elementi separati | Filamenti sottili in quota e struttura più aperta; nessuna colonna fino al suolo |
| Cirrostrato | Banco alto sottile e più continuo | Velo di ghiaccio disteso, poco rilievo, traslucido |
| Altocumulo | Cime a quota media con tessitura irregolare | Piccoli lobi arrotondati in banchi, scala più fine dello stratocumulo |
| Altostrato | Copertura media estesa, tessitura più uniforme | Lastra medio-alta continua e morbida, meno lobi dei cumuli |
| Cumulo | Cella bassa isolata con cima irregolare | Base alla quota di condensazione stimata e cupole verticali distinte |
| Stratocumulo | Banco basso con elementi tondeggianti | Lobi ampi e ravvicinati sopra una base bassa condivisa |
| Strato | Banco basso uniforme sostenuto anche dal prodotto notturno | Velo piatto con base abbastanza uniforme, poca scultura |

La quota massima di una incudine non supera la cima stimata da CTH. Il rumore
volumetrico scolpisce i contorni a scala inferiore a quella del satellite,
senza aggiungere una nube dove la maschera indica cielo libero. La classificazione
mostrata al tocco è indicativa: satellite e radar non osservano direttamente la
base nascosta né la distribuzione tridimensionale del ghiaccio e delle gocce.

Riferimenti morfologici: [Cumulonimbus](https://cloudatlas.wmo.int/definition-cumulonimbus-cb.html),
[incus](https://cloudatlas.wmo.int/clouds-supplementary-features-incus.html),
[Nimbostratus](https://cloudatlas.wmo.int/definition-nimbostratus-ns.html),
[Cirrus](https://cloudatlas.wmo.int/clouds-genera-cirrus.html),
[Cirrostratus](https://cloudatlas.wmo.int/en/definition-cirrostratus-cs.html),
[Altocumulus](https://cloudatlas.wmo.int/en/clouds-genera-altocumulus.html),
[Altostratus](https://cloudatlas.wmo.int/clouds-genera-altostratus.html),
[Stratocumulus](https://cloudatlas.wmo.int/en/stratocumulus-sc.html),
[Stratus](https://cloudatlas.wmo.int/en/clouds-genera-stratus.html),
[Cumulus congestus](https://cloudatlas.wmo.int/en/species-cumulus-congestus-cu-con.html)
e [rilevazione delle cime che superano l'incudine](https://cwg.eumetsat.int/overshooting-top-and-enhanced-v-detections/).

## Nubi 3D da fusione satellite + ICON-2I

Il comando "Nubi 3D · satellite + ICON-2I" usa un volume cotto lato server,
alternativo alla ricostruzione nel browser descritta sopra. Satellite e
modello hanno orologi diversi (un fotogramma ogni 10 minuti contro una
scadenza oraria di un run che esce due volte al giorno): il fotogramma
satellitare è la maschera spaziale esatta, il run ICON-2I è solo un
modificatore termodinamico dell'ambiente.

**Pipeline.** `process_data.py` salva, per le prime 36 scadenze del run,
`data_weather/cloud_environment.npz`: LCL di Lawrence sopra l'orografia,
T2m, gradiente T2m–T500 e CAPE (massimo di blocco), su una griglia ridotta
di circa 8 km. Ogni mezz'ora il workflow `bake_cloud_volume.yml` esegue
`scripts/bake_cloud_volume.py`: prende l'ultimo IR 10,5 µm e VIS 0,6 µm di
MTG e la maschera CLM di MSG da EUMETView, interpola linearmente l'ambiente
fra le due scadenze che racchiudono l'istante del satellite (o usa la più
vicina fino a 3 ore fuori intervallo; oltre rifiuta) e pubblica
`data_weather/live/clouds/volume.png` più `volume.json`.

| Canale | Contenuto | Da dove |
| --- | --- | --- |
| R | Quota della cima (km/16) | Temperatura di brillanza IR 10,5 µm invertita sul profilo termico ICON-2I; oltre la tropopausa (12 km) 7 K per km di sfondamento |
| G | Densità iniziale (0–1) | Albedo VIS eccedente il fondo sereno locale, corretta per cos(zenit); di notte contrasto IR con la superficie. Zero dove la CLM dice sereno |
| B | Quota della base (km/16) | LCL ICON-2I sopra l'orografia; lo spessore è limitato da densità e convezione, così un cirro non scende all'LCL |
| A | Modificatore convettivo (0–1) | Corrente potenziale 0,45·√(2·CAPE) / 40 m/s, massimo su 25 km per assorbire lo sfasamento fra cella osservata e simulata |

La scala dei conteggi IR è quella della legenda ufficiale EUMETView dello
stile `mtg_fd_ir105_hrfi_style_02` (−73 °C, −32 °C, +30 °C); la riflettanza
VIS è il grigio lineare del WMS. Entrambe sono dichiarate in
`meteo_analysis/clouds/volume_texture.py`. Un cirro semitrasparente appare
più caldo della sua cima: la quota ne risulta sottostimata, limite di ogni
metodo IR a canale singolo.

**Ray marching** (`nubi_fusione.js`). Il raggio attraversa solo la fascia
fra base (B) e cima (R); la sagoma si legge a piena risoluzione, le quote
un livello e mezzo più in su per non trasformare ogni pixel in un ago. Un
Perlin fBm apre i vuoti macroscopici dove la densità osservata è bassa; un
Worley a tre ottave viene sottratto per i bordi cumuliformi, con un morso
che il canale A porta da 0,14 (strati, rumore schiacciato in lamine) a 0,9
(convezione profonda, cavolfiori). Il canale A scolpisce solo nubi che il
satellite vede spesse. La luce solare segue Beer-Lambert verso il sole, il
termine polvere 1 − e^(−2·densità) sull'out-scattering (spento guardando
verso il sole) e una funzione di fase Henyey-Greenstein a due lobi
(g = 0,6 in avanti) che accende il bordo d'argento in controluce; due ottave
di diffusione multipla impediscono che il nucleo diventi nero. Il PNG si
decodifica in JavaScript perché il canale alfa è un dato e non deve essere
premoltiplicato dal browser.
