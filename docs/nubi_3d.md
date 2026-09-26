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

## Fusione con ICON-2I

Il comando **Nubi 3D · satellite + ICON-2I** usa il motore descritto sopra
(maschera CLM, CTH, IR e GeoColour MTG, Cloud Type/Phase/Fog, RDT, radar,
Lightning Imager, lampi Blitzortung, timeline satellitare) e vi aggiunge
l'ambiente termodinamico di ICON-2I. Il satellite resta l'autorità su
presenza, sagoma e cima; il modello è solo un modificatore dell'ambiente.

**Dal run al browser.** `process_data.py` scrive in `data_weather/cloud_env/`
una piastrella per ora di validità (prime 36 ore del run, circa 10 km di
passo): LCL di Lawrence sopra l'orografia, T2m, gradiente medio T2m–T500,
CAPE (massimo di blocco) e orografia, più `index.json`. Al deploy
`scripts/merge_cloud_environment.py` conserva le ore passate dei run
precedenti (48 ore), così la timeline satellitare mantiene l'ambiente anche
subito dopo un nuovo run. Il browser prende le due ore che racchiudono il
fotogramma che si sta guardando e interpola linearmente; se l'ora più vicina
dista più di 3 ore, o il punto è fuori dal dominio ICON-2I (bordo sfumato su
60 km), il motore torna alle stime dal solo satellite e lo dichiara.

**La texture del volume** è una sola RGBA:

| Canale | Contenuto |
| --- | --- |
| R | Cima: Cloud Top Height; dove manca, IR 10,5 µm invertito sul profilo termico ICON-2I (oltre la tropopausa di 12 km, 7 K per km di sfondamento) |
| G | Copertura/densità osservata: CLM, dettaglio IR, luminosità GeoColour |
| B | Base del genere: LCL ICON-2I per cumuli, cumulonembi, strati e nembostrati; quote dichiarate per nubi medie e alte |
| A | Convezione: corrente potenziale 0,45·√(2·CAPE) / 40 m/s, massimo su 25 km |

La densità ottica del genere è in `uGeneri.a`; la base delle torri (l'LCL)
in una piccola texture a parte, perché il nucleo di un cumulonembo scende
fino alla condensazione anche sotto l'incudine.

**Cosa cambia con il modello.**
- La base dei cumuli e delle torri è l'LCL sopra il rilievo, non 1 km fisso:
  sulle Alpi le basi salgono, in aria umida scendono.
- La cima IR, dove il CTH non è valido, è una quota fisica invece di una
  scala di grigio locale; il confronto con il CTH misura l'opacità dei veli.
- Il CAPE sostiene i cumuli bassi e medi (fino a metà strada verso il
  cumulo) e la prova di convezione profonda, e abbassa fino a 1,5 km la
  quota minima di una torre; in aria stabile la prova di una torre pesa
  meno. Non crea mai da solo né una nube né un cumulonembo.
- Nello shader il canale A gonfia le cupole e fa mordere più a fondo il
  Worley del dettaglio solo su cumuli e fianchi delle torri: cavolfiori in
  aria instabile, strati laminari invariati, nucleo della torre pieno.

Illuminazione (Beer-Lambert verso il sole, termine polvere, doppio lobo di
Henyey-Greenstein, diffusione multipla) e lampi restano quelli del motore:
le scariche Blitzortung accendono la nube osservata in diretta. Toccando una
nube si leggono genere, cima, base (con l'origine: CTH, IR su profilo
ICON-2I, LCL ICON-2I o stima) e CAPE.

Verifiche: `scripts/tests/test_cloud_environment.py` (piastrelle, fusione
temporale, conservazione delle ore passate), `scripts/tests/test_nubi_icon.js`
(lettura della piastrella Python nel browser, cima fisica, basi, CAPE,
texture) e `node scripts/tests/test_map_3d.js --gpu` (sezioni di densità dei
generi su GPU, anche con e senza CAPE).

## Il volume: fusione satellite + ICON-2I nel ray marcher

Il volume disegnato e' quello del primo ray marcher della fusione. Per ogni
istante della timeline il browser calcola una texture RGBA dal satellite
di quello slot e dall'ambiente ICON-2I interpolato (`fusioneDelCampo` in
index.html):

| Canale | Contenuto |
| --- | --- |
| R | Cima: temperatura di brillanza IR 10,5 µm risalita sul profilo ICON-2I, z = zs + (T2m - BT)/Γ, con 7 K/km di sfondamento oltre 12 km |
| G | Densita': albedo VIS 0,6 µm di giorno (scaricato per ogni slot), contrasto IR di notte |
| B | Base: LCL ICON-2I sopra l'orografia, limitata dallo spessore massimo 0,8 + 4·G + 11·(convezione profonda) km |
| A | Convezione: 0,45·√(2·CAPE)/40 m/s, massimo su 25 km |

La maschera CLM resta l'autorita' su dove c'e' nube; R, B e A sono estesi
appena fuori dalla maschera perche' i bordi non scendano a terra.

- **Raggio confinato** fra la base B e la cima R, con passo che cresce con
  la distanza ma segue la fascia della colonna: almeno quattro campioni
  nello spessore, mai un salto che la scavalchi (niente trama a puntini sui
  veli sottili).
- **Rumore 3D in km visti** (verticale moltiplicato per l'esagerazione):
  Perlin fBm a 48 km per i vuoti, Worley a tre ottave a 6 km sottratto per i
  bordi cumuliformi. Il CAPE varia il morso del Worley: cavolfiori per la
  convezione profonda, lamine per gli strati in aria stabile. Il Worley
  scolpisce la forma 0-1 prima della densita', e le torri convettive dense
  restano piene.
- **Luce**: Beer-Lambert verso il sole, con il dettaglio del Worley nei
  primi due passi (le ombre dei lobi sui lobi); powder 1 - e^(-densita' x 2)
  sull'out-scattering; Henyey-Greenstein (g = 0,6) per il bordo d'argento;
  diffusione multipla contenuta; cielo schermato dalla nube sopra
  (occlusione) e base della colonna piu' scura della cupola; foschia.
- **Cima frattale**: la cima del satellite (un valore ogni 1-2 km) e'
  alzata e abbassata da cupole alte quanto larghe (torri 4 km / 1,2 km,
  cumuli 1,5 km / 600 m, lobi 370 m / 120 m), piu' basse negli strati e
  quasi nulle nei veli; ogni ottava si accende quando supera un paio di
  pixel. Bordo superiore netto (120-350 m). Dove la nube e' densa i lobi
  sono pieni e separati da vuoti.
- **Coerenza con lo zoom**: le ombre si calcolano nella stessa geometria
  esagerata che si vede; l'esagerazione cala poco da vicino (da zoom 7,5,
  minimo 2,4); il raggio entra nella nube a passi quattro volte piu' corti,
  cosi' la cima netta non diventa grana.
- **Dettaglio da vicino**: ottave di Worley a 370 m e 90 m che si accendono
  solo quando sono piu' grandi di un paio di pixel.
- **Qualita'** (fissa, per dispositivo): desktop campo 2560 px, foto 3072,
  tela fino a 3,2 Mpx, 320 passi e 7 verso il sole; telefono campo 1200 px,
  tela 720 kpx, 176 passi e 5 verso il sole.
- **Lampi**: le scariche Blitzortung accendono la nube dall'interno e
  disegnano il canale sotto la base.
- **Timeline**: tornando indietro nel tempo si ricalcola la fusione con le
  immagini e l'ambiente di quell'istante.
- **Superficie**: sotto le nubi resta visibile la foto satellitare
  GeoColour (`SUPERFICIE_FOTOGRAFICA = true` in index.html).

Le verifiche GPU (`node scripts/tests/test_map_3d.js --gpu`) controllano la
torre del Cb piena con la cima alla CTH e la base all'LCL, lo strato nella
sua fascia, l'effetto del CAPE sui cumuli e l'assenza di nube fuori dalla
copertura osservata.

## Realismo (PC) e versione leggera (telefono)

- **Diffusione multipla a ottave** (Wrenninge): ogni ottava attenua meno e
  diffonde meno in avanti; da' il bagliore dentro i cumuli e il
  grigio-azzurro delle facce in ombra. Sul telefono una sola ottava.
- **Cielo e suolo**: il cielo azzurro illumina dall'alto e tinge le ombre,
  il suolo rimanda poca luce calda; le cavita' fra i lobi sono piu' scure.
- **Prospettiva aerea** azzurra con la distanza e **curva filmica ACES**.
- **Incudine**: con convezione profonda e cima sopra 7,5 km, fuori dalle
  torri la nube e' una lastra di 2-3 km in quota; le torri stanno dove il
  satellite vede la cima sporgere (overshooting top) e in celle di ~15 km.
- **Accumulo a mappa ferma** (solo PC): 12 fotogrammi con scarto del raggio
  diverso, media in mezza precisione; si ricomincia a ogni cambio di vista
  o di campo e non si accumula durante un lampo.
- **Qualita'**: PC 4,5 Mpx, 384 passi, 8 verso il sole; telefono 720 kpx,
  176 passi, 5 verso il sole, niente accumulo.

## Il motore d'inferenza delle nubi

Il satellite vede la sommità e la proiezione della copertura: è un
**vincolo**, non la geometria 3D. Fra i dati e il renderer c'è un motore
meteorologico (`InferenzaNubi` in `index.html`, testato da
`scripts/tests/test_inferenza_nubi.js`):

```
satellite (CLM, CTH, IR, VIS, Cloud Type/Phase/Fog, RDT) + radar + fulmini
+ ICON-2I (CAPE, CIN, LCL, zero termico, UR 850/700/500, vento 250/500,
  shear 0-6 km, coperture CLCL/CLCM/CLCH, pioggia convettiva e di scala)
  -> classificazione esistente (tessitura, nucleo, incudine, pioggia)
  -> punteggiNube: compatibilità fisica con 15 tipi (regole leggibili)
  -> inferisciStati: tipo, fiducia, alternativa, base, cima, morfologia
  -> trovaTorri: oggetti convettivi (massimi locali), tracciati nel tempo
  -> texture di stato + uniform delle torri
  -> shader: geometria per archetipo, illuminazione
```

**Tipi**: Cirro, Cirrostrato, Cirrocumulo, Altostrato, Altocumulo, Strato,
Stratocumulo, Nembostrato, Cumulo humilis/mediocris/congestus, Cumulonembo
calvus/capillatus/incus, Incudine.

**Esempi di coerenza fisica**: nembostrato = sommità liscia e spessa +
precipitazione estesa (radar o pioggia di scala ICON-2I) + aria satura, poca
convezione; cumulonembo = nucleo convettivo osservato (rilievo IR, RDT,
radar forte + fulmini) in alta troposfera, capillatus se la cima è glaciata
(IR < -38 °C), incus se c'è il manto freddo attorno; stratocumulo = banco
esteso a celle in aria poco convettiva (CIN), cumuli = celle separate con
CAPE. Il CAPE sostiene, non crea: senza nucleo osservato non nasce un Cb.

**Struttura verticale**: la cima è osservata (CTH, IR sul profilo ICON-2I);
la base è inferita per tipo: LCL per cumuli e cumulonembi, strato basso
dall'LCL limitato, altostrato sopra lo zero termico, cirri sopra 5,5 km,
incudine 2,6 km sotto la cima. Il tipo dominante pesa sulla base con la
quarta potenza dei punteggi, la morfologia con il quadrato: dove la fiducia
è bassa gli archetipi compatibili si mescolano, senza salti di forma.

**Morfologia per tipo** (parametri nella texture di stato): cumuliforme,
sviluppo verticale, scala delle celle, aperture, fibre, pioggia, onda.

- Celle (Cu, Sc, Ac, Cc): una cella per nodo di una griglia sfalsata con
  seme proprio; ogni elemento è una **pila di bolle** tonde in km visti, base
  piatta al livello di condensazione, altezza non oltre qualche volta la
  larghezza; celle vuote per le aperture (più numerose dove la copertura
  osservata è parziale).
- Strati (St, As, Ns, Cs): volumi con cima e base ondulate dolcemente.
- Cirri: fibre stirate nel verso del vento a 250 hPa.
- Oggetti convettivi: torre principale inclinata dallo shear 0-6 km, 1-4 torri
  secondarie (multicella), overshooting top se la cima supera 11,5 km,
  incudine sottovento (direzione osservata del manto freddo, altrimenti vento
  a 250 hPa) spessa sopra la torre e sottile ai bordi; nelle torri in
  dissipazione il corpo si assottiglia.
- Pioggia sotto la base per Ns e Cb.
- Il rumore arricchisce la superficie (microscala), non crea la forma.

**Coerenza temporale**: ogni torre ha id e seme (stessa forma procedurale);
ricalcolando lo stesso istante (arrivo di radar, fulmini, RGB) li conserva;
fra istanti vicini (≤ 45 min) la stessa torre è riconosciuta e lo stadio
(in formazione, in crescita, matura, in dissipazione) viene dall'andamento
di cima e fulmini.

**Debug meteorologico**: `?debug=nubi` (o il cursore nel pannello
`?regola=1`) colora le nubi per tipo. Toccando una nube si apre una scheda
con tipo, fiducia, alternativa, base/cima/spessore con la fonte,
temperatura della sommità, ambiente ICON-2I, radar e fulmini, la torre più
vicina (stadio, secondarie, overshooting, direzione e fonte dell'incudine,
shear), le ragioni della classificazione e i parametri morfologici.

**Dati ICON-2I nuovi** (backend, `prepare_icon_cloud_fields`, tutti
facoltativi, scaricati a parte dalla diagnostica temporali): vento a 250 hPa,
UR 850 e 500 hPa, CLCL/CLCM/CLCH, RAIN_CON e RAIN_GSP (intensità orarie);
nella piastrella anche CIN, zero termico, vento a 500 hPa, shear 0-6 km e UR
700 hPa derivata da T e QV.

## La geometria e le prestazioni secondo Nubis³ (Guerrilla, SIGGRAPH 2023)

Le forme non sono più disegnate a mano (pile di bolle, celle a griglia): il
renderer segue il metodo pubblicato da Guerrilla per Horizon Zero Dawn e
Horizon Forbidden West, lo stesso filone da cui vengono le nubi dei
simulatori moderni. Asobo non ha pubblicato i dettagli del motore nuvole di
Microsoft Flight Simulator; i riferimenti tecnici pubblici completi sono:

- A. Schneider, "Nubis, Cubed: Methods (and madness) to model and render
  immersive real-time voxel-based clouds", Advances in Real-Time Rendering,
  SIGGRAPH 2023 (https://www.guerrilla-games.com/read/nubis-cubed).
- A. Schneider, "Nubis, Evolved", SIGGRAPH 2022; "The Real-Time Volumetric
  Cloudscapes of Horizon Zero Dawn", SIGGRAPH 2015.
- SideFX, rumore Alligator (HDK, alligator.C).

Come si traduce qui:

1. **Profilo dimensionale** (0 fuori, 1 nel nucleo), costruito dallo stato
   inferito: profilo verticale per tipo (lastra per gli strati, base netta e
   cima che si assottiglia per i cumuliformi, fascia sottile per i cirri) per
   la copertura osservata; gli oggetti convettivi sono inviluppi a distanza
   con segno (colonna inclinata con cupola schiacciata, torri secondarie
   fuse, cupola che sfonda, incudine a tetto piatto sottovento).
2. **Forma**: il profilo fa da copertura sul rumore Perlin-Worley degli
   ammassi (`remap(grumi, 1 - profilo, 1, 0, 1)`), come in Horizon Zero
   Dawn; nel nucleo la nube resta piena.
3. **Dettaglio** (up-rez di Nubis³): un solo campione del rumore Alligator
   a scala più piccola, *billowy* per i cumuliformi e *wispy* riccioluto per
   veli e cirri, che erode il bordo tenue (`remap(forma, dettaglio, 1, 0,
   1)`), più il rumore "piegato due volte" da vicino e la nitidezza
   `pow(d, 0.3..0.6)`.
4. **Luce**: i primi due passi verso il sole vedono il dettaglio, gli altri
   solo la forma grande; bagliore interno dal profilo (campo di probabilità
   di diffusione, meno attenuato verso il sole); luce del cielo
   `sqrt(1 - profilo)`.
5. **Vuoto saltato** (sphere tracing): una griglia 3D grossa (384×355×24 sul
   PC, 192×178×24 sul telefono) con i km minimi dalla nube più vicina,
   calcolata in un Web Worker a ogni campo (trasformata di distanza euclidea
   esatta, sempre per difetto: la prova `vuoto: distanza sempre per difetto`
   lo verifica a forza bruta). Il raggio attraversa il cielo sereno in pochi
   passi.
6. **Risoluzione**: il volume si disegna in al massimo 2,1 Mpx sul PC (0,52
   sul telefono) e si accumula a mappa ferma; 256 passi massimi (128 sul
   telefono).

Sul banco GPU software (llvmpipe) la stessa scena costa: cumulonembo 3,9 →
0,5 s, stratocumulo 0,9 → 0,3 s, strato 0,45 → 0,23 s; sul sito si aggiungono
i salti del vuoto e i pixel dimezzati.

## La struttura verticale dal modello (ICON-2I + ICON-EU)

Il satellite vede la cima e la posizione, non la struttura verticale, e non
vede sotto una coltre. Il volume copre il satellite dove c'e' ICON-EU
(23,5 W - 42 E, 29,5 - 66 N); sull'Italia (dominio ICON-2I, 3-22 E,
33,7-48,9 N) si aggiunge l'ambiente ICON-2I.

Fonti verticali disponibili (verificate il 26/09/2026):

| Fonte | Livelli | Nubi per livello | Accesso |
|---|---|---|---|
| ICON-2I, MeteoHub (in uso) | 1000/925/850/700/500/250 hPa | no (CLCL/M/H) | aperto |
| ICON-2I model levels, MeteoHub | ~65 livelli nativi | da verificare | account + API `/api/data` (10 richieste/ora, 1 GB) |
| MOLOCH, MeteoHub | 9 (1000-300 hPa) | no | aperto |
| WRF, MeteoHub | 12 (1000-200 hPa) | no | aperto |
| **ICON-EU, DWD (in uso)** | 20 isobarici + 74 del modello | **CLC, QC, QI** | aperto, opendata.dwd.de |
| ICON-D2, DWD | 65, 2,2 km | sì | aperto, ma solo fino a ~43 N |

Cosa entra nella piastrella ambiente per ogni ora:

- da ICON-2I: spessore della nube (particella pseudoadiabatica fino al
  livello di equilibrio con T a 700/500/250 hPa, o strato umido UR >= 75%),
  gradiente 700-500 hPa, UR media 850-500;
- da ICON-EU, in una serie a parte `data_weather/cloud_eu/` su tutta
  l'Europa del volume: la frazione di nube CLC su 16 livelli (1000-200 hPa),
  `c1000` ... `c200`, mediata 2x2 (0,125 gradi, circa 12 km; ~780 kB per
  ora). Se il run DWD con la stessa ora non e' ancora uscito si usa il
  precedente (ogni 3 ore); le ore passate si conservano come per l'ambiente.

Nel browser:

- durezza (stabilita' 700-500, CAPE, secchezza): profilo verticale da strato
  soffice a cumulo compatto a base stretta; spessore del modello come limite
  dei cumuliformi; vento a 250 hPa della cella per i cirri;
- CLC per quota in una texture 3D (0-16 km, 500 m): dentro la fascia
  osservata sposta la massa dove il modello ha nube; SOTTO la fascia, dove il
  satellite non vede, disegna gli strati del modello (prova GPU "Strati");
- la griglia del vuoto include gli strati del modello;
- l'ispezione al tocco elenca gli strati ICON-EU della colonna.
