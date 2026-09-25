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

## Il volume: nubi piene, non superfici

Il ray marcher non scolpisce piu' la cima come una superficie (che faceva
sembrare le nubi montagne): ogni colonna osservata e' un volume pieno fra la
base (canale B) e la cima (canale R), con quote lette a scala leggermente
piu' grossa della sagoma ed estese fuori dalla maschera, cosi' i bordi non
scendono a terra.

- **Profilo verticale per genere**: base piatta e cupola per cumuli e torri;
  lastra con cima e base morbide per strati e veli; base sfumata dalla
  pioggia per il nembostrato; cima piatta e larga per l'incudine; fili nel
  verso del vento per il cirro.
- **Strutture**: celle da ~3 km per lo stratocumulo e da 1-1,5 km per
  l'altocumulo, separate da solchi sopra una base comune; cupole a lobi per
  i cumuli; fianco a cavolfiore per la torre del cumulonembo, con il nucleo
  pieno fino alla cima misurata e la base alla condensazione (LCL ICON-2I).
- **Rumore 3D**: Perlin a 48 km per i vuoti macroscopici dove la nube e'
  otticamente sottile; Worley a 6 km sottratto per i bordi cumuliformi. Il
  CAPE (canale A) varia il morso del Worley: cavolfiori estremi per la
  convezione profonda, lamine per gli strati in aria stabile. Si scolpisce
  la forma 0-1 e poi la si moltiplica per la densita' ottica del genere.
- **Luce**: Beer-Lambert verso il sole; powder 1 - e^(-densita' x 2)
  sull'out-scattering, spento guardando verso il sole; Henyey-Greenstein
  a doppio lobo (g = 0,6 in avanti) per il bordo d'argento in controluce;
  diffusione multipla in due ottave perche' il nucleo non diventi nero;
  luce del cielo dall'alto e del suolo dal basso; foschia con la distanza.
- I lampi Blitzortung accendono la nube dall'interno come prima.
- Sotto le nubi resta la carta: la foto GeoColour della superficie e'
  disattivata (`SUPERFICIE_FOTOGRAFICA` in index.html).

Le verifiche GPU (`node scripts/tests/test_map_3d.js --gpu`) controllano
nucleo pieno del cumulonembo, cima alla CTH, torre piu' stretta
dell'incudine, stratocumulo senza strisce, altocumulo piu' strutturato
dell'altostrato, nessuna nube fuori dalla copertura osservata, e l'effetto
del CAPE sui cumuli.
