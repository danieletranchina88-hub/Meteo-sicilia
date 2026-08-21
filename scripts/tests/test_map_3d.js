#!/usr/bin/env node
"use strict";

// Regression checks for the static GitHub Pages client. They intentionally do
// not require WebGL, so they also run on the standard GitHub Actions runner.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "../..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const radarHtml = fs.readFileSync(path.join(root, "radar.html"), "utf8");
const meteogramHtml = fs.readFileSync(path.join(root, "meteograms.html"), "utf8");

const inlineScripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
  .map((match) => match[1])
  .filter((source) => source.trim());
assert.ok(inlineScripts.length > 0, "script applicativo assente");
inlineScripts.forEach((source) => {
  assert.doesNotThrow(() => new Function(source), "JavaScript inline non valido");
});

const radarScripts = [...radarHtml.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
  .map((match) => match[1]).filter((source) => source.trim());
radarScripts.forEach((source) => {
  assert.doesNotThrow(() => new Function(source), "JavaScript radar non valido");
});
assert.match(radarHtml, /const MODEL_BOUNDS = \[\[DOMAIN\[0\], DOMAIN\[1\]\], \[DOMAIN\[2\], DOMAIN\[3\]\]\];/,
  "la pagina radar non usa il dominio ICON-2I come confine della mappa");
assert.match(radarHtml, /minZoom: startView\.zoom,\s*\n\s*maxBounds: MODEL_BOUNDS,/,
  "la pagina radar permette di uscire o rimpicciolire oltre il dominio");
assert.match(radarHtml, /map\.setMinZoom\(view\.zoom\);/,
  "il limite minimo radar non viene ricalcolato dopo il ridimensionamento");
assert.match(radarHtml, /const CLOUD_MAX_SIDE = 4096;/,
  "la pagina radar non richiede il massimo dettaglio satellitare WMS");
assert.match(radarHtml, /const sideLimitedWidth = Math\.floor\(CLOUD_MAX_SIDE \/ Math\.max\(aspect, 1\)\);/,
  "la pagina radar puo' superare il limite WMS sul lato verticale");
assert.match(radarHtml, /const RADAR_MAX_NATIVE_ZOOM = 7;/,
  "la pagina radar non dichiara il limite ufficiale RainViewer");
assert.match(radarHtml, /maxzoom: RADAR_MAX_NATIVE_ZOOM,/,
  "la pagina radar continua a chiedere tessere con zoom non supportato");
assert.match(radarHtml, /maxZoom: 16,/,
  "il limite delle tessere e' stato confuso con quello della mappa");
assert.match(radarHtml, /id: "satellite-clouds-layer"[\s\S]{0,260}?source: "satellite-clouds"/,
  "la pagina radar non fonde la copertura nuvolosa satellitare");
assert.match(radarHtml, /loadClouds\(frame, false\);/,
  "satellite e radar non cambiano insieme durante l'animazione");
assert.match(radarHtml, /\/512\/\{z\}\/\{x\}\/\{y\}/,
  "la pagina radar non usa i tile RainViewer 512 px");
assert.match(radarHtml, /tileSize: 512,/,
  "MapLibre non conosce la dimensione HD dei tile radar");
assert.match(radarHtml, /id="radar-toggle"[\s\S]{0,180}?Radar: ON/,
  "manca il pulsante indipendente per spegnere il radar");
assert.match(radarHtml, /id="product-select"/,
  "manca il selettore dei prodotti satellitari nella pagina radar");
assert.match(radarHtml, /World_Imagery\/MapServer\/tile/,
  "la pagina osservativa non usa una base fotografica reale");

const meteogramScripts = [...meteogramHtml.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
  .map((match) => match[1]).filter((source) => source.trim());
meteogramScripts.forEach((source) => {
  assert.doesNotThrow(() => new Function(source), "JavaScript meteogrammi non valido");
});
assert.match(meteogramHtml, /function interpolateTileValue\(values, nx, rows, columns\)/,
  "il meteogramma usa ancora il solo punto di griglia piu' vicino");

// La mappa parte sull'intero dominio nativo ICON-2I e quello stesso zoom
// diventa il pavimento: si puo' entrare nel dettaglio, mai tornare al mondo.
assert.match(html, /const MODEL_DOMAIN = \{ west: 3\.0, south: 33\.7, east: 22\.0, north: 48\.9 \};/,
  "i confini della mappa non coincidono con il dominio ICON-2I");
assert.match(html, /const CLOUD_DOMAIN = MODEL_DOMAIN;/,
  "il satellite non copre esattamente la stessa area del modello");
assert.match(html, /const zoomX = Math\.log2\(width \/ \(512 \* longitudeFraction\)\);[\s\S]{0,520}?zoom: Math\.min\(zoomX, zoomY\)/,
  "lo zoom iniziale non adatta tutto il dominio allo schermo");
assert.match(html, /minZoom: startView\.zoom,\s*\n\s*maxBounds: MODEL_BOUNDS,/,
  "zoom-out o trascinamento possono ancora uscire dal dominio del modello");
assert.match(html, /map\.resize\(\);\s*\n\s*lockMapToModelDomain\(false\);/,
  "ruotando il telefono il limite del dominio non viene ricalcolato");
assert.match(meteogramHtml, /id="weather-strip"/,
  "manca la sintesi visuale ogni tre ore");
assert.match(meteogramHtml, /label:"Confidenza", values:s\.stormConfidence/,
  "la probabilita' temporalesca e' mostrata senza confidenza");
assert.match(meteogramHtml, /label:"Raffica", values:s\.gust10/,
  "il meteogramma del vento ignora le raffiche");

assert.match(html, /maplibre-gl@5\.24\.0/, "MapLibre v5 stabile non caricata");
assert.doesNotMatch(html, /map\.transform\b/, "uso di API MapLibre interna e fragile");
assert.match(html, /map\.setSky\(/, "cielo MapLibre nativo assente");
assert.match(html, /function terrainExaggerationForZoom\(/,
  "esagerazione verticale adattiva assente");
assert.match(html, /rasterCanvas\.toBlob\(/,
  "pubblicazione raster asincrona assente");

const projectParticle = html.match(
  /function projectParticle\([\s\S]*?\n {6}\}/
);
assert.ok(projectParticle, "proiezione particelle assente");
// Le particelle si proiettano con la matrice della scena, ottenuta da un layer
// custom: evita di interrogare il modello di elevazione a ogni punto. Verificata
// identica a map.project (0,000000 px) su nove configurazioni di camera.
assert.match(projectParticle[0], /if \(particleMatrix && window\.maplibregl/,
  "proiezione tramite la matrice della scena assente");
assert.match(projectParticle[0], /MercatorCoordinate\.fromLngLat\(/,
  "la quota non entra nella proiezione: il vento in quota resterebbe al suolo");
// Il ripiego pubblico deve restare: se su un dispositivo il layer custom non
// venisse mai disegnato, la matrice resta nulla e si torna a map.project
// invece di lasciare le particelle ferme.
assert.match(projectParticle[0], /return map\.project\(\[longitude, latitude\]\);/,
  "manca il ripiego sulla proiezione pubblica");
assert.match(html, /id: "wind-projection-probe"/,
  "layer custom che fornisce la matrice assente");

const startParticles = html.match(
  /function startParticles\([\s\S]*?\n\s*\}/
);
assert.ok(startParticles, "avvio particelle assente");
assert.doesNotMatch(startParticles[0], /prefersReducedMotion/,
  "l'attivazione manuale non deve essere bloccata dal movimento ridotto");
assert.doesNotMatch(html, /particleCanvas\.style\.visibility\s*=\s*show3D/,
  "il canvas visibile non deve sparire quando si attiva il 3D");
assert.doesNotMatch(html, /terrain-particles-layer/,
  "la CanvasSource 3D incompatibile con alcuni browser mobili è tornata");

const animate = html.match(/function animateParticles\([\s\S]*?\n {6}\}/);
assert.ok(animate, "ciclo di animazione delle particelle assente");
// In 3D ogni map.project interroga il DEM: riusare la posizione proiettata al
// frame precedente come punto di partenza dimezza quelle interrogazioni. Vale
// perche' le particelle vengono azzerate a ogni movimento della mappa.
assert.match(animate[0], /particle\.projected/,
  "cache della proiezione delle particelle assente");
assert.match(animate[0], /particle\.px\s*=\s*end\.x/,
  "la posizione proiettata non viene memorizzata per il frame successivo");
// La cache va invalidata quando la particella salta altrove.
const respawn = html.match(/function respawnParticle\([\s\S]*?\n {6}\}/);
assert.ok(respawn, "respawnParticle assente");
assert.match(respawn[0], /particle\.projected\s*=\s*false/,
  "la cache della proiezione non viene invalidata al respawn");
// Vicino all'orizzonte un passo minimo diventa una corsa enorme sullo schermo.
assert.match(animate[0], /maximumStreak/,
  "tetto alla lunghezza dei tratti vicino all'orizzonte assente");

// La quota del DEM arriva moltiplicata per l'esagerazione verticale: senza
// dividerla i rilievi risulterebbero molto piu' alti del vero.
const elevation = html.match(/function terrainElevationText\([\s\S]*?\n {6}\}/);
assert.ok(elevation, "lettura della quota del terreno assente");
assert.match(elevation[0], /queryTerrainElevation/,
  "la quota non usa l'API pubblica queryTerrainElevation");
assert.match(elevation[0], /raw \/ exaggeration/,
  "la quota non viene divisa per l'esagerazione verticale");

// Le particelle sono bianche: il colore per velocita' competeva con la scala
// cromatica del campo sotto. La velocita' resta leggibile da spessore e alfa.
const style = html.match(/function particleStyle\([\s\S]*?\n {6}\}/);
assert.ok(style, "particleStyle assente");
assert.doesNotMatch(style[0], /rgba\((?!255,\s*255,\s*255)/,
  "le particelle sono tornate colorate");
// Nessun contorno scuro: su tratti sottili un alone piu' largo copre piu'
// superficie del bianco e le particelle si leggono scure.
assert.doesNotMatch(animate[0], /strokeStyle\s*=\s*"rgba\((?!255,\s*255,\s*255)/,
  "le particelle hanno di nuovo un contorno scuro");
assert.doesNotMatch(animate[0], /lineWidth\s*=\s*lineWidth\s*\+/,
  "traccia allargata da alone attorno alle particelle");

// Opzioni utente: densita' e lunghezza delle scie.
assert.match(html, /id="particle-density-select"/, "selettore densita' assente");
assert.match(html, /id="particle-trail-select"/, "selettore scie assente");
assert.match(html, /PARTICLE_DENSITY_FACTOR\s*=\s*\{[^}]*light[^}]*medium[^}]*dense/,
  "fattori di densita' incompleti");
assert.match(html, /PARTICLE_TRAIL_FADE\s*=\s*\{[^}]*short[^}]*medium[^}]*long/,
  "fattori delle scie incompleti");
assert.match(html, /particleTrailFade\(\)/,
  "la dissolvenza non usa la preferenza sulle scie");
assert.match(html, /base \* particleDensityFactor\(\)/,
  "il numero di particelle non usa la preferenza sulla densita'");

// Bulbo umido: derivato nel browser da T e umidita' con l'equazione
// psicrometrica. Verificato contro Stull (2011): scarto medio 0,32 C.
const wetBulb = html.match(/function wetBulbCelsius\([\s\S]*?\n {6}\}/);
assert.ok(wetBulb, "calcolo del bulbo umido assente");
assert.match(wetBulb[0], /saturationVapourHpa/,
  "il bulbo umido non usa la pressione di vapore saturo");
// Cerco la chiamata dentro prepareData, non la definizione della funzione:
// altrimenti l'asserzione passerebbe anche con la derivazione scollegata.
const prepare = html.match(/function prepareData\([\s\S]*?\n {6}\}/);
assert.ok(prepare, "prepareData assente");
assert.match(prepare[0], /deriveWetBulb\(data\)/,
  "il campo bulbo umido non viene derivato al caricamento del passo");
assert.match(html, /data-layer="wetbulb"/, "selettore del campo bulbo umido assente");
assert.match(html, /id="detail-wetbulb"/, "lettura puntuale del bulbo umido assente");

// Fondo cartografico grigio, non bianco.
assert.match(html, /"raster-saturation": -0\.82/,
  "il fondo cartografico non e' desaturato a grigio");
assert.doesNotMatch(html, /#dfeaf1/, "residui del vecchio fondo azzurro");

// In 3D ogni proiezione interroga il DEM. Le particelle si aggiornano a
// rotazione, un sottoinsieme per frame, e il passo si adatta al costo
// misurato sul dispositivo: la densita' a schermo resta quella scelta.
assert.match(animate[0], /particleIndex % stride !== phase/,
  "aggiornamento a rotazione delle particelle assente");
assert.match(animate[0], /const strideDt = dt \* stride;/,
  "il passo temporale non compensa la rotazione: la velocita' del vento cambierebbe");
assert.match(animate[0], /particle\.lng \+= \(u \* strideDt/,
  "l'avanzamento non usa il passo temporale compensato");
assert.match(animate[0], /particleWorkAverage/,
  "misura del costo del ciclo assente");
assert.match(animate[0], /particleStride \+= 1/,
  "il passo non aumenta sui dispositivi lenti");
assert.match(animate[0], /particleStride -= 1/,
  "il passo non torna a diminuire quando c'e' margine");

// Vento in quota: in 3D le particelle salgono all'altezza del livello barico,
// con la stessa esagerazione verticale del terreno per restare coerenti.
assert.match(html, /LEVEL_ALTITUDE\s*=\s*\{\s*"925"[^}]*"850"/,
  "quote dei livelli barici assenti");
assert.match(animate[0], /levelAltitude = particleLevelAltitude\(\) \* verticalScale/,
  "la quota del vento non segue l'esagerazione verticale del terreno");
assert.match(animate[0], /queryTerrainElevation/,
  "la quota del suolo non viene letta per posare le particelle sul rilievo");
assert.match(animate[0], /particle\.groundAt/,
  "la quota del suolo non viene messa in cache: tornerebbe il costo per frame");

// Etichette vettoriali: quelle raster si stiravano sul terreno inclinato.
assert.match(html, /CARTO_LABEL_LAYERS/, "livelli di etichetta vettoriali assenti");
assert.match(html, /tiles\.basemaps\.cartocdn\.com\/fonts/,
  "font delle etichette vettoriali non configurati");
assert.doesNotMatch(html, /id: "labels-layer"/,
  "le etichette raster, che si stirano in 3D, sono tornate");

// Le isoipse a 500 hPa in 3D vanno alla loro quota vera: e' una superficie
// che sta davvero a circa 5,5 km e che si avvalla nelle saccature.
assert.match(html, /const isoElevated = Boolean\(isoTerrain\) && show3D;/,
  "le isoipse non salgono alla loro quota in 3D");
assert.match(html, /projectParticle\(\s*coordinates\[index\]\[0\], coordinates\[index\]\[1\], height\s*\)/,
  "le isoipse non usano la proiezione con quota");

// Sull'ortofoto il campo si alleggerisce zoomando, per lasciar riconoscere il
// terreno sotto. Espressione legata allo zoom: valutata a ogni fotogramma,
// quindi segue la pinch invece di scattare a fine gesto.
assert.match(html, /showSatellite\s*\?\s*\[\s*\n\s*"interpolate", \["linear"\], \["zoom"\]/,
  "il campo non si alleggerisce zoomando sull'ortofoto");
assert.match(html, /id: "satellite-layer"/, "base fotorealistica assente");

// Isoterme: recuperano la lettura in gradi quando il colore del campo si
// alleggerisce sull'ortofoto. Lo zero termico e' evidenziato perche' separa
// pioggia e neve.
assert.match(html, /function buildTemperatureProducts\(/, "isoterme assenti");
assert.match(html, /createContourFeatures\(temperature, meta, 2, 10\)/,
  "passo delle isoterme non conforme (2 gradi, marcate ogni 10)");
assert.match(html, /const freezing = Math\.abs\(value\) < 0\.01;/,
  "lo zero termico non e' distinto dalle altre isoterme");
assert.match(html, /data-toggle="isotherms"/, "interruttore isoterme assente");
// Non basta che il codice esista: deve essere raggiungibile. La guardia del
// disegno sinottico deve elencare ogni prodotto, altrimenti accendendo solo
// le isoterme la funzione esce subito e non compare nulla.
const synoptic = html.match(/function drawSynopticCanvas\([\s\S]*?\n {8}if \([^\n]*\n/);
assert.ok(synoptic, "guardia del disegno sinottico non trovata");
assert.match(synoptic[0], /!showIsotherms/,
  "le isoterme non superano la guardia del disegno sinottico");

// Avviso dati vecchi: servire l'ultimo dato riuscito va bene, non dichiararlo no.
assert.match(html, /const STALE_RUN_HOURS = 9;/, "soglia dati vecchi assente");
assert.match(html, /function markStaleRun\(/, "avviso dati vecchi assente");
assert.match(html, /ui\.runValue\.classList\.toggle\("stale", stale\)/,
  "l'avviso non viene applicato all'intestazione");

// In 3D map.project interroga il DEM a ogni vertice: sulle isolinee, che ne
// hanno migliaia, il disegno si bloccava. Nessun disegno vettoriale deve piu'
// usarla.
assert.doesNotMatch(html, /map\.project\(coordinate\)/,
  "il disegno vettoriale usa ancora la proiezione che interroga il DEM a ogni vertice");
assert.match(html, /function synopticGroundProfile\(/,
  "profilo del suolo campionato assente");
assert.match(html, /const GROUND_SAMPLE_STRIDE = 6;/,
  "campionamento del profilo del suolo assente");
assert.match(html, /if \(measured\) feature\.groundProfile = profile;/,
  "un profilo tutto a zero verrebbe congelato in cache prima che il DEM sia pronto");
// Il viewport della proiezione va aggiornato anche senza particelle attive.
assert.match(html, /function syncProjectionViewport\(/, "sincronizzazione viewport assente");
const drawVectors = html.match(/function drawVectors\([\s\S]*?\n {6}\}/);
assert.ok(drawVectors, "drawVectors assente");
assert.match(drawVectors[0], /syncProjectionViewport\(\)/,
  "senza particelle attive le isolinee userebbero un viewport non aggiornato");

// Frecce del vento: colorate con la stessa scala del campo vento, quindi
// leggibili con la legenda gia' presente, e proiettate senza interrogare il
// DEM a ogni chiamata.
assert.match(html, /const shade = colorFor\(speed, LAYER_INFO\.wind\);/,
  "le frecce non usano la scala di colore del campo vento");
assert.match(html, /const centerPoint = projectParticle\(location\.lng, location\.lat, groundHeight\);/,
  "le frecce usano ancora una proiezione che interroga il DEM");
assert.doesNotMatch(html, /const centerPoint = map\.project\(location\);/,
  "proiezione costosa delle frecce ancora presente");

// Campo di frecce fitto e regolare, leggibile come un flusso continuo anche
// sullo schermo stretto di un telefono.
assert.match(html, /const denseSpacing = \(mobile \? 20 : 23\)/,
  "il campo di frecce non e' fitto");
assert.match(html, /vectorElevated \? 1\.55 : 1/,
  "in 3D il passo delle frecce non si allarga: ricavare la posizione costa di piu'");
// La lunghezza deve variare molto: cortissima con la calma, oltre il passo
// della griglia con vento forte, cosi' le frecce si fondono in linee continue
// lungo il flusso. E' quello a dare l'aspetto di corrente.
assert.match(html, /const length = clamp\(5 \+ speed \* 0\.46/,
  "la lunghezza delle frecce non segue abbastanza la velocita'");
assert.match(html, /function strokeArrowPath\(/,
  "la freccia non usa il tratto a punta aperta");
assert.doesNotMatch(html, /function drawArrow[\s\S]*?closePath\(\)[\s\S]*?function drawVectors/,
  "la punta dei vettori e' tornata a triangolo pieno");

// Fondo scuro e piatto predefinito. Il chiaro e il rilievo sono due scelte
// esplicite, mentre su scuro il rilievo (quando richiesto) inverte la luce.
assert.match(html, /let showTerrain = false;/,
  "il rilievo e' ancora attivo all'avvio");
assert.match(html, /let showLightBase = false;/,
  "il fondo chiaro e' ancora attivo all'avvio");
assert.match(html, /const onDark = !showLightBase && !showSatellite;/,
  "il rilievo non distingue il fondo scuro");
// Niente piu' simulazione del Sole: il rilievo ha una luce cartografica
// fissa da nord-ovest, uguale a qualunque ora della previsione.
assert.match(html, /const RELIEF_ILLUMINATION = 315;/,
  "manca la luce cartografica fissa del rilievo");
assert.match(html, /"hillshade-illumination-direction", RELIEF_ILLUMINATION/,
  "il rilievo non usa la direzione di luce fissa");
assert.match(html, /onDark \? "rgba\(196,224,255,0\.44\)" : "rgba\(255,250,238,0\.18\)"/,
  "su fondo scuro i versanti esposti non si accendono");
assert.doesNotMatch(html, /solarPosition|SOLAR_CENTER|updateSolarLighting/,
  "la simulazione della luce solare e' tornata");
assert.doesNotMatch(html, /updateSkyPalette\(day/,
  "il cielo dipende ancora dall'altezza del Sole");
assert.match(html, /data-toggle="lightbase"/, "interruttore fondo chiaro assente");
assert.match(html, /id: "basemap-dark-layer"/, "base scura assente");
assert.match(html, /id: "basemap-layer"[\s\S]*?layout: \{ visibility: "none" \}/,
  "la base chiara non e' spenta nello stile iniziale");
assert.match(html, /id: "terrain-base"[\s\S]*?layout: \{ visibility: "none" \}/,
  "il rilievo raster non e' spento nello stile iniziale");
assert.match(html, /id: "terrain-detail"[\s\S]*?layout: \{ visibility: "none" \}/,
  "il rilievo DEM non e' spento nello stile iniziale");
assert.match(html, /#map\s*\{[\s\S]*?background: #0a1018;/,
  "prima del caricamento compare ancora un lampo di fondo chiaro");

// Interfaccia: i pannelli galleggiano sulla mappa, quindi il campo di vento
// arriva ai bordi dello schermo invece di lasciare fasce vuote.
assert.match(html, /const edgeInset = 2;/,
  "le frecce non arrivano ai bordi dello schermo");
assert.match(
  html,
  /for \(let y = edgeInset; y < window\.innerHeight - edgeInset; y \+= spacing\) \{\s*\n\s*for \(let x = edgeInset; x < window\.innerWidth - edgeInset; x \+= spacing\) \{/,
  "la griglia delle frecce non copre tutto il riquadro"
);
assert.doesNotMatch(html, /const topInset =|const bottomInset =/,
  "i vecchi margini che lasciavano i lati scoperti sono tornati");

// I pannelli coprono la mappa: restano leggibili grazie alla sfocatura, ma
// lasciano intravedere il campo sotto.
assert.match(html, /--surface: rgba\(8, 24, 38, 0\.72\);/,
  "i pannelli sono tornati opachi sulla mappa");
assert.match(html, /#bulletin-card \{[\s\S]*?background: rgba\(6, 18, 29, 0\.94\);/,
  "il bollettino, che e' testo lungo, deve restare piu' opaco degli altri pannelli");

// Header: su telefono il pannello del marchio cresceva in altezza e finiva
// sopra la scheda dei metadati. Su mobile resta il solo marchio.
assert.match(html, /\.brand-panel \.eyebrow,\s*\n\s*\.brand-panel \.brand-subtitle \{\s*\n\s*display: none;/,
  "su mobile la riga di contesto del marchio invade ancora la mappa");
assert.match(
  html,
  /@media \(min-width: 960px\) \{[\s\S]*?\.brand-panel \.eyebrow,\s*\n\s*\.brand-panel \.brand-subtitle \{\s*\n\s*display: block;/,
  "sul desktop, dove c'e' spazio, il marchio completo non torna"
);
assert.match(html, /\.brand-copy \{\s*\n\s*min-width: 0;/,
  "senza min-width il testo del marchio non puo' stringersi e sfonda l'header");
assert.match(html, /class="brand-copy"/,
  "il blocco di testo del marchio non ha la classe che lo rende comprimibile");

// Nubi osservate dal satellite in tempo reale: una sola immagine sul dominio,
// resa trasparente sulla canvas perche' MapLibre non sa ricavare l'opacita'
// dal colore di un raster.
// Tutti i formati offerti, ognuno con il pixel nativo dello strumento.
assert.match(html, /layer: "mtg_fd:ir105_hrfi"[\s\S]{0,200}?metres: 1000/,
  "canale infrarosso assente o con risoluzione sbagliata");
assert.match(html, /layer: "mtg_fd:vis06_hrfi"[\s\S]{0,200}?metres: 500/,
  "canale visibile a 500 m assente: e' la risoluzione migliore disponibile");
assert.match(html, /layer: "mtg_fd:rgb_truecolour"/, "colori reali assenti");
assert.match(html, /layer: "mtg_fd:rgb_geocolour"/, "GeoColour assente");
assert.match(html, /layer: "mtg_fd:rgb_cloudtype"/, "tipo di nube assente");
assert.match(html, /id="satclouds-select"/, "manca il selettore del formato");
[
  "ir105", "vis06", "truecolour", "geocolour", "cloudphase",
  "cloudtype", "fog", "dust", "snow", "lightning", "satprecip",
  "firetemp", "frp"
].forEach((key) => {
  assert.ok(html.includes('value="' + key + '"'),
    "il formato " + key + " non e' selezionabile dall'interfaccia");
  assert.ok(radarHtml.includes('value="' + key + '"'),
    "il formato " + key + " manca nella pagina radar");
});

// L'istante va fissato: senza, il servizio compone il mosaico con passaggi
// diversi e sul dominio intero si vedono i riquadri.
assert.match(html, /"&time=" \+ slot\.iso/,
  "senza istante esplicito il mosaico mescola passaggi diversi");
assert.match(html, /const CLOUD_LATENCY_MS = 22 \* 60 \* 1000;/,
  "manca la latenza di diffusione: si chiederebbe uno slot inesistente");

// Risoluzione: mai piu' pixel di quanti ne ha lo strumento, e il riquadro
// segue la vista.
assert.match(html, /const native = Math\.ceil\(spanX \/ product\.metres\);/,
  "la richiesta non e' legata al pixel nativo dello strumento");
assert.match(html, /const CLOUD_MAX_SIDE = 4096;/,
  "la qualita' satellitare non sale fino a 4096 px per lato");
assert.match(html, /memory >= 8\s*\n\s*\? CLOUD_MAX_SIDE \* CLOUD_MAX_SIDE/,
  "i dispositivi potenti non ricevono la risoluzione satellitare massima");
assert.match(html, /const sideLimitedWidth = Math\.floor\(CLOUD_MAX_SIDE \/ Math\.max\(aspect, 1\)\);/,
  "il lato verticale del WMS puo' superare i 4096 px");
assert.match(html, /context\.imageSmoothingQuality = "high";/,
  "la ricampionatura delle immagini satellitari non e' in alta qualita'");
assert.match(html, /function cloudRequestBox\(pad\)/, "il riquadro non segue la vista");
// Il margine serve a non ricaricare a ogni panoramica: quindi il confronto
// deve usare il riquadro visibile, non quello gia' allargato.
assert.match(html, /cloudNeedsReload\(cloudRequestBox\(0\), size, slot\)/,
  "il margine e' inutile se si confronta il riquadro allargato");
// L'ascolto va registrato dove la mappa esiste: setupMap() gira dopo il
// blocco dei listener dell'interfaccia, e li' "map" e' ancora undefined.
assert.match(
  html,
  /map\.on\("load"[\s\S]*?map\.on\("moveend", scheduleCloudReload\);[\s\S]*?id: "satellite-clouds-layer"/,
  "l'ascolto del movimento non e' dentro il caricamento della mappa"
);
assert.match(html, /return cloudLoadedMetres > size\.metres \* 1\.35;/,
  "la soglia di ricarica non lascia raggiungere il pixel nativo");

// Le RGB ufficiali sono già diagnostiche calibrate: modificarne i colori
// distruggerebbe la legenda. Solo i canali singoli diventano maschere.
assert.match(html, /if \(product\.mode === "grey"\) \{/,
  "i canali IR e VIS non vengono trattati come maschere di nube");
assert.match(html, /mode: "diagnostic"/,
  "le RGB scientifiche non sono conservate come prodotti diagnostici");
// Di notte il visibile e' cieco: va detto invece di mostrare un livello vuoto.
assert.match(html, /if \(product\.dayOnly && pixels\) \{[\s\S]{0,420}?e' notte: il visibile non vede nulla/,
  "di notte il visibile resta vuoto senza spiegazione");
assert.match(html, /image\.crossOrigin = "anonymous";/,
  "senza CORS la canvas resta sporca e getImageData fallisce");
assert.match(html, /const alpha = level \* level \* \(3 - 2 \* level\);/,
  "manca la rampa di opacita' che rende trasparente il cielo sereno");
assert.match(html, /pixels\[i \+ 3\] = Math\.round\(alpha \* 255\);/,
  "l'opacita' calcolata non finisce nel canale alfa");
assert.match(html, /id: "satellite-clouds-layer"/, "layer delle nubi assente");
assert.match(
  html,
  /id: "satellite-clouds-layer"[\s\S]{0,400}?layout: \{ visibility: "none" \}/,
  "le nubi satellitari non sono spente all'avvio"
);
// Radar e satellite restano sincronizzati quando sono entrambi accesi, ma
// ciascuno ha il proprio comando: il radar può essere rimosso senza perdere
// l'immagine satellitare.
assert.match(html, /data-toggle="satclouds"[\s\S]{0,220}?<b>Satellite MTG<\/b>/,
  "manca il controllo satellitare indipendente");
assert.match(html, /data-toggle="radar"[\s\S]{0,220}?<b>Radar precipitazioni<\/b>/,
  "manca il controllo radar indipendente");
assert.match(html, /radar: showRadar,[\s\S]{0,120}?satclouds: showSatelliteClouds,/,
  "gli stati indipendenti non sono riportati nell'interfaccia");
assert.match(html, /} else if \(name === "satclouds"\) \{\s*\n\s*showSatelliteClouds = !showSatelliteClouds;/,
  "il satellite non si accende senza il radar");
assert.match(
  html,
  /cloudTimer = setInterval\(function \(\) \{ loadSatelliteClouds\(false\); \},\s*\n\s*\(product\.slotMs \|\| CLOUD_SLOT_MS\) \/ 2\);/,
  "le nubi non si aggiornano da sole: non sarebbero in tempo reale"
);
assert.match(html, /if \(typeof document\.hidden === "boolean" && document\.hidden\) return;/,
  "in secondo piano si continua a chiedere immagini al servizio");
// L'ordine conta: le nubi coprono il campo previsto, non i confini.
const cloudsAt = html.indexOf('id: "satellite-clouds-layer"');
const weatherAt = html.indexOf('id: "weather-layer"');
const regionsAt = html.indexOf('id: "regions-layer"');
assert.ok(weatherAt > 0 && cloudsAt > weatherAt && regionsAt > cloudsAt,
  "le nubi non stanno fra il campo del modello e i confini");

// Le nubi osservate si sovrappongono al campo previsto: i due devono
// raccontare lo stesso istante, altrimenti si leggerebbe la previsione di
// domani sotto le nubi di adesso.
assert.match(html, /function syncTimelineToSatellite\(slot\)/,
  "manca l'ancoraggio della timeline al satellite");
assert.match(html, /markCloudFreshness\(slot\);\s*\n\s*if \(showRadar\) syncRadarToSatellite\(slot\);\s*\n\s*syncTimelineToSatellite\(slot\);/,
  "radar e timeline non seguono il passaggio satellitare");
assert.match(html, /function radarFrameNearestTo\(when\)/,
  "manca la scelta del radar temporalmente piu' vicino al satellite");
assert.match(html, /latest\.path \+ "\/512\/\{z\}\/\{x\}\/\{y\}/,
  "la mappa principale non usa i tile radar RainViewer da 512 px");
assert.match(html, /source: "satellite",\s*\n\s*layout: \{ visibility: "visible" \}/,
  "il fondo fotografico satellitare non e' quello predefinito");
assert.match(html, /maxzoom: 7,/,
  "RainViewer riceverebbe ancora richieste oltre lo zoom nativo supportato");
assert.match(html, /function nearestStepTo\(when\)/,
  "manca la ricerca della scadenza piu' vicina all'immagine");
assert.match(html, /if \(target\.index !== currentIndex\) loadStep\(target\.index\);/,
  "la timeline non si sposta sulla scadenza del satellite");
assert.match(html, /if \(locked && isPlaying\) setPlaying\(false\);/,
  "con il livello acceso l'animazione continuerebbe a scorrere");
assert.match(html, /if \(ui\.slider\) ui\.slider\.disabled = locked;/,
  "il cursore del tempo resta manovrabile con le nubi accese");
assert.match(html, /if \(ui\.playButton\) ui\.playButton\.disabled = locked;/,
  "il tasto di riproduzione resta attivo con le nubi accese");
assert.match(html, /id="satellite-lock"/, "manca l'indicazione visibile dell'ancoraggio");
// L'attributo disabled ferma il dito, non un evento sintetico: la guardia
// vera sta nei gestori.
assert.match(
  html,
  /ui\.slider\.addEventListener\("input"[\s\S]{0,600}?if \(showSatelliteClouds\) \{[\s\S]{0,300}?ui\.slider\.value = String\(currentIndex\);/,
  "un evento sintetico sul cursore sfuggirebbe all'ancoraggio"
);
assert.match(
  html,
  /ui\.playButton\.addEventListener\("click"[\s\S]{0,300}?if \(showSatelliteClouds\) \{/,
  "il tasto di riproduzione sfuggirebbe all'ancoraggio"
);
assert.match(html, /if \(target\.distance > 90 \* 60 \* 1000\)/,
  "un modello che non copre l'ora del satellite passerebbe per una coincidenza");

// --- Sezione temporali ---
// I campi stanno in un file proprio, su griglia dimezzata: cercarli in
// currentData li dichiarava sempre assenti e l'interruttore non rispondeva.
assert.match(html, /if \(usesStormData\(key\)\) \{[\s\S]{0,300}?const item = catalog\[currentIndex\];/,
  "la disponibilita' dei campi temporaleschi non viene dal catalogo");
assert.match(html, /function stormSample\(name, gx, gy\)/,
  "manca il campionamento sulla griglia dei temporali");
assert.match(html, /const sx = \(lon - stormData\.lo1\) \/ stormData\.dx;/,
  "la griglia dei temporali non viene riproiettata: e' dimezzata, non uguale");
["storm_prob", "updraft", "cape"].forEach((key) => {
  assert.match(html, new RegExp(key + ":\\s*\\{[\\s\\S]{0,700}?storm:"),
    "il campo " + key + " non dichiara da quale griglia viene");
});

// Nel selettore pubblico deve comparire un solo prodotto temporalesco. Le
// variabili fisiche restano disponibili internamente alla catena diagnostica,
// ma non devono sembrare sei algoritmi concorrenti nella mappa.
const publicStormCards = [...html.matchAll(/data-layer="([^"]+)"/g)]
  .map((match) => match[1])
  .filter((key) => [
    "convection_prob", "storm_prob", "updraft", "cape", "trigger", "bowen"
  ].includes(key));
assert.deepEqual(publicStormCards, ["storm_prob"],
  "il selettore deve mostrare soltanto Algoritmo temporali");
assert.match(html,
  /data-layer="storm_prob"[\s\S]*?<b>Algoritmo temporali<\/b>/,
  "il solo prodotto pubblico deve dichiararsi come algoritmo unico");
// "Solo carta" e "Schermo intero" sono due azioni diverse: la prima spegne
// davvero i dati, la seconda nasconde soltanto l'interfaccia.
assert.match(html, /data-layer="none"[\s\S]{0,220}?<b>Solo carta<\/b>/,
  "manca il comando per una carta senza dati sovrapposti");
assert.match(html, /function clearMeteorologicalLayers\(\)[\s\S]{0,900}?"weather-layer", "visibility", "none"/,
  "Solo carta non spegne il raster meteorologico");
assert.match(html, /document\.body\.classList\.toggle\("map-only-layer", nextLayer === "none"\)/,
  "lo stato Solo carta non e' persistito nell'interfaccia");
assert.match(html, />Schermo intero<\/span>/,
  "il vecchio pulsante che nasconde solo la UI resta chiamato Solo mappa");
assert.match(html, /if \(item\.storm === false\) \{/,
  "senza il controllo sul catalogo si chiederebbe un file inesistente");
assert.match(html, /!upperActive && !stormActive && fieldGrid/,
  "la fusione con le stazioni si applicherebbe anche ai campi temporaleschi");

// La catena: sette anelli piu' i due rischi, e l'ultimo deve parlare la
// stessa lingua della probabilita' che sta sopra.
assert.match(html, /function updateStormChain\(gx, gy\)/, "manca la catena convettiva");
assert.match(html, /name: "Fulminazione · entro 10 km"/,
  "l'ultimo anello contraddirebbe la probabilita': LPI e' puntuale, lei no");
assert.match(html, /state: stormLinkState\(probability, 10, 40\)/,
  "l'ultimo anello non usa la probabilita' di vicinato");
// La trappola misurata: dentro un nucleo maturo la CAPE e' gia' consumata.
assert.match(html, /const insideCore = Number\.isFinite\(updraft\) && updraft >= 5\s*\n\s*&& Number\.isFinite\(cape\) && cape < 500;/,
  "manca l'avviso sulla CAPE letta dentro una cella gia' matura");
assert.match(html, /updateStormChain\(point\.gx, point\.gy\);/,
  "la catena non viene aggiornata con il punto selezionato");

// --- Innesco dal terreno ---
["trigger", "bowen"].forEach((key) => {
  assert.match(html, new RegExp(key + ":\\s*\\{[\\s\\S]{0,700}?storm:"),
    "il campo " + key + " non dichiara da quale griglia viene");
});
// La brezza e' pubblicata moltiplicata per 1e5: confrontarla con la soglia
// non scalata la faceva vincere sempre, e sulle Alpi usciva "brezza di mare".
assert.match(html, /const breezeShare = Number\.isFinite\(breeze\) \? breeze \/ 20\.0 : 0;/,
  "la brezza va normalizzata sulla scala con cui e' pubblicata");
assert.match(html, /upslopeShare >= 0\.25 && upslopeShare >= breezeShare/,
  "i due meccanismi non vengono confrontati sulla stessa scala");
assert.match(html, /"Innesco · brezza di mare o lago"/,
  "la maschera ICON non distingue mare e laghi: il nome non deve deciderlo");
assert.match(html, /name: "Suolo · rapporto di Bowen"/,
  "manca l'anello del suolo nella catena");

// --- Simbologia frontale WMO ---
// Il lato dei simboli non e' decorativo: dice da che parte si muove il
// fronte. L'analyzer orienta ogni linea pubblicata con l'aria calda a
// sinistra, quindi nel sistema locale del simbolo -y e' il lato caldo e +y
// il freddo. Da li' discende tutto il resto, e un'inversione accidentale
// scambierebbe un fronte freddo con uno caldo senza rompere nulla.
const triangle = html.match(/function drawFrontTriangle\([\s\S]*?\n {6}\}/);
assert.ok(triangle, "simbolo del fronte freddo assente");
assert.match(triangle[0], /lineTo\(0, -10\.5 \* s\)/,
  "il triangolo non punta piu' verso il lato caldo (-y)");
const warmArc = html.match(/function drawFrontSemicircle\([\s\S]*?\n {6}\}/);
assert.ok(warmArc, "simbolo del fronte caldo assente");
assert.match(warmArc[0], /arc\(0, 0, 7 \* s, 0, Math\.PI, false\)/,
  "il semicerchio caldo non sporge piu' verso il lato freddo (+y)");
const occludedArc = html.match(/function drawOccludedSemicircle\([\s\S]*?\n {6}\}/);
assert.ok(occludedArc, "simbolo del fronte occluso assente");
assert.match(occludedArc[0], /arc\(0, 0, 7 \* s, Math\.PI, 2 \* Math\.PI, false\)/,
  "sul fronte occluso i due simboli devono stare dalla stessa parte");
const styled = html.match(/function drawFrontStyled\([\s\S]*?\n {6}\}\n/);
assert.ok(styled, "disegno del fronte assente");
// Stazionario: triangolo e semicerchio su lati OPPOSTI, quindi deve usare
// il semicerchio caldo e non quello dell'occluso.
assert.match(styled[0], /frontType === "stationary" && index % 2 === 0[\s\S]{0,220}?drawFrontSemicircle/,
  "il fronte stazionario non alterna i due simboli su lati opposti");
assert.match(styled[0], /drawOccludedSemicircle\(x, y, angle, "#8b3fd0"/,
  "il fronte occluso non usa il semicerchio dallo stesso lato del triangolo");

// La scala cartografica lega dimensione dei simboli, spessore della linea e
// passo fra i simboli: se crescessero separatamente i simboli si
// sovrapporrebbero in una fascia continua.
assert.match(html, /function frontDrawScale\(\)/,
  "manca la scala cartografica dei fronti");
assert.match(styled[0], /const spacing = \(window\.innerWidth < 900 \? 48 : 54\) \* s;/,
  "il passo fra i simboli non segue la loro dimensione");
assert.match(styled[0], /strokeFrontLine\(points, frontType, alpha, s\)/,
  "lo spessore della linea non segue la scala");

console.log("3D map regression checks: OK");
