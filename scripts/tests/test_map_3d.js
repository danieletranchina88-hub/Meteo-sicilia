#!/usr/bin/env node
"use strict";

// Regression checks for the static GitHub Pages client. They intentionally do
// not require WebGL, so they also run on the standard GitHub Actions runner.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "../..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");

const inlineScripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)]
  .map((match) => match[1])
  .filter((source) => source.trim());
assert.ok(inlineScripts.length > 0, "script applicativo assente");
inlineScripts.forEach((source) => {
  assert.doesNotThrow(() => new Function(source), "JavaScript inline non valido");
});

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

// Campo di frecce fitto e regolare, con dimensione quasi uniforme: a quella
// densita' l'intensita' la porta il colore, non la lunghezza.
assert.match(html, /const denseSpacing = \(mobile \? 27 : 31\)/,
  "il campo di frecce non e' fitto");
assert.match(html, /vectorElevated \? 1\.55 : 1/,
  "in 3D il passo delle frecce non si allarga: ricavare la posizione costa di piu'");
// La lunghezza deve variare molto: cortissima con la calma, oltre il passo
// della griglia con vento forte, cosi' le frecce si fondono in linee continue
// lungo il flusso. E' quello a dare l'aspetto di corrente.
assert.match(html, /const length = clamp\(5 \+ speed \* 0\.46/,
  "la lunghezza delle frecce non segue abbastanza la velocita'");

// Fondo scuro: il rilievo va invertito, altrimenti le ombre non hanno contro
// cosa staccare e le montagne spariscono. Su scuro sono i versanti illuminati
// ad accendersi, che e' anche piu' fedele alla luce solare reale.
assert.match(html, /const onDark = showDarkBase && !showSatellite;/,
  "il rilievo non distingue il fondo scuro");
assert.match(html, /onDark\s*\n?\s*\? \(0\.16 \+ 0\.42 \* direct\)/,
  "su fondo scuro i versanti illuminati non si accendono");
assert.match(html, /data-toggle="darkbase"/, "interruttore fondo scuro assente");
assert.match(html, /id: "basemap-dark-layer"/, "base scura assente");

console.log("3D map regression checks: OK");
