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
assert.doesNotMatch(radarHtml, /Osservazioni sincronizzate · tempo reale/,
  "la pagina osservativa promette tempo reale senza dichiarare la latenza");
assert.match(radarHtml, /Radar composito \+ Meteosat MTG/,
  "le due sorgenti osservate non sono identificate con precisione");
assert.match(radarHtml, /EUMETSAT · radar composito RainViewer/,
  "mancano le fonti visibili di satellite e radar");
assert.match(radarHtml, /function observationTime\(value\)/,
  "i frame osservativi non espongono data e fuso del timestamp");
assert.doesNotMatch(html, /radar in tempo reale/i,
  "la mappa promette tempo reale per un mosaico osservato con latenza");
assert.match(html, /const FUSION_TIME_TOLERANCE_MS = 45 \* 60 \* 1000;/,
  "la fusione METAR non impone una tolleranza temporale dichiarata");
assert.match(html, /function stationMatchesValidTime\(station\)/,
  "le stazioni METAR possono ancora correggere scadenze temporalmente lontane");
assert.match(html, /Math\.abs\(station\.obsTime - validTime\) <= FUSION_TIME_TOLERANCE_MS/,
  "il valid time ICON non viene confrontato con l'ora della singola stazione");
assert.match(html, /MSLP solo SLP/,
  "l'interfaccia confonde ancora altimeter setting e pressione MSLP");
assert.match(html, /data-toggle="stations"[\s\S]{0,220}?rete METAR Italia/,
  "manca il controllo della rete nazionale delle stazioni");
assert.match(html, /let stationNetwork = \[\];/,
  "il client non conserva il catalogo stazioni separato dai report live");
assert.match(html, /payload\.stationNetwork[\s\S]{0,180}?stationNetwork\.stations/,
  "il sito non legge il catalogo nazionale pubblicato dalla pipeline");
assert.match(html, /cluster: true,[\s\S]{0,100}?clusterMaxZoom: 8/,
  "la rete nazionale non viene raggruppata alle piccole scale");
assert.match(html, /report recente[\s\S]{0,100}?report ritardato[\s\S]{0,120}?nessun report ricevuto/,
  "gli stati delle stazioni non sono dichiarati in modo scientificamente chiaro");
assert.match(html, /SLP e QNH sono grandezze distinte/,
  "il dettaglio stazione non distingue SLP e QNH");

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
// Lo zoom iniziale adatta il dominio alla fascia UTILE, non all'altezza
// intera: intestazione e timeline sono opachi, e quello che finisce sotto di
// loro e' come se non ci fosse. Un centro barico sul bordo sud veniva
// disegnato a 816 pixel su 900, cioe' dietro la timeline.
assert.match(html, /const usableHeight = Math\.max\(height - topPanel - bottomPanel, 120\);/,
  "lo zoom iniziale non tiene conto dei pannelli opachi");
assert.match(html, /const zoomY = Math\.log2\(usableHeight \/ \(512 \* latitudeFraction\)\);/,
  "lo zoom iniziale non adatta tutto il dominio alla fascia utile");
assert.match(html, /\(\(topPanel - bottomPanel\) \/ 2 \/ worldPx\) \* 2 \* Math\.PI/,
  "il centro non si sposta per compensare i pannelli");
// I limiti di trascinamento sono il dominio ALLARGATO quanto basta perche'
// tutto il dominio sia raggiungibile: con limiti esattamente uguali al
// dominio, su uno schermo largo MapLibre stringe la vista per far entrare la
// longitudine e le fasce sud e nord restano irraggiungibili.
assert.match(html, /minZoom: startView\.zoom,\s*\n\s*maxBounds: paddedModelBounds\(startView\),/,
  "zoom-out o trascinamento possono ancora uscire dal dominio del modello");
// Due zoom: si parte pieni, ma non si scende sotto quello che fa entrare tutto
// il dominio. Il dominio e' quasi quadrato e uno schermo e' largo: aprirsi
// gia' inquadrati lascia due fasce vuote ai lati.
assert.match(html, /zoom: startView\.fillZoom,/,
  "la vista iniziale non riempie lo schermo");
assert.match(html, /const fill = Math\.max\(zoomX, zoomY\);/,
  "manca lo zoom che riempie");
assert.match(html, /function paddedModelBounds\(/,
  "i limiti non tengono conto della forma dello schermo");
assert.match(html, /map\.setMaxBounds\(paddedModelBounds\(view\)\);/,
  "dopo un ridimensionamento i limiti tornano a tagliare il dominio");
assert.match(html, /map\.resize\(\);\s*\n\s*lockMapToModelDomain\(false\);/,
  "ruotando il telefono il limite del dominio non viene ricalcolato");
assert.match(meteogramHtml, /id="weather-strip"/,
  "manca la sintesi visuale ogni tre ore");
assert.match(meteogramHtml, /label:"Coerenza interna", values:s\.stormConfidence/,
  "lo score temporalesco e' mostrato senza coerenza interna");
assert.match(meteogramHtml, /label:"Raffica", values:s\.gust10/,
  "il meteogramma del vento ignora le raffiche");

// Il sito deve distinguere variabili, diagnostiche e probabilita' calibrate.
assert.match(html, /id="field-method-card" class="field-method-card" open/,
  "manca la scheda scientifica del campo attivo");
assert.match(html, /const LAYER_SCIENCE = \{/,
  "i layer non dichiarano natura, metodo, scala e limite d'uso");
const scienceCatalog = html.match(
  /const LAYER_SCIENCE = \{([\s\S]*?)\n      \};\n\n      const MODEL_INFO/
);
assert.ok(scienceCatalog, "catalogo scientifico non delimitabile");
[
  "temp", "feels", "wind", "gust", "rain", "cloud", "rh", "wetbulb",
  "press", "theta850", "geopot500", "t850", "storm_prob", "visibility",
  "freezing_rain", "foehn"
].forEach((key) => {
  assert.match(scienceCatalog[1], new RegExp(
    key + ":\\s*\\{[\\s\\S]{0,900}?nature:[\\s\\S]{0,900}?method:"
      + "[\\s\\S]{0,900}?scale:[\\s\\S]{0,900}?limit:"
  ), "scheda scientifica incompleta per " + key);
});
assert.match(html, /Score diagnostico deterministico non calibrato/,
  "lo score temporalesco non dichiara la propria semantica");
assert.doesNotMatch(html, />probabilità fisica · entro 10 km</,
  "uno score deterministico viene ancora presentato come probabilita' fisica");
assert.doesNotMatch(html, /Rotazione · elicità/,
  "UH_MAX viene ancora confusa con la SRH ambientale");
assert.match(html, /Updraft helicity · UH_MAX/,
  "il campo UH_MAX non e' identificato in modo scientifico");
assert.match(html, /const ISOBAR_INTERVAL_HPA = 4;/,
  "le isobare non rispettano il passo sinottico di 4 hPa");
assert.match(html, /const ISOBAR_MAJOR_EVERY = 8;/,
  "le isobare principali non rispettano il passo di 8 hPa");
assert.match(html, /passo 4 hPa · principali ogni 8/,
  "l'interfaccia contraddice il passo reale delle isobare");

// Il bollettino generale e quello puntuale sono prodotti distinti. L'analisi
// esperta deve seguire la timeline e non richiedere la selezione di un punto.
assert.match(html, /id="bulletin-mode-general"[^>]*class="bulletin-tab active"/,
  "l'analisi generale non e' la vista predefinita del bollettino");
assert.match(html, /id="bulletin-mode-point"/,
  "manca il prodotto puntuale separato");
assert.match(html, /let bulletinMode = "general";/,
  "lo stato JavaScript non parte dall'analisi generale");
assert.match(html, /expert_bulletin\.json\.gz/,
  "il sito non carica il prodotto sinottico dell'intero run");
assert.match(html, /candidate\.leadHours\) === lead/,
  "il bollettino generale non segue il forecast hour della timeline");
assert.match(html, /function renderExpertSection\(section\)/,
  "le sezioni tecniche del motore non vengono rappresentate");
assert.match(html, /Perché\? /,
  "manca la catena causale verificabile nel bollettino");
assert.match(html, /docs\/motore_bollettino\.md/,
  "l'interfaccia non collega la documentazione del motore esperto");
assert.match(html, /ai_expert_bulletin\.json\.gz/,
  "il sito non tenta di caricare la sintesi IA verificata");
assert.match(html, /aiBulletin\.runTime === bulletin\.runTime/,
  "una sintesi IA di un altro run potrebbe essere mostrata");
assert.match(html, /function renderAiAgentBrief\(aiBulletin, leadHours\)/,
  "la sintesi IA non viene rappresentata con il proprio periodo");
assert.match(html, /Prove utilizzate · /,
  "i claim IA non espongono le prove citate");
assert.match(html, /docs\/agente_meteorologico\.md/,
  "l'interfaccia non collega il protocollo verificabile dell'agente");
assert.doesNotMatch(html, /GEMINI_API_KEY|GROQ_API_KEY/,
  "un nome di segreto API è comparso nel frontend pubblico");

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

// --- Il fondo cartografico e' nostro ---------------------------------------
// Un fondo raster sta SOTTO il campo colorato: quando il campo si accende la
// costa sparisce proprio dove serve. Queste sono linee, e si disegnano SOPRA.
assert.match(html, /const MAP_LAND = "#e7e5d9";/, "colore della terra assente");
assert.match(html, /const MAP_SEA = "#9dc0de";/, "colore del mare assente");
["land", "sea", "coast", "borders", "lakes", "rivers"].forEach((tema) => {
  assert.ok(html.includes('data: "data_base/' + tema + '.json"'),
    "il fondo non carica il tema: " + tema);
});
assert.doesNotMatch(html, /basemaps\.cartocdn\.com/,
  "il fondo dipende ancora dalle tessere di un fornitore esterno");
assert.doesNotMatch(html, /raw\.githubusercontent\.com/,
  "i confini regionali arrivano ancora dal repository di un terzo");
// L'ordine e' il punto: l'inchiostro va aggiunto dopo il livello meteo.
assert.ok(
  html.indexOf('id: "weather-layer"') < html.indexOf('id: "base-coast-layer"'),
  "la costa e' sotto il campo colorato: sparirebbe appena si accende un campo"
);
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

// I toponimi si disegnano sulla stessa tela di isobare, fronti e cartiglio:
// niente server di glifi, niente font da scaricare, e il diradamento per zoom
// lo decidiamo noi invece di subirlo.
assert.match(html, /function drawPlaceNames\(/, "toponimi sul canvas assenti");
assert.match(html, /data_base\/places\.json/, "i nomi dei luoghi non vengono caricati");
assert.doesNotMatch(html, /CARTO_LABEL_LAYERS/,
  "sono tornati i diciotto livelli di etichette del fornitore esterno");
assert.doesNotMatch(html, /glyphs:/,
  "e' tornato il server di glifi esterno");
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

// Le linee di corrente vengono proiettate con la stessa matrice economica
// usata dalle particelle e restano sotto isobare, fronti e toponimi.
assert.match(html, /const seedPoint = projectParticle\(location\.lng, location\.lat, altitude\);/,
  "le streamline non usano la proiezione vettoriale ottimizzata");
assert.doesNotMatch(html, /const seedPoint = map\.project/,
  "le streamline interrogano ancora il DEM attraverso map.project");
assert.ok(
  drawVectors[0].indexOf("drawWindStreamlines(wind)")
    < drawVectors[0].indexOf("drawSynopticCanvas()"),
  "le streamline coprono l'inchiostro dell'analisi sinottica"
);

// La carta chiara e' il fondo, e basta. Rilievo e terreno reale sono due
// scelte esplicite, spente all'avvio.
assert.match(html, /let showTerrain = false;/,
  "il rilievo e' ancora attivo all'avvio");
assert.doesNotMatch(html, /let showLightBase/,
  "esiste ancora una seconda base da cui distinguere quella chiara");
assert.doesNotMatch(html, /const onDark =/,
  "il codice distingue ancora un fondo scuro che non esiste piu'");
// Niente piu' simulazione del Sole: il rilievo ha una luce cartografica
// fissa da nord-ovest, uguale a qualunque ora della previsione.
assert.match(html, /const RELIEF_ILLUMINATION = 315;/,
  "manca la luce cartografica fissa del rilievo");
assert.match(html, /"hillshade-illumination-direction", RELIEF_ILLUMINATION/,
  "il rilievo non usa la direzione di luce fissa");
// Il fondo e' sempre chiaro: il rilievo si legge in un modo solo, scurendo i
// versanti in ombra. La taratura invertita serviva alla base scura, sparita.
assert.match(html, /"hillshade-highlight-color",\s*\n?\s*"rgba\(255,250,238,0\.18\)"/,
  "il rilievo non e' tarato sul fondo chiaro");
assert.doesNotMatch(html, /solarPosition|SOLAR_CENTER|updateSolarLighting/,
  "la simulazione della luce solare e' tornata");
assert.doesNotMatch(html, /updateSkyPalette\(day/,
  "il cielo dipende ancora dall'altezza del Sole");
// Una sola carta, e la fotografia come alternativa spenta di default.
assert.doesNotMatch(html, /data-toggle="lightbase"/,
  "e' tornato l'interruttore del fondo chiaro, che ora e' l'unico fondo");
assert.doesNotMatch(html, /basemap-dark-layer/, "e' tornata la base scura");
assert.match(html, /let showSatellite = false;/,
  "il terreno reale e' di nuovo acceso di default");
assert.match(html, /id: "satellite-layer"[\s\S]*?layout: \{ visibility: "none" \}/,
  "il terreno reale non e' spento nello stile iniziale");
assert.match(html, /id: "terrain-base"[\s\S]*?layout: \{ visibility: "none" \}/,
  "il rilievo raster non e' spento nello stile iniziale");
assert.match(html, /id: "terrain-detail"[\s\S]*?layout: \{ visibility: "none" \}/,
  "il rilievo DEM non e' spento nello stile iniziale");
assert.match(html, /#map\s*\{[\s\S]*?background: #9dc0de;/,
  "prima del caricamento il fondo non e' gia' quello della carta");

// Interfaccia: i pannelli galleggiano sulla mappa, quindi il campo di vento
// arriva ai bordi dello schermo invece di lasciare fasce vuote.
assert.match(
  html,
  /for \(let row = 0, y = seedSpacing \* 0\.5; y < height;/,
  "i semi delle streamline non coprono tutto il riquadro"
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

// Nubi osservate dal satellite con timestamp dichiarato: una sola immagine sul dominio,
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
// Il terreno reale non e' piu' il fondo predefinito: la carta lo e'.
assert.match(html, /source: "satellite",\s*\n\s*layout: \{ visibility: "none" \}/,
  "il terreno reale e' di nuovo il fondo predefinito");
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
// stessa lingua dello score di vicinato che sta sopra.
assert.match(html, /function updateStormChain\(gx, gy\)/, "manca la catena convettiva");
assert.match(html, /name: "Score temporalesco · entro 10 km"/,
  "l'ultimo anello contraddirebbe lo score: LPI e' puntuale, lo score no");
assert.match(html, /state: stormLinkState\(probability, 10, 40\)/,
  "l'ultimo anello non usa lo score di vicinato");
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
// Il colore arriva da frontInk, perche' sulla carta gli inchiostri sono altri:
// qui conta che il fronte occluso usi il semicerchio dallo stesso lato, non
// con quale tinta lo disegni.
assert.match(styled[0], /drawOccludedSemicircle\(x, y, angle, frontInk\("occluded"\)/,
  "il fronte occluso non usa il semicerchio dallo stesso lato del triangolo");

// La scala cartografica lega dimensione dei simboli, spessore della linea e
// passo fra i simboli: se crescessero separatamente i simboli si
// sovrapporrebbero in una fascia continua.
assert.match(html, /function frontDrawScale\(\)/,
  "manca la scala cartografica dei fronti");
assert.match(styled[0], /const spacing = \(isMobile\(\) \? 48 : 54\) \* s;/,
  "il passo fra i simboli non segue la loro dimensione");
assert.match(styled[0], /strokeFrontLine\(points, frontType, alpha, s\)/,
  "lo spessore della linea non segue la scala");

// --- Il disegno dei fronti deve terminare, a ogni zoom -----------------------
// Le altre verifiche di questo file leggono il sorgente; questa lo ESEGUE,
// perche' il difetto che ha bloccato i telefoni era invisibile a una regex.
// Le bande del fronte stazionario accumulavano una distanza e ne facevano il
// modulo sulla lunghezza di banda: con una banda di 21,84 px la distanza
// arrivava a 65,52, il resto usciva 21,84 meno 3,6e-15, e il passo successivo
// valeva 3,6e-15 -- troppo piccolo per cambiare un numero di grandezza 65. Il
// ciclo avanzava di nulla, per sempre.
function frontDrawingApi(viewportWidth, viewportHeight, zoom, budget) {
  const names = ["walkFront", "drawFrontTriangle", "drawFrontSemicircle",
                 "drawOccludedSemicircle", "strokeFrontLine", "frontSlice",
                 "frontDrawScale", "drawFrontStyled", "frontInk",
                 "frontHaloColour", "isMobile"];
  // Le tavolozze sono dati, non funzioni, ma il disegno non gira senza: le
  // prendo dal sorgente cosi' il test non ne tiene una copia che puo'
  // divergere da quella vera.
  const breakpoint = html.match(/const MOBILE_BREAKPOINT = \d+;/);
  assert.ok(breakpoint, "confine unico fra telefono e scrivania assente");
  const tables = [breakpoint[0]].concat(["FRONT_SCREEN_INK", "FRONT_PAPER_INK"].map((name) => {
    const found = html.match(new RegExp("const " + name + " = \\{[\\s\\S]*?\\n {6}\\};"));
    assert.ok(found, "tavolozza dei fronti assente: " + name);
    return found[0];
  })).join("\n");
  const source = tables + "\n" + names.map((name) => {
    const found = html.match(new RegExp("function " + name + "\\([\\s\\S]*?\\n {6}\\}"));
    assert.ok(found, "funzione di disegno assente: " + name);
    return found[0];
  }).join("\n");

  const calls = { stroke: 0, fill: 0 };
  const ctx = new Proxy({}, {
    get(_target, property) {
      if (property === "stroke" || property === "fill") {
        return function () {
          calls[property] += 1;
          // Un ciclo che non avanza si manifesta qui: senza questo tetto il
          // test non fallirebbe, si bloccherebbe come il telefono.
          if (calls.stroke + calls.fill > budget) {
            throw new Error("BUDGET");
          }
        };
      }
      if (property === "setLineDash" || property === "save" || property === "restore"
          || property === "beginPath" || property === "moveTo" || property === "lineTo"
          || property === "arc" || property === "translate" || property === "rotate"
          || property === "closePath") {
        return function () {};
      }
      return undefined;
    },
    set() { return true; },
  });

  const factory = new Function(
    "vectorContext", "map", "clamp", "window", "synopticChart",
    source + "\nreturn { drawFrontStyled: drawFrontStyled, frontDrawScale: frontDrawScale };"
  );
  const api = factory(
    ctx,
    { getZoom: () => zoom },
    (v, a, b) => Math.min(Math.max(v, a), b),
    { innerWidth: viewportWidth, innerHeight: viewportHeight },
    false
  );
  return { api, calls };
}

// Un fronte come quelli pubblicati: una dozzina di vertici su ~7 gradi.
const frontLonLat = [];
for (let i = 0; i < 12; i += 1) {
  frontLonLat.push([6.0 + i * 0.63, 44.0 + Math.sin(i * 0.7) * 0.8]);
}

["cold", "warm", "stationary", "occluded", "uncertain"].forEach((frontType) => {
  let previous = null;
  [4, 8, 12, 16].forEach((zoom) => {
    // Viewport da telefono: e' li' che il blocco e' stato osservato.
    const { api, calls } = frontDrawingApi(390, 844, zoom, 20000);
    const worldPx = 512 * Math.pow(2, zoom);
    const points = frontLonLat.map(([lon, lat]) => ({
      x: (lon - frontLonLat[0][0]) / 360 * worldPx + 195,
      y: -(lat - frontLonLat[0][1]) / 360 * worldPx + 422,
    }));
    assert.doesNotThrow(
      () => api.drawFrontStyled(points, frontType, 0.95, api.frontDrawScale()),
      `il disegno del fronte ${frontType} non termina a zoom ${zoom}`
    );
    const total = calls.stroke + calls.fill;
    // Il costo non deve crescere con lo zoom: la linea si allunga in pixel,
    // ma la parte visibile no.
    if (previous !== null) {
      assert.ok(total <= previous * 3 + 24,
        `il costo del fronte ${frontType} cresce con lo zoom (${previous} -> ${total})`);
    }
    previous = total;
  });
});

// Un passo non valido non deve mai diventare un ciclo che non avanza.
[0, -5, NaN, Infinity].forEach((spacing) => {
  const { api } = frontDrawingApi(390, 844, 6, 5000);
  const straight = [{ x: 0, y: 0 }, { x: 4000, y: 0 }];
  assert.doesNotThrow(
    () => api.drawFrontStyled(straight, "cold", 1, spacing / 54),
    `passo ${spacing} non gestito`
  );
});

// --- Gerarchia dell'interfaccia sul telefono --------------------------------
// La mappa e' il contenuto; tutto il resto e' cornice. Queste verifiche
// difendono la gerarchia, non l'estetica: ogni pannello permanente in piu'
// toglie mappa su uno schermo da 390 px.

// Un solo confine fra telefono e scrivania, letto sia dal CSS sia dal JS.
assert.match(html, /const MOBILE_BREAKPOINT = 960;/,
  "il confine fra telefono e scrivania non e' una costante");
assert.doesNotMatch(html, /window\.innerWidth < 900/,
  "e' tornato il secondo confine a 900 px nel JavaScript");
assert.doesNotMatch(html, /@media \(min-width: 720px\)/,
  "e' tornato il gradino intermedio a 720 px nel foglio di stile");

// Tre sole azioni fisse nell'intestazione: pannello, bollettino, altro. Il
// resto sta nel menu, che sulla scrivania torna una fila in linea.
assert.match(html, /id="header-extra"/,
  "le azioni secondarie non sono raccolte in un blocco unico");
assert.match(html, /id="more-button"/, "manca il menu delle azioni secondarie");
assert.match(html, /body\.more-open #header-extra \{\s*display: flex;/,
  "il menu secondario non si apre");
assert.match(html, /#drawer-button,\s*\n\s*#more-button \{\s*\n\s*display: none;/,
  "sulla scrivania il menu secondario non sparisce");
// Un solo pannello pesante alla volta.
const moreMenu = html.match(/function setMoreMenu\([\s\S]*?\n {6}\}/);
assert.ok(moreMenu, "gestione del menu secondario assente");
assert.match(moreMenu[0], /setDrawer\(false\)/,
  "aprire il menu non chiude il pannello dei livelli");
assert.match(moreMenu[0], /setBulletin\(false\)/,
  "aprire il menu non chiude il bollettino");

// La scheda run/validita' e' la stessa cosa che dice l'intestazione.
assert.match(html, /id="brand-meta"/,
  "manca la riga compatta di corsa e validita' nell'intestazione");
assert.match(html, /@media \(max-width: 959px\) \{\s*\n\s*#meta-card \{\s*\n\s*display: none;/,
  "sul telefono la scheda dei metadati occupa ancora una fascia di mappa");

// Il riquadro del punto nasce compatto: valore, posizione, una riga.
assert.match(html, /id="readout-details"/,
  "il dettaglio del punto non e' separato dalla sintesi");
assert.match(html, /#readout-card\.expanded #readout-details \{\s*\n\s*display: block;/,
  "il dettaglio del punto non si apre a richiesta");
const readoutClose = html.match(/function closeReadout\([\s\S]*?\n {6}\}/);
assert.match(readoutClose[0], /classList\.remove\("expanded"\)/,
  "il riquadro del punto resta espanso per il punto successivo");
// La sintesi deve restare una riga: il fronte entra solo se e' vicino.
assert.match(html, /function nearestFrontSummary\(/,
  "la sintesi del punto usa il testo lungo del fronte");

// Legenda a pastiglia sul telefono.
assert.match(html, /#legend-card:not\(\.expanded\) #legend-note \{\s*\n\s*display: none;/,
  "la legenda sul telefono non e' ridotta a pastiglia");
assert.match(html, /body\.readout-open #legend-card \{\s*\n\s*pointer-events: none;/,
  "la pastiglia sbiadita intercetta ancora i tocchi del riquadro del punto");

// Il pannello dei livelli e' organizzato, non un catalogo.
assert.match(html, /<h3>Campo<\/h3>/, "manca la sezione Campo");
assert.match(html, /<h3>Analisi<\/h3>/, "manca la sezione Analisi");
assert.match(html, /<h3>Punto e strumenti<\/h3>/, "manca la sezione Punto e strumenti");
assert.match(html, /<details class="drawer-more">/,
  "gli interruttori rari non sono raccolti in Altro");
["isotherms", "isohypses", "satclouds", "terrain", "satellite",
 "threed", "graticule", "raw"].forEach((rare) => {
  const more = html.match(/<details class="drawer-more">[\s\S]*?<\/details>/);
  assert.ok(more[0].includes('data-toggle="' + rare + '"'),
    "e' tornato in prima vista un interruttore raro: " + rare);
});
const analysisSection = html.match(/<h3>Analisi<\/h3>[\s\S]*?<details class="drawer-more">/);
assert.ok(analysisSection && analysisSection[0].includes('data-toggle="fusion"'),
  "il downscaling richiesto non e' visibile nella sezione Analisi");

// Streamline e particelle sono due disegni dello stesso campo u/v ma
// rispondono a domande diverse -- geometria istantanea e animazione -- quindi
// restano due interruttori indipendenti. Le particelle non si accendono da
// sole: sopra le streamline ridurrebbero chiarezza e prestazioni.
assert.match(html, /data-toggle="vectors"/, "manca l'interruttore dei vettori");
assert.match(html, /data-toggle="flow"/, "manca l'interruttore delle particelle");
assert.doesNotMatch(html, /data-toggle="windanim"/,
  "i due interruttori del vento sono di nuovo uno solo");
const windToggle = html.match(/if \(name === "vectors"\)[\s\S]*?\} else if \(name === "isotherms"\)/);
assert.ok(windToggle, "i due interruttori del vento non sono gestiti");
assert.match(windToggle[0], /name === "flow"/, "le particelle non hanno un ramo proprio");
const windLayer = html.match(/if \(nextLayer === "wind" && !synopticChart\)[\s\S]*?\n {8}\}/);
assert.ok(windLayer, "scelta del campo vento assente");
assert.match(windLayer[0], /showVectors = true/,
  "il campo vento non attiva le linee di corrente");
assert.doesNotMatch(windLayer[0], /showParticles = true/,
  "il campo vento accende ancora automaticamente le particelle");

// Le linee non sono segmenti indipendenti: vengono integrate in entrambe le
// direzioni con RK2 e passo metrico dipendente dalla risoluzione della vista.
assert.match(html, /<b>Linee di corrente<\/b>/,
  "il controllo del vento continua a promettere semplici vettori");
assert.match(html, /const STREAMLINE_CONFIG = \{/,
  "configurazione delle linee di corrente assente");
const streamlineStep = html.match(/function advanceWindStreamline\([\s\S]*?\n {6}\}/);
assert.ok(streamlineStep, "integratore delle linee di corrente assente");
assert.match(streamlineStep[0], /stepMetres \* 0\.5/,
  "l'integratore non valuta il vento al punto medio");
assert.match(streamlineStep[0], /const middle = sampleWindVector/,
  "l'integratore e' tornato al passo di Eulero");
const streamlineRenderer = html.match(/function drawWindStreamlines\([\s\S]*?\n {6}\}/);
assert.ok(streamlineRenderer, "renderer delle linee di corrente assente");
assert.match(streamlineRenderer[0], /const backward = trace\(-1\)\.reverse\(\)/,
  "le linee non vengono integrate controvento");
assert.match(streamlineRenderer[0], /const forward = trace\(1\)/,
  "le linee non vengono integrate sottovento");
assert.match(streamlineRenderer[0], /new Uint8Array\(occupancyColumns \* occupancyRows\)/,
  "manca il controllo di densita' e sovrapposizione");
assert.match(streamlineRenderer[0], /strokeStreamlineHeads/,
  "le linee non mostrano il verso del moto");
assert.doesNotMatch(streamlineRenderer[0], /colorFor\(/,
  "la velocita' viene codificata due volte anche sulle linee");

// Verifica numerica dell'integratore su campi noti. In un vento uniforme il
// passo deve avere verso e distanza corretti; in un vortice il punto medio RK2
// deve iniziare a curvare gia' nel primo passo, cosa che Eulero non farebbe.
{
  function functionSource(name) {
    const start = html.indexOf("function " + name + "(");
    assert.ok(start >= 0, "funzione non trovata: " + name);
    const open = html.indexOf("{", start);
    let depth = 0;
    for (let index = open; index < html.length; index += 1) {
      if (html[index] === "{") depth += 1;
      if (html[index] === "}") depth -= 1;
      if (depth === 0) return html.slice(start, index + 1);
    }
    throw new Error("funzione non delimitabile: " + name);
  }

  const numerical = new Function(
    "const STREAMLINE_CONFIG={minSpeedMps:0.22,screenStepPx:4.8," +
      "minStepMetres:650,maxStepMetres:30000};\n" +
    "function clamp(v,a,b){return Math.max(a,Math.min(b,v));}\n" +
    "function sampleBilinear(array,gx,gy,nx,ny){" +
      "const x=Math.max(0,Math.min(nx-1,gx));" +
      "const y=Math.max(0,Math.min(ny-1,gy));" +
      "const x0=Math.max(0,Math.min(nx-2,Math.floor(x)));" +
      "const y0=Math.max(0,Math.min(ny-2,Math.floor(y)));" +
      "const fx=x-x0,fy=y-y0,i=y0*nx+x0;" +
      "return array[i]*(1-fx)*(1-fy)+array[i+1]*fx*(1-fy)+" +
        "array[i+nx]*(1-fx)*fy+array[i+nx+1]*fx*fy;}\n" +
    functionSource("sampleWindVector") + "\n" +
    functionSource("offsetWindLocation") + "\n" +
    functionSource("advanceWindStreamline") + "\n" +
    functionSource("streamlineStepMetres") + "\n" +
    "return {advanceWindStreamline,streamlineStepMetres};"
  )();

  const uniformMeta = { lo1: 0, la1: 10, dx: 1, dy: 1, nx: 11, ny: 11 };
  const east = {
    meta: uniformMeta,
    u: new Float32Array(121).fill(10),
    v: new Float32Array(121)
  };
  const forward = numerical.advanceWindStreamline(
    east, { lng: 5, lat: 5 }, 1, 10000
  );
  const backward = numerical.advanceWindStreamline(
    east, { lng: 5, lat: 5 }, -1, 10000
  );
  assert.ok(forward.lng > 5 && backward.lng < 5,
    "il verso della streamline uniforme e' errato");
  assert.ok(Math.abs(forward.lat - 5) < 1e-10,
    "un vento zonale produce uno spostamento meridionale spurio");
  assert.ok(Math.abs(forward.speedKmh - 36) < 1e-6,
    "la velocita' della streamline non conserva la conversione m/s-km/h");

  const calm = {
    meta: uniformMeta,
    u: new Float32Array(121).fill(0.1),
    v: new Float32Array(121)
  };
  assert.equal(
    numerical.advanceWindStreamline(calm, { lng: 5, lat: 5 }, 1, 10000),
    null,
    "la calma numerica genera linee senza direzione definita"
  );

  const vortexMeta = { lo1: 0, la1: 10, dx: 0.05, dy: 0.05, nx: 201, ny: 201 };
  const vortexU = new Float32Array(vortexMeta.nx * vortexMeta.ny);
  const vortexV = new Float32Array(vortexU.length);
  for (let row = 0; row < vortexMeta.ny; row += 1) {
    const latitude = vortexMeta.la1 - row * vortexMeta.dy;
    for (let column = 0; column < vortexMeta.nx; column += 1) {
      const longitude = vortexMeta.lo1 + column * vortexMeta.dx;
      const index = row * vortexMeta.nx + column;
      vortexU[index] = -(latitude - 5);
      vortexV[index] = longitude - 5;
    }
  }
  const curved = numerical.advanceWindStreamline(
    { meta: vortexMeta, u: vortexU, v: vortexV },
    { lng: 6, lat: 5 }, 1, 20000
  );
  assert.ok(curved.lat > 5.16 && curved.lng < 5.995,
    "RK2 non segue la curvatura del vortice sintetico");

  const lowLatitudeStep = numerical.streamlineStepMetres(35, 6);
  const highLatitudeStep = numerical.streamlineStepMetres(50, 6);
  assert.ok(highLatitudeStep < lowLatitudeStep,
    "il passo metrico non corregge la risoluzione zonale con la latitudine");
}

// Il bollettino sale dal basso invece di coprire la mappa dall'alto.
assert.match(html, /@media \(max-width: 959px\) \{[\s\S]{0,400}?#bulletin-card \{[\s\S]{0,300}?border-radius: 20px 20px 0 0;/,
  "il bollettino non e' un foglio dal basso sul telefono");
assert.match(html, /function enableSwipeToClose\(/,
  "i fogli non si chiudono con lo scorrimento verso il basso");

// Un bersaglio da dito non scende sotto 44 px.
assert.match(html, /\.header-button \{[\s\S]{0,120}?min-width: 44px;\s*\n\s*height: 44px;/,
  "i pulsanti dell'intestazione sono sotto la misura minima da dito");

// Il confine del modello va detto, non subito. Il campo colorato si ferma al
// bordo del dominio: senza una maschera fuori resta la carta nuda, e si legge
// come un difetto di disegno invece che come "qui la previsione finisce".
assert.match(html, /function generateModelMask\(/, "maschera del dominio assente");
assert.match(html, /id: "model-mask-layer"/, "il fuori dominio non viene velato");
assert.match(html, /id: "model-edge-layer"/, "il bordo del dominio non e' tracciato");

// --- Carta sinottica -------------------------------------------------------
assert.ok(html.includes('data-toggle="synoptic"'),
  "manca l'interruttore della carta sinottica");
const synopticMode = html.match(/function setSynopticChart\([\s\S]*?\n {6}\}\n/);
assert.ok(synopticMode, "modalita' carta sinottica assente");
// La modalita' deve spegnere la base fotografica, l'animato e tutto cio' che
// compete con il tratto.
["showSatellite = false", "showVectors = false", "showParticles = false",
 "showIsotherms = false", "show3D = false", "showRadar = false"]
  .forEach((expected) => {
    assert.ok(synopticMode[0].includes(expected),
      "la carta sinottica non spegne: " + expected);
  });
// Il campo colorato invece resta: velato sotto l'inchiostro e' quello che
// distingue una carta di analisi da una carta muta. Se la modalita' tornasse
// a forzare activeLayer a "none" il foglio si svuoterebbe.
assert.ok(!synopticMode[0].includes('activeLayer = "none"'),
  "la carta spegne ancora il campo colorato invece di velarlo");
assert.match(synopticMode[0], /activeLayer === "none" \? "none" : "visible"/,
  "la carta non lascia visibile il campo colorato");
// Scegliere un campo non deve piu' far uscire dalla carta.
const layerSwitch = html.match(/function setLayer\([\s\S]*?\n {6}\}\n/);
assert.ok(layerSwitch, "setLayer assente");
assert.ok(!/synopticChart = false/.test(layerSwitch[0]),
  "scegliere un campo colorato chiude ancora la carta");
// Sul foglio il campo e' un acquerello: se restasse pieno coprirebbe isobare
// e simboli frontali, che sono il motivo per cui la carta esiste.
assert.match(html, /const paperWash = synopticChart \?/,
  "il campo colorato non viene velato sulla carta");
// ...e accendere quello che una carta al suolo contiene.
["showIsobars = true", "showGraticule = true", "showFronts = frontsAvailable()"]
  .forEach((expected) => {
    assert.ok(synopticMode[0].includes(expected),
      "la carta sinottica non accende: " + expected);
  });
// Uscendo si deve tornare esattamente allo stato di prima, non a un default.
assert.match(synopticMode[0], /synopticRestore = \{[\s\S]*?layer: activeLayer/,
  "lo stato precedente non viene messo da parte");
assert.match(synopticMode[0], /const previous = synopticRestore \|\| \{\};/,
  "lo stato precedente non viene ripristinato");

// Il colore del foglio ha una sola definizione: tre punti diversi lo
// impostavano e l'ultimo vinceva, ed e' per questo che la carta restava scura.
assert.match(html, /function mapBackgroundColour\(\)/,
  "il colore del fondo non ha una definizione unica");
const backgroundAssignments = html.match(
  /setPaintProperty\(\s*\n?\s*"background",?\s*\n?\s*"background-color"/g
) || [];
backgroundAssignments.forEach(() => {});
assert.equal(
  (html.match(/"background-color",\s*\n?\s*mapBackgroundColour\(\)/g) || []).length
  + (html.match(/mapBackgroundColour\(\)\s*\n?\s*:/g) || []).length,
  3,
  "non tutti i punti che impostano il fondo passano dalla definizione unica"
);

// Il passo delle isobare dichiarato nel cartiglio deve essere quello con cui
// sono davvero tracciate, non un numero scritto a mano.
// Quattro hPa e' il passo operativo sinottico; due produce un eccesso di
// dettaglio su una griglia convection-permitting senza aggiungere chiarezza.
assert.match(html, /const ISOBAR_INTERVAL_HPA = 4;/,
  "le isobare non seguono il passo sinottico di 4 hPa");
assert.match(html, /const ISOBAR_MAJOR_EVERY = 8;/,
  "le isobare principali non sono marcate ogni 8 hPa");
assert.match(html, /createContourFeatures\(\s*\n?\s*pressure, meta, ISOBAR_INTERVAL_HPA, ISOBAR_MAJOR_EVERY\s*\n?\s*\)/,
  "le isobare non usano la costante dichiarata nel cartiglio");
// ISOBAR_MAJOR_EVERY e' un MODULO sul valore, non un moltiplicatore: sono
// marcate le isobare il cui valore e' multiplo di 8, quindi ogni 8 hPa.
assert.match(html, /"isobare ogni " \+ ISOBAR_INTERVAL_HPA/,
  "il cartiglio non legge il passo dalla stessa costante");
assert.ok(!/ISOBAR_INTERVAL_HPA \* ISOBAR_MAJOR_EVERY/.test(html),
  "il cartiglio moltiplica passo e modulo invece di dichiarare il modulo reale");

// Colore e linee devono raccontare la stessa struttura: stesso passo, stessi
// bordi, stesso campo. Se le fasce avessero un passo proprio, ogni cambio di
// tinta cadrebbe fra due isobare invece che sopra una.
assert.match(html, /buildDiscreteBands\(\s*\n?\s*PRESS_ANCHORS, 976, 1048, ISOBAR_INTERVAL_HPA, 1, true\s*\n?\s*\)/,
  "le fasce della pressione non condividono passo e bordi con le isobare");
// Le costanti servono gia' alla costruzione delle fasce: se restassero
// dichiarate piu' in basso, leggerle da li' sarebbe un errore di zona morta.
assert.ok(
  html.indexOf("const ISOBAR_INTERVAL_HPA = 4;") < html.indexOf("const PRESS_ANCHORS"),
  "il passo delle isobare e' dichiarato dopo le fasce che lo usano"
);
// La scala e' fitta dove la pressione vive davvero: senza le ancore ogni 4 hPa
// fra 1000 e 1032 la giornata normale ricade in tre tinte quasi uguali.
{
  const anchors = html.match(/const PRESS_ANCHORS = \[([\s\S]*?)\];/);
  assert.ok(anchors, "ancore della pressione assenti");
  const values = [...anchors[1].matchAll(/v: (\d+)/g)].map((m) => Number(m[1]));
  const core = values.filter((v) => v >= 1000 && v <= 1032);
  assert.ok(core.length >= 9,
    "la scala della pressione non e' infittita fra 1000 e 1032 hPa");
  assert.ok(Math.max(...values) >= 1048 && Math.min(...values) <= 976,
    "la scala della pressione non copre piu' gli estremi reali");
}

// La PMSL a 2,2 km porta gli artefatti della riduzione al livello del mare.
// Colore, lettura puntuale, isobare e centri devono quindi condividere una
// sola analisi sinottica, fisicamente isotropa in chilometri.
assert.match(html, /const PRESSURE_ANALYSIS_RADIUS_KM = 60;/,
  "manca la scala fisica della media sinottica della pressione");
assert.match(html, /const PRESSURE_ANALYSIS_PASSES = 2;/,
  "la media sinottica non usa i due passaggi previsti");
const pressureAnalysis = html.match(/function pressureAnalysisGrid\([\s\S]*?\n {6}\}\n/);
assert.ok(pressureAnalysis, "pressureAnalysisGrid assente");
assert.match(pressureAnalysis[0], /Math\.cos\(middleLatitude \* Math\.PI \/ 180\)/,
  "la media non corregge la distanza zonale con la latitudine");
assert.match(pressureAnalysis[0], /radiusX, radiusY/,
  "la media usa lo stesso numero di celle sui due assi e non lo stesso raggio in km");
assert.match(pressureAnalysis[0], /pass < PRESSURE_ANALYSIS_PASSES/,
  "il numero di passaggi non usa la costante dichiarata");

const pressureProducts = html.match(/function buildPressureProducts\([\s\S]*?\n {6}\}\n/);
assert.ok(pressureProducts, "buildPressureProducts assente");
assert.match(pressureProducts[0], /createIsobarFeatures\(analysis, meta\)/,
  "le isobare sono ancora tracciate sul campo grezzo");
// I centri seguono lo stesso campo delle isobare, e devono dichiarare che
// arriva gia' analizzato: lisciarlo una seconda volta lo appiattisce al punto
// che nessun contorno chiuso raggiunge un'isobara di prominenza, e sui passi
// reali sparivano tutti i centri.
assert.match(pressureProducts[0], /detectPressureCenters\(analysis, meta, true\)/,
  "i centri barici non seguono il campo mostrato dalle isobare");
const centreDetector = html.match(/function detectPressureCenters\([\s\S]*?\n {6}\}\n/);
assert.ok(centreDetector, "detectPressureCenters assente");
assert.match(centreDetector[0], /if \(!presmoothed\) \{/,
  "il rilevatore liscia di nuovo un campo gia' analizzato");
assert.match(pressureProducts[0], /_pressureProductKind === productKind/,
  "la cache delle isobare non distingue modello e fusione METAR");
assert.match(html, /if \(activeLayer === "press" && fieldGrid\)[\s\S]*?pressureAnalysisGrid\(/,
  "il colore della pressione usa ancora la griglia grezza");
const fieldValue = html.match(/function fieldValue\([\s\S]*?\n {6}\}\n/);
assert.ok(fieldValue, "fieldValue assente");
assert.match(fieldValue[0], /key === "press"[\s\S]*?pressureAnalysisGrid\(/,
  "la lettura puntuale della pressione non usa la media sinottica");

// La legenda disegna i simboli con la stessa funzione della mappa: due
// disegni separati potrebbero raccontare due convenzioni diverse.
const legend = html.match(/function paperTitleBlock\([\s\S]*?\n {6}\}\n/);
assert.ok(legend, "cartiglio della carta assente");
assert.match(legend[0], /drawFrontStyled\(/,
  "la legenda non usa il disegno reale dei fronti");
assert.match(html, /if \(synopticChart\) drawChartFurniture\(\);/,
  "il cartiglio non viene disegnato");

// --- La carta e' una carta: cornice, margine, reticolo, scala, rosa ---------
// Senza il margine il disegno sconfina fino ai bordi dello schermo e la
// cornice diventa una decorazione appoggiata sopra una mappa infinita.
const furniture = html.match(/function drawChartFurniture\([\s\S]*?\n {6}\}\n/);
assert.ok(furniture, "corredo della carta assente");
["paperMask", "paperGraticuleTicks", "paperScaleBar", "paperCompass",
 "paperNeatline", "paperTitleBlock"].forEach((piece) => {
  assert.match(furniture[0], new RegExp(piece + "\\(box\\)"),
    "il corredo della carta non disegna: " + piece);
});
// Il margine va riempito PRIMA della cornice e delle etichette di bordo,
// altrimenti coprirebbe proprio quello che deve incorniciare.
assert.ok(
  furniture[0].indexOf("paperMask(box)") < furniture[0].indexOf("paperNeatline(box)"),
  "il margine viene riempito dopo la cornice"
);
// Le etichette delle isobare sulla carta stanno DENTRO la linea: la linea si
// interrompe e il numero occupa il vuoto. E' anche il motivo per cui restano
// leggibili sopra un campo colorato -- non c'e' alone chiaro che lo sporchi.
assert.match(html, /function strokePaperIsobar\(/,
  "manca il tratto cartaceo delle isobare");
assert.match(html, /function paperIsobarGaps\(/,
  "le etichette delle isobare non sono incassate nella linea");
const paperIsobar = html.match(/function strokePaperIsobar\([\s\S]*?\n {6}\}\n/);
assert.match(paperIsobar[0], /setLineDash\(dash\)/,
  "il buco per il numero non e' un tratteggio calcolato");
assert.ok(!/strokeText/.test(paperIsobar[0]),
  "l'etichetta della carta usa ancora un alone invece del buco nella linea");

// --- Scala della temperatura: la linea del gelo cade a zero ------------------
// La scala ha un salto voluto in meno di un kelvin, fra il blu di 273,15 K e
// il verde di 274 K. Il sito colora ogni fascia discreta con un solo colore:
// prendendolo sul bordo inferiore, la fascia [0, 2) leggeva il colore a zero
// gradi esatti -- il blu del gelo -- e dipingeva come sottozero temperature
// fino a due gradi sopra. Il colore va preso al centro della fascia.
{
  const anchorsSource = html.match(/const TEMP_ANCHORS = \[\n([\s\S]*?)\n {6}\];/);
  assert.ok(anchorsSource, "ancoraggi della temperatura assenti");
  const bandsSource = html.match(/const TEMP_BANDS = [^;]+;/);
  assert.ok(bandsSource, "fasce della temperatura assenti");
  assert.match(bandsSource[0], /,\s*true\s*\)/,
    "le fasce di temperatura non sono campionate al centro");

  const pick = (name) => {
    const found = html.match(new RegExp("function " + name + "\\([\\s\\S]*?\\n {6}\\}"));
    assert.ok(found, "manca " + name);
    return found[0];
  };
  const temperature = new Function(
    pick("interpolateScale") + "\n" + pick("buildDiscreteBands") + "\n"
    + "const TEMP_ANCHORS=[" + anchorsSource[1] + "];\n" + bandsSource[0] + "\n"
    + pick("colorFor") + "\nreturn { TEMP_BANDS, colorFor };"
  )();
  const info = { stops: temperature.TEMP_BANDS, discrete: true };
  const at = (value) => temperature.colorFor(value, info).slice(0, 3).map(Math.round);
  const isBlueish = (c) => c[2] > c[1] && c[2] > c[0];
  const isGreenish = (c) => c[1] > c[2] && c[1] >= c[0];

  assert.ok(isBlueish(at(-0.1)),
    "sotto zero deve restare il blu del gelo, trovato " + at(-0.1));
  assert.ok(isGreenish(at(0)),
    "a zero gradi deve iniziare il verde, trovato " + at(0));
  assert.ok(isGreenish(at(1.9)),
    "la fascia sopra lo zero non deve tornare blu, trovato " + at(1.9));
  // Gli estremi della scala fornita devono arrivare intatti.
  assert.match(anchorsSource[1], /\{ v: 0, c: \[93, 133, 198\] \}/,
    "l'ancoraggio a 0 gradi non e' quello della scala fornita");
  assert.match(anchorsSource[1], /\{ v: 0\.85, c: \[68, 125, 99\] \}/,
    "l'ancoraggio subito sopra lo zero non e' quello della scala fornita");
  assert.match(anchorsSource[1], /\{ v: -70\.15, c: \[115, 70, 105\] \}/,
    "l'estremo freddo non e' quello della scala fornita");
  assert.match(anchorsSource[1], /\{ v: 46\.85, c: \[71, 14, 0\] \}/,
    "l'estremo caldo non e' quello della scala fornita");
}
// --- Palette meteorologiche ---
// Le tre temperature al suolo sono una scelta grafica consolidata e devono
// restare sulla stessa scala. La temperatura in libera atmosfera usa invece
// una rampa dedicata, cosi' una futura revisione non le accorpa per errore.
["temp", "feels", "wetbulb"].forEach((key) => {
  assert.match(html, new RegExp(key + ":\\s*\\{[\\s\\S]{0,180}?stops:\\s*TEMP_BANDS,"),
    "il layer " + key + " non usa piu' la palette termica protetta");
});
assert.match(html, /const T850_STOPS = buildDiscreteBands\(T850_ANCHORS, -30, 35, 1, 1\);/,
  "manca la scala dedicata della temperatura a 850 hPa");
assert.match(html, /t850:\s*\{[\s\S]{0,180}?stops:\s*T850_STOPS,/,
  "la temperatura a 850 hPa e' tornata sulla palette al suolo");

// Vento medio: fasce ogni 5 km/h e grammatica cromatica della carta di
// riferimento. La fascia estrema resta colorata: un valore oltre scala non
// deve diventare bianco e sembrare calma.
{
  const windSource = html.match(/const WIND_STOPS = \[([\s\S]*?)\n {6}\];/);
  assert.ok(windSource, "scala esplicita del vento assente");
  const edges = [...windSource[1].matchAll(/\{ v: (\d+), c:/g)]
    .map((match) => Number(match[1]));
  assert.deepEqual(edges, Array.from({ length: 28 }, (_, index) => index * 5),
    "le fasce del vento non avanzano piu' ogni 5 km/h fino a 135");
  assert.match(windSource[1], /\{ v: 0, c: \[250, 250, 249\] \}/,
    "la calma non e' quasi bianca");
  assert.match(windSource[1], /\{ v: 15, c: \[63, 154, 245\] \}/,
    "la brezza non raggiunge il blu della carta di riferimento");
  assert.match(windSource[1], /\{ v: 20, c: \[113, 248, 163\] \}/,
    "manca il passaggio netto blu-verde a 20 km/h");
  assert.match(windSource[1], /\{ v: 85, c: \[232, 54, 42\] \}/,
    "il vento molto forte non raggiunge il rosso operativo");
  assert.match(windSource[1], /\{ v: 135, c: \[246, 194, 247\] \}/,
    "l'estremo oltre scala torna bianco e diventa invisibile");

  const paletteGenerator = fs.readFileSync(
    path.join(root, "scripts", "generate_palettes.py"), "utf8"
  );
  assert.match(paletteGenerator, /out\["WIND_STOPS"\] = \[/,
    "il generatore puo' ancora ripristinare la vecchia rampa del vento");
  assert.doesNotMatch(paletteGenerator, /out\["WIND_ANCHORS"\]/,
    "il generatore conserva ancora gli ancoraggi obsoleti");
}

// Resa Meteociel: integrazione piu' fitta, linee uniformi e alone appena
// percettibile. La velocita' e' gia' nel colore e non deve gonfiare il tratto.
assert.match(html, /screenStepPx: 3\.6,/,
  "il passo delle streamline e' troppo largo per curve regolari");
assert.match(html, /seedSpacingMobile: 14,/,
  "le streamline sono ancora troppo rade sul telefono");
assert.match(html, /occupancyMobile: 8,/,
  "il controllo di occupazione impedisce la densita' della carta di riferimento");
assert.match(html, /maxStepsMobile: 128,/,
  "le linee mobili restano troppo corte");
assert.match(html, /strokeStreamline\(line, "rgba\(4,10,13,0\.84\)", 0\.62\)/,
  "il tratto del vento non e' piu' sottile e uniforme");
assert.match(html, /strokeStreamline\(line, "rgba\(255,255,255,0\.30\)", 1\.08\)/,
  "l'alone discreto delle streamline e' assente");
assert.doesNotMatch(html, /widthBySpeed/,
  "la velocita' viene codificata di nuovo nello spessore delle linee");

// I campi senza fenomeno devono lasciare visibile la carta geografica.
assert.match(html, /const RAIN_STOPS = \[\s*\{ v: 0, c: \[0, 0, 0\], a: 0 \}/,
  "assenza di pioggia non trasparente");
assert.match(html, /const STORM_STOPS = \[\s*\{ v: 0, c: \[221, 234, 242\], a: 0 \}/,
  "assenza di probabilita' temporalesca non trasparente");
assert.match(html, /const CAPE_STOPS = \[\s*\{ v: 0, c: \[238, 244, 240\], a: 0 \}/,
  "assenza di CAPE non trasparente");

// --- Raffiche ---
// La scala delle raffiche vale come scala solo se i suoi bordi restano i
// gradi Beaufort: e' il motivo per cui una fascia cambia colore dove cambia
// il grado, e le tre soglie che fanno danni si leggono senza legenda.
{
  const gustSource = html.match(/const GUST_STOPS = \[[\s\S]*?\n {6}\];/);
  assert.ok(gustSource, "scala delle raffiche assente");
  const edges = [...gustSource[0].matchAll(/\{ v: (-?[\d.]+),/g)]
    .map((m) => Number(m[1]));
  assert.deepEqual(edges, [0, 12, 20, 29, 39, 50, 62, 75, 89, 103, 118, 140],
    "i bordi delle fasce non sono piu' i gradi Beaufort in km/h");
  const layerInfo = html.match(/\n {8}gust: \{[\s\S]*?\n {8}\},/);
  assert.ok(layerInfo, "il layer delle raffiche non e' in LAYER_INFO");
  assert.match(layerInfo[0], /stops: GUST_STOPS,/,
    "il layer delle raffiche non usa la propria scala");
  assert.match(layerInfo[0], /discrete: true,/,
    "le raffiche devono restare a fasce nette, non sfumate");
  assert.match(layerInfo[0], /unit: "km\/h",/,
    "le raffiche non sono piu' pubblicate in km/h");
  assert.match(html, /data-layer="gust"/,
    "manca la scheda delle raffiche nel pannello dei livelli");
  assert.match(html, /id="detail-gust"/,
    "manca la riga delle raffiche nella lettura del punto");
  assert.match(html, /"temp", "feels_like", "rain", "press", "rh", "cloud", "gust",/,
    "il campo delle raffiche non viene compattato come gli altri");
}
// Il campo deve anche esistere: la scala serve a poco se la pipeline non lo
// pubblica. Le raffiche arrivano da vmax_10m, in m/s, e vanno in km/h.
{
  const pipeline = fs.readFileSync(
    path.join(__dirname, "..", "process_data.py"), "utf8");
  assert.match(pipeline, /"gust": \(\s*\n\s*clean_for_json\(np\.asarray\(wind_gust_10m\) \* 3\.6, 0\)/,
    "la pipeline non pubblica le raffiche in km/h");
}
console.log("3D map regression checks: OK");
