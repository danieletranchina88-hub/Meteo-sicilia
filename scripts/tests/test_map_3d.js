#!/usr/bin/env node
"use strict";

// Regression checks for the static GitHub Pages client. They intentionally do
// not require WebGL, so they also run on the standard GitHub Actions runner.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "../..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const modernUi = fs.readFileSync(path.join(root, "modern-ui.js"), "utf8");
function implementazione(nome) {
  const i = html.search(new RegExp("      function " + nome + "\\("));
  assert.ok(i >= 0, "manca " + nome);
  return html.slice(i, html.indexOf("\n      }", i) + 8);
}
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
assert.match(html, /pressione stazione ≠ MSLP/,
  "l'interfaccia confonde ancora altimeter setting e pressione MSLP");
assert.match(html, /pressureAtElevation/,
  "la pressione di stazione non viene confrontata nello spazio osservativo");
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
// L'osservato NON ha piu' il dominio del previsto, ed e' il punto: erano la
// stessa costante, e per questo il satellite si fermava ai bordi di ICON-2I
// anche se MTG vede mezzo emisfero. Adesso il riquadro osservato deve
// contenere quello del modello con margine vero da ogni lato -- altrimenti
// una perturbazione atlantica si vedrebbe solo quando e' gia' arrivata.
const numero = (nome, campo) => {
  const trovato = html.match(
    new RegExp("const " + nome + " = \\{[^}]*" + campo + ": (-?[0-9.]+)")
  );
  assert.ok(trovato, "manca " + campo + " in " + nome);
  return Number(trovato[1]);
};
for (const campo of ["west", "south", "east", "north"]) {
  assert.notEqual(numero("CLOUD_DOMAIN", campo), numero("MODEL_DOMAIN", campo),
    "l'osservato ha ancora lo stesso " + campo + " del modello");
}
assert.ok(numero("CLOUD_DOMAIN", "west") <= -20,
  "l'osservato non arriva in mezzo all'Atlantico: le perturbazioni si vedrebbero solo all'arrivo");
assert.ok(numero("CLOUD_DOMAIN", "north") >= 60,
  "l'osservato non arriva alle latitudini islandesi, dove nascono le saccature");
assert.ok(numero("CLOUD_DOMAIN", "south") <= numero("MODEL_DOMAIN", "south")
  && numero("CLOUD_DOMAIN", "east") >= numero("MODEL_DOMAIN", "east"),
  "l'osservato non contiene piu' tutto il dominio del modello");
// E deve restare dentro cio' che il servizio dichiara di avere: il
// GetCapabilities di EUMETView da' +-70 gradi per l'infrarosso e per il
// Lightning Imager, che sono i piu' stretti dei prodotti usati qui.
for (const campo of ["west", "south", "east", "north"]) {
  assert.ok(Math.abs(numero("CLOUD_DOMAIN", campo)) <= 70,
    "l'osservato esce dai limiti dichiarati dal servizio su " + campo);
}
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
assert.match(html, /map\.setMaxBounds\(paddedModelBounds\(view, dominio\)\);/,
  "dopo un ridimensionamento i limiti tornano a tagliare il dominio");
// I limiti si azzerano PRIMA di riscriverli. MapLibre tiene la vista dentro
// quelli vecchi mentre li si cambia, quindi passando dal dominio del modello
// a quello dell'osservato la mappa restava incollata al bordo di ICON-2I e
// l'Atlantico non si raggiungeva comunque.
assert.match(html, /map\.setMaxBounds\(null\);\s*\n\s*map\.setMinZoom\(view\.zoom\);\s*\n\s*map\.setMaxBounds\(/,
  "i limiti vecchi non vengono tolti prima di allargarli");
// Accendere l'osservato deve cambiare il dominio percorribile, altrimenti il
// satellite copre l'Europa ma la mappa non ci lascia arrivare.
assert.match(html, /function dominioNavigabile\(\) \{\s*\n\s*return \(showSatelliteClouds \|\| showRadar \|\| showLightning\s*\n\s*\|\| showLiveLightning\) \? CLOUD_DOMAIN : MODEL_DOMAIN;/,
  "il dominio percorribile non dipende dai livelli osservati accesi");
// Il velo che copre il fuori-dominio dice "qui non c'e' previsione", ed e'
// vero; sull'osservato e' il contrario, e lasciandolo acceso sbiancava
// l'Europa al 72% lasciando in chiaro un rettangolo sull'Italia.
assert.match(html, /map\.setLayoutProperty\("model-mask-layer", "visibility",\s*\n\s*largo \? "none" : "visible"\);/,
  "il velo del fuori-modello resta acceso anche quando si guarda l'osservato");
// Il bordo tratteggiato invece resta sempre: non copre niente e continua a
// dire dove finisce la previsione.
assert.doesNotMatch(html, /setLayoutProperty\("model-edge-layer", "visibility"/,
  "anche il bordo del modello viene nascosto: si perde il confine della previsione");
assert.match(html, /map\.resize\(\);\s*\n\s*lockMapToModelDomain\(false\);/,
  "ruotando il telefono il limite del dominio non viene ricalcolato");
assert.match(meteogramHtml, /id="weather-strip"/,
  "manca la sintesi visuale ogni tre ore");
assert.match(meteogramHtml, /label:"Coerenza interna", values:s\.stormConfidence/,
  "lo score temporalesco e' mostrato senza coerenza interna");
assert.match(meteogramHtml, /label:"Raffica", values:s\.gust10/,
  "il meteogramma del vento ignora le raffiche");

// Il sito deve distinguere variabili, diagnostiche e probabilita' calibrate.
assert.match(html, /id="field-method-card" class="field-method-card"/,
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
// Due hPa e' il passo di una carta al suolo a scala regionale. A quattro il
// dominio intero aveva 7 isobare e un centro: e siccome le fasce di colore
// hanno per costruzione lo stesso passo delle isobare, un campo da 11,9 hPa
// dava tre sole tinte distinguibili su tutta l'Italia.
assert.match(html, /const ISOBAR_INTERVAL_HPA = 2;/,
  "le isobare non rispettano il passo regionale di 2 hPa");
assert.match(html, /const ISOBAR_MAJOR_EVERY = 8;/,
  "le isobare principali non rispettano il passo di 8 hPa");
assert.match(html, /passo 2 hPa · principali ogni 8/,
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
// Il bollettino e' deterministico e basta: nessuna sintesi da modello
// linguistico, nessun percorso che la carichi. Se rientra, deve rientrare per
// una decisione esplicita, non di nascosto.
assert.doesNotMatch(html, /ai_expert_bulletin|ai_agent_status/,
  "il sito e' tornato a caricare un prodotto IA");
assert.doesNotMatch(html, /renderAiAgentBrief|appendAiClaim/,
  "il sito e' tornato a rappresentare una sintesi IA");
assert.match(html, /docs\/agente_meteorologico\.md/,
  "l'interfaccia non collega il protocollo verificabile dell'agente");
assert.doesNotMatch(html, /GEMINI_API_KEY|GROQ_API_KEY/,
  "un nome di segreto API è comparso nel frontend pubblico");

assert.match(html, /maplibre-gl@5\.24\.0/, "MapLibre v5 stabile non caricata");
assert.doesNotMatch(html, /map\.transform\b/, "uso di API MapLibre interna e fragile");
assert.match(html, /map\.setSky\(/, "cielo MapLibre nativo assente");
assert.match(html, /function terrainExaggerationForZoom\(/,
  "esagerazione verticale adattiva assente");
// La sorgente canvas sembra la scelta ovvia -- niente codifica PNG per
// fotogramma -- ma in MapLibre 5.24 non carica la texture da un canvas fuori
// documento: provato in Chromium, ogni aggiornamento del livello meteo
// lanciava "InvalidStateError: The source image could not be decoded" e sulla
// mappa non arrivava un pixel colorato, pur avendo il canvas i colori giusti.
// Attaccare il canvas al documento e ricreare la sorgente non bastano.
assert.match(html, /map\.addSource\("weather", \{\s*[^}]*type: "image"/,
  "il livello meteo non usa piu' la sorgente immagine, l'unica che disegna");
assert.doesNotMatch(html, /type: "canvas",\s*canvas: rasterCanvas/,
  "sorgente canvas reintrodotta: in questa versione di MapLibre lascia la "
  + "mappa nera");
assert.match(html, /rasterCanvas\.toBlob\(/,
  "la pubblicazione del raster non passa piu' da un blob");

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

// Le frecce vengono proiettate con la stessa matrice economica usata dalle
// particelle -- map.project interroga il DEM a ogni chiamata, e qui le
// chiamate sono due per freccia -- e restano sotto isobare, fronti e toponimi.
assert.match(html, /const centre = projectParticle\(location\.lng, location\.lat, groundHeight\);/,
  "le frecce non usano la proiezione vettoriale ottimizzata");
assert.doesNotMatch(html, /const centre = map\.project/,
  "le frecce interrogano ancora il DEM attraverso map.project");
assert.ok(
  drawVectors[0].indexOf("drawWindArrows(wind)")
    < drawVectors[0].indexOf("drawSynopticCanvas()"),
  "le frecce coprono l'inchiostro dell'analisi sinottica"
);

// La geometria di una scadenza non cambia finche' la mappa sta ferma: va
// costruita una volta e ridisegnata dalla cache, altrimenti torna la spesa
// per fotogramma che rendeva a scatti la riproduzione.
const drawArrows = html.match(/function drawWindArrows\([\s\S]*?\n {6}\}/);
assert.ok(drawArrows, "drawWindArrows assente");
assert.match(drawArrows[0], /windArrowCache\.get\(key\)/,
  "le frecce non riusano i tracciati gia' costruiti");
assert.match(drawArrows[0], /vectorContext\.stroke\(group\.path\)/,
  "le frecce non vengono ridisegnate come Path2D");
// Un Path2D per classe di velocita': e' cio' che permette di conservare
// colore e spessore proporzionali al vento senza una chiamata per freccia.
assert.match(html, /speedClasses: \d+/, "manca la suddivisione in classi di velocita'");
assert.match(html, /function windArrowSpeedClass\(/, "manca la classificazione per velocita'");
const buildArrows = html.match(/function buildWindArrowPaths\([\s\S]*?\n {6}\}/);
assert.ok(buildArrows, "buildWindArrowPaths assente");
assert.match(buildArrows[0], /clamp\(5 \+ speed \* 0\.46/,
  "la lunghezza della freccia non dipende piu' dalla velocita'");
assert.match(buildArrows[0], /colorFor\(speed, LAYER_INFO\.wind\)/,
  "la freccia non usa piu' la scala di colore del campo vento");

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
  buildArrows[0],
  /for \(let y = inset; y < window\.innerHeight - inset; y \+= spacing\)/,
  "le frecce non coprono tutto il riquadro"
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
assert.match(html, /update\(image\.src, false\);/,
  "il satellite non conserva il prodotto originale del fornitore");
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

// In the standalone satellite view IR/VIS and RGB retain the original pixels.
const publishClouds = html.match(/function publishSatelliteClouds\([\s\S]*?\n {6}\}/)[0];
assert.match(publishClouds, /update\(image\.src, false\)/, "immagine satellitare originale assente");
assert.doesNotMatch(publishClouds, /getImageData|putImageData/, "il satellite viene ancora trasformato in maschera");
assert.match(html, /mode: "diagnostic"/, "RGB scientifiche assenti");
assert.match(html, /solo di giorno/, "i canali diurni non dichiarano il limite");
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
assert.match(html, /setWeatherView\(weatherView === "satellite"/,
  "il satellite non usa la vista indipendente");
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
const satelliteTime = html.match(/function syncTimelineToSatellite\([\s\S]*?\n {6}\}/)[0];
assert.doesNotMatch(satelliteTime, /loadStep/, "il satellite cambia ancora la scadenza del modello");
assert.match(satelliteTime, /satellite-time/, "ora osservata non visibile nella vista satellite");
assert.match(modernUi, /weatherView = 'satellite'; clearMeteorologicalLayers\(\);/,
  "la vista satellite conserva sovrapposizioni previste");
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
assert.match(modernUi, /activeLayer=saved.layer/, "uscendo dal satellite non torna il campo scelto");

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
assert.match(html, /kind === "storm" && catalog\[index\]\.storm === false/,
  "senza il controllo sul catalogo si chiederebbe un file inesistente");
assert.match(html, /!upperActive && !stormActive\s*\n\s*&& !probActive && !profileActive && fieldGrid/,
  "la fusione con le stazioni si applicherebbe anche ai campi derivati");

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
const windLayer = html.match(/if \(layerShowsFlow\(nextLayer\) && !synopticChart\)[\s\S]*?\n {8}\}/);
assert.ok(windLayer, "scelta del campo vento assente");
assert.match(windLayer[0], /showVectors = true/,
  "il campo vento non attiva le linee di corrente");
assert.doesNotMatch(windLayer[0], /showParticles = true/,
  "il campo vento accende ancora automaticamente le particelle");
// La raffica e' uno scalare senza direzione propria: le linee di corrente
// sopra il suo campo colorato sono quelle del vento medio a 10 m.
const flowLayers = html.match(/function layerShowsFlow\([\s\S]*?\n {6}\}/);
assert.ok(flowLayers, "manca la scelta dei campi con linee di corrente");
assert.match(flowLayers[0], /key === "wind" \|\| key === "gust"/,
  "le linee di corrente non coprono anche le raffiche");

// Le frecce non sono decorazione: lunghezza, spessore, punta e colore
// dicono la velocita', e il verso dice dove soffia il vento.
assert.match(html, /<b>Frecce del vento<\/b>/,
  "il controllo del vento non promette piu' le frecce");
assert.match(html, /const ARROW_CONFIG = \{/,
  "configurazione delle frecce assente");
assert.doesNotMatch(html, /function traceWindStreamlines\(/,
  "il tracciatore delle linee di corrente e' tornato");
assert.doesNotMatch(html, /function advanceWindStreamline\(/,
  "l'integratore delle linee di corrente e' tornato");

// Verifica numerica su campi noti: il campionamento del vento e la geometria
// della freccia. Il campionatore e' lo stesso che alimentava le linee, e resta
// il punto in cui un errore di segno passerebbe inosservato a occhio.
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
    "const ARROW_CONFIG={minSpeedMps:0.22,minSpeedKmh:0.8,speedClasses:24," +
      "maxSpeedKmh:140};\n" +
    // Nessun terreno in questo sandbox numerico: orographicWindDeflection
    // deve ritornare subito null, e resta il campionamento sul campo grezzo.
    "let flowTerrainGradient=null;\n" +
    "let selectedLevel='surface';\n" +
    "function clamp(v,a,b){return Math.max(a,Math.min(b,v));}\n" +
    "function sampleBilinear(array,gx,gy,nx,ny){" +
      "const x=Math.max(0,Math.min(nx-1,gx));" +
      "const y=Math.max(0,Math.min(ny-1,gy));" +
      "const x0=Math.max(0,Math.min(nx-2,Math.floor(x)));" +
      "const y0=Math.max(0,Math.min(ny-2,Math.floor(y)));" +
      "const fx=x-x0,fy=y-y0,i=y0*nx+x0;" +
      "return array[i]*(1-fx)*(1-fy)+array[i+1]*fx*(1-fy)+" +
        "array[i+nx]*(1-fx)*fy+array[i+nx+1]*fx*fy;}\n" +
    functionSource("orographicWindDeflection") + "\n" +
    functionSource("sampleWindVector") + "\n" +
    functionSource("windArrowSpeedClass") + "\n" +
    functionSource("windArrowClassSpeed") + "\n" +
    functionSource("appendWindArrow") + "\n" +
    "return {sampleWindVector,windArrowSpeedClass,windArrowClassSpeed," +
      "appendWindArrow};"
  )();

  const uniformMeta = { lo1: 0, la1: 10, dx: 1, dy: 1, nx: 11, ny: 11 };
  const east = {
    meta: uniformMeta,
    u: new Float32Array(121).fill(10),
    v: new Float32Array(121)
  };
  const sample = numerical.sampleWindVector(east, 5, 5);
  assert.ok(sample.u > 0 && Math.abs(sample.v) < 1e-10,
    "un vento zonale produce una componente meridionale spuria");
  assert.ok(Math.abs(sample.speedKmh - 36) < 1e-6,
    "la conversione m/s - km/h non torna");

  const calm = {
    meta: uniformMeta,
    u: new Float32Array(121).fill(0.1),
    v: new Float32Array(121)
  };
  assert.equal(numerical.sampleWindVector(calm, 5, 5), null,
    "sulla calma numerica viene disegnata una freccia senza direzione");

  // Le classi coprono la scala senza buchi e saturano in cima, e la velocita'
  // rappresentativa di una classe ricade nella classe stessa.
  assert.equal(numerical.windArrowSpeedClass(0), 0);
  assert.equal(numerical.windArrowSpeedClass(1000), 23,
    "una velocita' fuori scala non ricade nell'ultima classe");
  let precedente = -1;
  for (let speed = 0; speed <= 140; speed += 1) {
    const index = numerical.windArrowSpeedClass(speed);
    assert.ok(index >= precedente, "le classi non sono monotone in velocita'");
    assert.equal(numerical.windArrowSpeedClass(numerical.windArrowClassSpeed(index)),
      index, "la velocita' rappresentativa esce dalla propria classe");
    precedente = index;
  }

  // Geometria: la freccia punta sottovento e la punta sta all'estremita' di
  // valle. Un segno invertito qui disegnerebbe un campo plausibile e
  // completamente sbagliato, che a occhio non si distingue.
  const passi = [];
  const finto = {
    moveTo: (x, y) => passi.push(["move", x, y]),
    lineTo: (x, y) => passi.push(["line", x, y])
  };
  numerical.appendWindArrow(finto, 100, 100, 1, 0, 10, 3);
  const asta = [passi[0], passi[1]];
  assert.equal(asta[0][0], "move");
  assert.ok(asta[1][1] > asta[0][1],
    "l'asta della freccia non punta nel verso del vento");
  const punta = passi.slice(2);
  assert.ok(punta.every(passo => passo[1] <= asta[1][1] + 1e-9),
    "la punta non sta all'estremita' sottovento dell'asta");
  assert.ok(punta.some(passo => passo[2] > 100) && punta.some(passo => passo[2] < 100),
    "i due bracci della punta non si aprono ai lati dell'asta");
}

// La deflessione orografica: verso piegato verso le isoipse su un versante
// ripido, nessuna correzione in pianura, velocita' sempre conservata.
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

  const deflection = new Function(
    "const OROGRAPHIC_DEFLECTION={minSlope:0.02,fullSlope:0.35,maxWeight:0.55," +
      "calmSpeedMps:3,strongSpeedMps:15,minWeightAtStrongWind:0.15," +
      "sampleOffsetMetres:[300,3000]};\n" +
    "const FLOW_GRADIENT_MAX_SIDE=512;\n" +
    "let selectedLevel='surface';\n" +
    "let flowTerrainGradient=null;\n" +
    "function clamp(v,a,b){return Math.max(a,Math.min(b,v));}\n" +
    functionSource("offsetWindLocation") + "\n" +
    functionSource("buildFlowGradientField") + "\n" +
    functionSource("orographicWindDeflection") + "\n" +
    "return {\n" +
    "  run: orographicWindDeflection,\n" +
    // Si passa dal campionatore vero al reticolo, come in produzione: la
    // prova copre la catena intera, non solo l'ultimo anello.
    "  useSampler: function(sampler){\n" +
    "    flowTerrainGradient = buildFlowGradientField(sampler, 9.5, 10.5, 44.5, 45.5);\n" +
    "  },\n" +
    "  useLevel: function(level){ selectedLevel = level; }\n" +
    "};"
  )();

  // Versante ripido che sale solo verso est: le isoipse corrono nord-sud, e
  // un vento che punta dritto in salita (verso est) deve piegare verso nord
  // o sud senza cambiare velocita'.
  deflection.useSampler({
    metresPerPixel: 100,
    heightAt: function (longitude) { return longitude * 100000; }
  });
  const upslope = deflection.run(5, 0, 5, 10, 45);
  assert.ok(upslope, "nessuna deflessione su un pendio marcato");
  assert.ok(Math.abs(upslope.v) > 0.5,
    "il vento in salita diretta non viene incanalato lungo le isoipse");
  assert.ok(Math.abs(Math.hypot(upslope.u, upslope.v) - 5) < 1e-6,
    "la deflessione altera la velocita' invece del solo verso");
  assert.ok(Math.abs(upslope.u) < 5,
    "il verso non si sposta affatto verso le isoipse");

  // Stesso versante, vento forte: il sinottico deve prevalere di piu' sul
  // rilievo, quindi la deviazione angolare deve essere minore che a vento
  // debole a parita' di pendenza.
  const weakDeviation = Math.atan2(upslope.v, upslope.u);
  const strong = deflection.run(20, 0, 20, 10, 45);
  const strongDeviation = Math.atan2(strong.v, strong.u);
  assert.ok(Math.abs(strongDeviation) < Math.abs(weakDeviation),
    "il vento forte viene deviato quanto quello debole");

  // Pianura: gradiente nullo, nessuna correzione da applicare.
  deflection.useSampler({
    metresPerPixel: 100,
    heightAt: function () { return 0; }
  });
  assert.equal(deflection.run(5, 0, 5, 10, 45), null,
    "la pianura genera una deflessione che non dovrebbe esistere");

  // In quota (livello diverso da "surface") il rilievo non deve intervenire.
  deflection.useSampler({
    metresPerPixel: 100,
    heightAt: function (longitude) { return longitude * 100000; }
  });
  deflection.useLevel("850hPa");
  assert.equal(deflection.run(5, 0, 5, 10, 45), null,
    "la deflessione al suolo agisce anche sul vento in quota");
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
//
// Due hPa, non quattro: il dominio e' l'Italia, non l'emisfero. A quattro il
// campo del 16/09 12Z dava 7 isobare e un centro in tutto il dominio, e il
// dettaglio in piu' non viene dalla griglia del modello -- il campo resta
// quello mediato a scala sinottica -- ma dal passo con cui lo si legge.
assert.match(html, /const ISOBAR_INTERVAL_HPA = 2;/,
  "le isobare non seguono il passo regionale di 2 hPa");
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

// Vento e raffiche condividono una scala sola. Erano due, e lo stesso valore
// usciva di due colori diversi a seconda del livello scelto: impossibile
// vedere quanto la raffica superi la media, che e' la lettura per cui i due
// livelli esistono.
{
  assert.ok(!/const WIND_STOPS = \[/.test(html),
    "il vento e' tornato ad avere una scala tutta sua");
  assert.ok(!/const GUST_STOPS = \[/.test(html),
    "le raffiche sono tornate ad avere una scala tutta loro");

  const windSource = html.match(/const WIND_SPEED_STOPS = \[([\s\S]*?)\n {6}\];/);
  assert.ok(windSource, "scala condivisa della velocita' del vento assente");
  const edges = [...windSource[1].matchAll(/\{ v: (-?[\d.]+), c:/g)]
    .map((match) => Number(match[1]));
  // I bordi restano i gradi Beaufort in km/h -- ed e' il motivo per cui la
  // scala vale anche sulla media: Beaufort e' definito sul vento medio.
  assert.deepEqual(edges, [0, 12, 20, 29, 39, 50, 62, 75, 89, 103, 118, 140],
    "i bordi delle fasce non sono piu' i gradi Beaufort in km/h");
  assert.match(windSource[1], /\{ v: 0, c: \[236, 242, 244\] \}/,
    "la calma non e' quasi bianca");
  assert.match(windSource[1], /\{ v: 50, c: \[214, 199, 74\] \}/,
    "manca lo scalino del vento forte a 50 km/h");
  assert.match(windSource[1], /\{ v: 140, c: \[70, 22, 92\] \}/,
    "l'estremo oltre scala torna bianco e diventa invisibile");

  // Nessuna fascia in piu' ai gradi bassi: fra il bianco della calma e il
  // verde acqua dei 20 km/h ci sono 20,5 unita' CAM02-UCS in tutto, e quattro
  // fasce disterebbero 5,1 -- sotto i 7,6 che la scala si e' data.
  assert.ok(!edges.includes(1) && !edges.includes(6),
    "aggiunte fasce che il divario percettivo non sostiene");

  const windInfo = html.match(/\n {8}wind: \{[\s\S]*?\n {8}\},/);
  assert.ok(windInfo, "il layer del vento non e' in LAYER_INFO");
  assert.match(windInfo[0], /stops: WIND_SPEED_STOPS,/,
    "il vento non usa la scala condivisa");

  const paletteGenerator = fs.readFileSync(
    path.join(root, "scripts", "generate_palettes.py"), "utf8"
  );
  assert.match(paletteGenerator, /out\["WIND_SPEED_STOPS"\] = \[/,
    "il generatore non conosce la scala condivisa");
  assert.doesNotMatch(paletteGenerator, /out\["WIND_STOPS"\]|out\["GUST_STOPS"\]/,
    "il generatore puo' ancora ripristinare le due scale separate");
  assert.doesNotMatch(paletteGenerator, /out\["WIND_ANCHORS"\]/,
    "il generatore conserva ancora gli ancoraggi obsoleti");
}

// Campo fitto e regolare come nelle mappe vettoriali dei modelli: le frecce
// coprono la mappa invece di essere sparse, e il flusso si legge come un
// disegno continuo perche' frecce vicine puntano dalla stessa parte.
assert.match(html, /spacingMobile: 20,/,
  "le frecce sono troppo rade sul telefono");
assert.match(html, /spacingDesktop: 23,/,
  "le frecce sono troppo rade sullo schermo grande");
assert.match(html, /minSpeedKmh: 0\.8,/,
  "sotto la calma la freccia viene disegnata lo stesso");
// Un bordo scuro sottile separa la freccia dal campo colorato sotto senza
// scolorire i blu e i ciano.
assert.match(html, /vectorContext\.strokeStyle = "rgba\(1,7,18,0\.72\)"/,
  "il bordo scuro della freccia e' assente");
assert.match(html, /vectorContext\.lineWidth = group\.width \+ 1\.05/,
  "il bordo non e' piu' sottile del corpo della freccia");

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
  const layerInfo = html.match(/\n {8}gust: \{[\s\S]*?\n {8}\},/);
  assert.ok(layerInfo, "il layer delle raffiche non e' in LAYER_INFO");
  assert.match(layerInfo[0], /stops: WIND_SPEED_STOPS,/,
    "le raffiche non usano piu' la scala condivisa con il vento medio");
  assert.match(layerInfo[0], /discrete: true,/,
    "le raffiche devono restare a fasce nette, non sfumate");
  assert.match(layerInfo[0], /unit: "km\/h",/,
    "le raffiche non sono piu' pubblicate in km/h");
  assert.match(html, /data-layer="gust"/,
    "manca la scheda delle raffiche nel pannello dei livelli");
  assert.match(html, /id="detail-gust"/,
    "manca la riga delle raffiche nella lettura del punto");
  // Le griglie non passano piu' per un elenco di nomi scritto a mano (dove
  // la raffica una volta mancava): decodeBinaryStep tratta ogni campo
  // dichiarato nell'header del contenitore binario allo stesso modo,
  // qualunque sia il suo nome.
  const decodeStep = html.match(/function decodeBinaryStep\([\s\S]*?\n {6}\}/);
  assert.ok(decodeStep, "decodeBinaryStep assente");
  assert.match(decodeStep[0], /fields\.forEach/,
    "la decodifica dello step non tratta piu' tutti i campi allo stesso modo");
  assert.doesNotMatch(decodeStep[0], /"temp"[\s\S]*?"gust"/,
    "la decodifica torna a un elenco di campi scritto a mano");
}
// Il campo deve anche esistere: la scala serve a poco se la pipeline non lo
// pubblica. Le raffiche arrivano da vmax_10m, in m/s, e vanno in km/h.
{
  const pipeline = fs.readFileSync(
    path.join(__dirname, "..", "process_data.py"), "utf8");
  assert.match(pipeline, /"gust": \(\s*\n\s*np\.asarray\(wind_gust_10m\) \* 3\.6/,
    "la pipeline non pubblica le raffiche in km/h");
}
// --- Probabilita' di vicinato ---
// La probabilita' non e' un campo del modello ma una grandezza derivata, e il
// sito la deve trattare come tale: griglia propria, scala unica, e una scheda
// che dica che non e' calibrata. Se qualcuna di queste cade, la mappa mostra
// numeri che sembrano piu' solidi di quanto sono.
{
  const scala = html.match(/const PROB_STOPS = \[[\s\S]*?\n {6}\];/);
  assert.ok(scala, "scala delle probabilita' assente");
  assert.match(scala[0], /\{ v: 0, c: \[236, 244, 248\], a: 0 \}/,
    "la probabilita' nulla non e' trasparente: coprirebbe la carta con un velo");
  const bordi = [...scala[0].matchAll(/\{ v: (\d+),/g)].map((m) => Number(m[1]));
  assert.deepEqual(bordi, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "le fasce di probabilita' non sono piu' decili");

  const livelli = ["prob_rain_1", "prob_rain_5", "prob_rain_10", "prob_rain_20",
                   "prob_gust_50", "prob_gust_75"];
  livelli.forEach((chiave) => {
    const blocco = html.match(new RegExp("\\n {8}" + chiave + ": \\{[\\s\\S]*?\\n {8}\\},"));
    assert.ok(blocco, "manca il livello " + chiave);
    assert.match(blocco[0], /stops: PROB_STOPS,/,
      chiave + " non usa la scala comune delle probabilita'");
    assert.match(blocco[0], /unit: "%",/, chiave + " non e' in percentuale");
    assert.match(blocco[0], /prob: "/, chiave + " non punta a un campo pubblicato");
    assert.match(html, new RegExp('data-layer="' + chiave + '"'),
      "manca la scheda di " + chiave + " nel pannello");
  });

  // La griglia e' diradata: campionarla con gli indici del campo nativo
  // sposterebbe il campo di meta' dominio.
  assert.match(html, /function probSample\(key, gx, gy\)/,
    "manca il campionamento sulla griglia diradata");
  assert.match(html, /const sx = \(lon - payload\.lo1\) \/ payload\.dx;/,
    "la probabilita' non viene campionata per coordinate");
  assert.match(html, /if \(usesProbData\(key\)\) return Boolean\(probGrid\(key\)\);/,
    "la disponibilita' della probabilita' non viene verificata");
  assert.match(html, /Frequenza geometrica su una singola corsa/,
    "la scheda non dice che la probabilita' viene da una sola corsa");
  assert.match(html, /probabilità calibrata: nessun archivio la verifica/,
    "la scheda non dichiara che la probabilita' non e' calibrata");
}
// --- Downscaling di quota ---
// Il modello smussa le montagne di centinaia di metri. La correzione legge la
// quota vera dalle stesse tessere DEM della vista 3D e sposta la temperatura
// lungo il profilo del modello. Le prove qui sotto proteggono i punti in cui
// una svista produrrebbe numeri plausibili ma sbagliati.
{
  assert.match(html, /const metres = data\[p\] \* 256 \+ data\[p \+ 1\] \+ data\[p \+ 2\] \/ 256 - 32768;/,
    "la decodifica terrarium non e' piu' quella dello standard");
  // La batimetria e' negativa: senza azzerarla la correzione inventerebbe
  // gradi in mezzo al mare, dove il modello ha giustamente quota zero.
  assert.match(html, /heights\[i\] = metres > 0 \? metres : 0;/,
    "la batimetria negativa non viene azzerata");
  assert.match(html, /image\.crossOrigin = "anonymous";/,
    "senza CORS le tessere si disegnano ma non si leggono");
  assert.match(html, /resolve\(null\);/,
    "una tessera illeggibile deve annullare la correzione, non falsarla");

  // Il profilo viene dal modello, non da un gradiente di manuale: e' l'unico
  // modo di avere il segno giusto sotto un'inversione.
  assert.match(html, /function profileTemperature\(profile, gx, gy, metres\)/,
    "manca l'interpolazione sul profilo del modello");
  assert.doesNotMatch(html, /elevationCorrection[\s\S]{0,600}?0\.0065/,
    "la correzione di quota e' tornata a un gradiente fisso");

  // Guardie sui casi che non vanno corretti.
  assert.match(html, /if \(!\(Math\.abs\(delta\) > 0\.5\) \|\| Math\.abs\(delta\) > 2500\) return 0;/,
    "manca la guardia su dislivelli nulli o assurdi");
  assert.match(html, /const DEM_TILE_BUDGET = 48;/,
    "senza tetto alle tessere una vista larga ne chiederebbe centinaia");

  // Mappa e lettura del punto devono dire la stessa cosa.
  assert.match(html, /if \(key === "temp" && Number\.isFinite\(value\) && elevationDownscalingActive\(\)\)/,
    "la lettura del punto non applica la correzione che applica la mappa");
  // La correzione non ha bisogno di stazioni: vale anche dove non misura
  // nessuno, che e' esattamente la montagna.
  assert.match(html, /if \(!matchingCount && !elevationReady\)/,
    "senza osservazioni il selettore si spegnerebbe anche in montagna");
  // La firma del raster deve seguire la copertura del terreno: ogni tessera
  // nuova cambia il disegno, e senza questo il raster resterebbe quello
  // parziale disegnato con le prime tessere arrivate.
  assert.match(html, /elevationDownscalingActive\(\) \? terrainCoverageStamp : 0,/,
    "la copertura del terreno non entra nella firma del raster");

  // Il difetto misurato: il campionatore del terreno era identificato dal
  // riquadro esatto del raster, quindi ogni spostamento della mappa lo
  // invalidava, annullava la richiesta in volo e ridisegnava senza
  // correzione. Sulla build precedente, dopo uno spostamento i pixel
  // corretti passavano da 429.823 a zero e non tornavano; adesso la
  // copertura appartiene alla cache delle tessere e puo' solo crescere.
  assert.ok(!/terrainSamplerKey/.test(html),
    "il campionatore del terreno e' di nuovo legato al riquadro del raster");
  assert.match(html, /function requestTerrainTiles\(west, east, south, north, width\)/,
    "manca la richiesta di tessere indipendente dal riquadro");
  assert.match(html, /function terrainHeightAt\(longitude, latitude\)/,
    "manca il campionatore di quota sulla cache delle tessere");

  // La correzione di quota deve stare nella funzione pura che finisce nel
  // worker: sul thread principale costava 478 ms per raster, e con essa non
  // si potevano ne' scaldare ne' mettere in cache i fotogrammi. Averla nel
  // worker e' cio' che la rende accendibile di default.
  const pura = html.slice(html.indexOf("function fillRasterPixels(params)"));
  assert.match(pura.slice(0, 4000), /const elevation = params\.elevation \|\| null;/,
    "la correzione di quota non e' nella funzione pura del worker");
  assert.ok(!/fillRasterPixelsWithElevation/.test(html),
    "esiste ancora una copia della funzione di disegno sul thread principale");
  assert.match(html, /profileTemperature, profileVectorAt,\n\s+elevationCorrection, windElevationFactor, fillRasterPixels/,
    "il worker non riceve le funzioni della correzione di quota");

  // Non dipende piu' dal selettore delle osservazioni: e' attiva di suo.
  assert.ok(!/if \(!showFusion \|\| selectedLevel !== "surface" \|\| !payload\) return/.test(html),
    "la correzione di quota dipende ancora dal selettore delle osservazioni");
  assert.match(html, /if \(selectedLevel !== "surface" \|\| !payload\) return "";/,
    "elevationCorrectionKind non e' piu' riconoscibile");
}
// --- Rete osservativa su tutta Italia ---
// La rete e' passata da 455 stazioni della sola Sicilia a circa 3640 su 27
// reti regionali. A quel numero due dettagli diventano decisivi.
{
  // Il confronto di ogni stazione con tutte le precedenti costava 299 ms su
  // 3399 stazioni; a secchielli ne costa 8, scartando le stesse.
  assert.match(html, /const secchi=new Map\(\);/,
    "il controllo dei doppioni e' tornato quadratico");
  assert.doesNotMatch(html, /points\.some\(p=>MeteoLocalDownscaling\.distance/,
    "il confronto tutti-con-tutti e' rientrato");
  assert.match(html, /if\(gia\(station\)\) continue;/,
    "le stazioni sovrapposte non vengono piu' scartate");
}
// --- Le due correzioni non sono la stessa cosa ---
// La quota del terreno non invecchia e vale a ogni scadenza; le stazioni
// misurano adesso e valgono solo vicino all'ora corrente. Il pannello deve
// dirlo, perche' "nessuna cella correggibile" in previsione sembrava un guasto
// mentre e' il comportamento corretto.
{
  assert.match(html, /const FUSION_TIME_TOLERANCE_MS = 45 \* 60 \* 1000;/,
    "la fusione non e' piu' vincolata alla validita' della scadenza");
  assert.match(html, /Math\.abs\(station\.obsTime - validTime\) <= FUSION_TIME_TOLERANCE_MS/,
    "un'osservazione di adesso finirebbe su una previsione di domani");
  assert.match(html, /le stazioni misurano adesso/,
    "il pannello non spiega perche' in previsione la fusione si spegne");
  assert.match(html, /function updateDownscalingStatus\(\)/,
    "manca lo stato che distingue quota e osservazioni");
  assert.match(html, /scarto medio dal modello/,
    "lo stato non riporta quanto pesa la correzione di quota");
  assert.match(html, /Quota vera del terreno · /,
    "lo stato non distingue la quota, sempre attiva, dalle osservazioni");
}
// --- Quota neve e zero termico ---
{
  const scala = html.match(/const ALTITUDE_STOPS = \[[\s\S]*?\n {6}\];/);
  assert.ok(scala, "scala delle quote assente");
  const bordi = [...scala[0].matchAll(/\{ v: (\d+),/g)].map((m) => Number(m[1]));
  assert.equal(bordi[0], 0, "la scala delle quote non parte dal livello del mare");
  assert.equal(bordi[bordi.length - 1], 4500, "la scala non arriva a 4500 m");
  ["snow_level", "freezing_level"].forEach((chiave) => {
    const blocco = html.match(new RegExp("\\n {8}" + chiave + ": \\{[\\s\\S]*?\\n {8}\\},"));
    assert.ok(blocco, "manca il livello " + chiave);
    assert.match(blocco[0], /stops: ALTITUDE_STOPS,/,
      chiave + " non usa la scala delle quote");
    assert.match(blocco[0], /unit: "m",/, chiave + " non e' in metri");
    assert.match(html, new RegExp('data-layer="' + chiave + '"'),
      "manca la scheda di " + chiave);
  });
  // La quota neve viene dal bulbo bagnato: se tornasse allo zero termico
  // sbaglierebbe di centinaia di metri in aria secca, sempre verso l'alto.
  assert.match(html, /bulbo bagnato passa per lo zero, non lo zero termico/,
    "la quota neve non dichiara piu' di venire dal bulbo bagnato");
  assert.match(html, /il valore è ignoto, non alto/,
    "la scheda non dice che sopra il profilo la quota neve e' ignota");
  // Su una griglia da 17 km la regola severa sui NaN faceva un gradino
  // visibile: il campionatore tollerante vale solo li'.
  assert.match(html, /function sampleCoarseBilinear\(array, gx, gy, nx, ny\)/,
    "manca il campionatore per le griglie diradate");
  assert.match(html, /return weight >= 0\.25 \? sum \/ weight : NaN;/,
    "un valore stiracchiato da un angolo solo passerebbe per buono");
  // Il raster sceglie il campionatore per nome (serve al worker, che non
  // puo' ricevere una funzione via postMessage) e lo risolve dentro
  // fillRasterPixels: le due meta' della stessa garanzia vanno controllate
  // insieme, altrimenti il nome potrebbe non corrispondere a nulla.
  assert.match(html, /usesProfileData\(activeLayer\) \? "coarseBilinear" : "bilinear"/,
    "il raster non sceglie il campionatore tollerante sulle griglie diradate");
  assert.match(html, /samplerKind === "coarseBilinear" \? sampleCoarseBilinear : sampleBilinear/,
    "il nome del campionatore tollerante non viene piu' risolto alla funzione vera");
}

// --- Il riempimento del raster fuori dal thread principale -----------------
// fillRasterPixels deve restare pura (nessuna lettura di stato esterno):
// e' la sua stessa fonte, non una copia, a diventare lo script del worker.
{
  const fillPure = html.match(/function fillRasterPixels\([\s\S]*?\n {6}\}/);
  assert.ok(fillPure, "fillRasterPixels assente");
  ["map\\.", "currentData\\.", "terrainHeightAt\\(", "window\\."].forEach((pattern) => {
    assert.doesNotMatch(fillPure[0], new RegExp(pattern),
      "fillRasterPixels non e' piu' pura: legge " + pattern + " dal thread principale");
  });
  assert.match(fillPure[0], /new Uint8ClampedArray\(width \* height \* 4\)/,
    "fillRasterPixels non produce piu' un buffer di pixel autonomo");

  // La correzione di quota viaggia dentro la stessa funzione pura: le quote
  // arrivano gia' calcolate nei parametri, quindi il DEM non serve al worker.
  assert.match(fillPure[0], /windElevationFactor\(profile, high, low/,
    "la correzione di quota del vento e' sparita dal disegno");
  assert.match(fillPure[0], /elevationCorrection\(profile, high, low/,
    "la correzione di quota della temperatura e' sparita dal disegno");

  // Un solo sorgente per i due worker -- quello che disegna adesso e quello
  // che scalda i fotogrammi vicini -- perche' due liste copiate a mano
  // divergono alla prima funzione aggiunta da una parte sola.
  const buildWorker = html.match(/function rasterWorkerSource\([\s\S]*?\n {6}\}/);
  assert.ok(buildWorker, "rasterWorkerSource assente");
  ["clamp", "getGrid", "sampleNearest", "sampleBilinear", "sampleCoarseBilinear",
   "colorFor", "profileTemperature", "profileVectorAt", "elevationCorrection",
   "windElevationFactor", "fillRasterPixels"].forEach((name) => {
    assert.match(buildWorker[0], new RegExp("\\b" + name + "\\b"),
      "il worker del raster non porta con se' " + name);
  });
  assert.match(buildWorker[0], /\.toString\(\)/,
    "il worker copia le funzioni a mano invece di estrarne la fonte vera");
  // Il campo delle quote pesa megabyte e non cambia scorrendo la barra del
  // tempo: si manda una volta per vista, non a ogni fotogramma.
  assert.match(buildWorker[0], /heldTerrain\.key === data\.terrainKey/,
    "il worker non riusa il campo delle quote gia' ricevuto");
  assert.match(html, /function rasterMessage\(params, state\)/,
    "manca il messaggio che allega il campo delle quote solo quando serve");

  // Un raster superato da uno piu' recente non deve mai arrivare sullo
  // schermo: stesso principio del loadingToken di loadStep.
  const dispatch = html.match(/function dispatchRasterFill\([\s\S]*?\n {6}\}/);
  assert.ok(dispatch, "dispatchRasterFill assente");
  assert.match(dispatch[0], /\+\+rasterRenderToken/,
    "dispatchRasterFill non genera un token per scartare i risultati superati");
  const workerBody = html.match(/function buildRasterWorker\([\s\S]*?\n {6}\}/);
  assert.ok(workerBody, "buildRasterWorker assente");
  assert.match(workerBody[0], /event\.data\.token !== rasterRenderToken/,
    "il worker non scarta piu' i risultati di un raster superato");

  // Un solo percorso di disegno: il caso con correzione di quota e quello
  // senza vanno entrambi al worker, altrimenti i fotogrammi corretti non si
  // possono ne' mettere in cache ne' scaldare in anticipo.
  const render = html.match(/function renderWeather\(prewarm, onReady\) \{[\s\S]*?\n {6}\}/);
  assert.ok(render, "renderWeather assente");
  assert.ok(!/if \(correctElevation\) \{/.test(html),
    "renderWeather ha di nuovo un percorso separato sul thread principale");
  assert.match(render[0], /dispatchRasterFill/,
    "renderWeather non manda piu' il disegno al worker");
}

// --- Vento riportato sulla quota vera ---
// Stesso ragionamento della temperatura e stessa assenza di parametri liberi:
// al Gran Sasso il modello calcola il vento sopra i suoi 2078 m, non sopra i
// 2912 veri. Si usa il RAPPORTO fra il vento del profilo alle due quote, non
// la differenza, perche' il rapporto cancella a primo ordine lo scarto fra il
// vento a 10 m -- che sente l'attrito -- e quello dell'aria libera.
{
  assert.match(html, /function windElevationFactor\(profile, trueHeight, modelHeight,/,
    "manca il fattore di quota per il vento");
  assert.match(html, /return clamp\(high \/ low, WIND_ELEVATION_LIMITS\[0\], WIND_ELEVATION_LIMITS\[1\]\);/,
    "il fattore del vento non e' piu' un rapporto limitato");
  assert.match(html, /const WIND_ELEVATION_LIMITS = \[0\.5, 2\.0\];/,
    "senza limiti un profilo quasi calmo farebbe esplodere il rapporto");
  assert.match(html, /if \(!Number\.isFinite\(high\) \|\| !Number\.isFinite\(low\) \|\| low < 1\) return 1;/,
    "con vento quasi nullo alla quota del modello il rapporto va evitato");
  // Colore della mappa e freccia devono raccontare la stessa velocita'.
  assert.match(html, /&& elevationCorrectionKind\(\) === "wind"\) \{/,
    "le frecce mostrerebbero una velocita' diversa dal colore");
  // Il vento si sposta solo se il profilo porta davvero il vento, e la
  // raffica si sposta con lui: e' la stessa velocita' vista al suo massimo,
  // quindi la vetta vera che sporge in un flusso piu' veloce la riguarda
  // esattamente come la media.
  assert.match(html, /if \(\(activeLayer === "wind" \|\| activeLayer === "gust"\) && payload\.wind\) \{/,
    "il vento verrebbe corretto anche senza il profilo del vento");

  // Quale correzione applicare non si deduce dalla forma del campo: la
  // raffica e' uno scalare, e dedurlo l'avrebbe spostata con la legge della
  // temperatura invece che con il rapporto dei venti.
  assert.match(html, /const windKind = Boolean\(elevation && elevation\.kind === "wind"\);/,
    "il tipo di correzione torna a dipendere dalla forma del campo");

  // Le letture puntuali passano dalle stesse due quote del raster: se
  // divergessero, il colore direbbe un valore e il punto cliccato un altro.
  assert.match(html, /function elevationPairAt\(longitude, latitude\)/,
    "manca la sorgente unica delle due quote per le letture puntuali");
  assert.match(html, /profilePayload\(\), heights\.trueHeight, heights\.modelHeight,/,
    "le letture puntuali non usano piu' le due quote vere");
}
// --- Pannello del satellite: selettore e barra del tempo convivono ---
// Il selettore del prodotto (infrarosso, fase delle nubi, polvere...) non sta
// nel markup dove lo si vede: modern-ui.js lo SPOSTA dentro
// #satellite-controls. Costruendo li' la barra del tempo con innerHTML lo si
// cancellava, e con esso l'unico modo di cambiare canale -- misurato sulla
// build pubblicata: selettore assente, tredici canali irraggiungibili.
{
  const costruisci = html.match(/function buildSatelliteControls\([\s\S]*?\n {6}\}/);
  assert.ok(costruisci, "buildSatelliteControls assente");
  assert.ok(!/host\.innerHTML\s*=/.test(costruisci[0]),
    "la barra del tempo torna a svuotare il contenitore, cancellando il "
    + "selettore del prodotto satellitare che modern-ui.js ci sposta dentro");
  assert.match(costruisci[0], /host\.appendChild\(/,
    "la barra del tempo non si aggiunge al contenitore");

  const interfaccia = fs.readFileSync(path.join(root, "modern-ui.js"), "utf8");
  assert.match(interfaccia, /getElementById\('satellite-controls'\)\.append\(satellitePicker\)/,
    "il selettore del prodotto non viene piu' spostato nel pannello");
  assert.match(html, /id="satclouds-select"/, "manca il selettore del prodotto");
}

// --- Scorrimento fluido dell'osservato ---
// "Senza caricamenti" non si ottiene chiedendo piu' in fretta: EUMETView
// limita a venti richieste per finestra. Si ottiene non chiedendo affatto,
// cioe' ripubblicando dalla memoria. Misurato: sei scorrimenti consecutivi,
// ZERO richieste nuove.
{
  assert.match(html, /const cloudFrames = new Map\(\);/,
    "manca la cache dei fotogrammi osservati");
  assert.match(html, /function pubblicaFotogrammaInCache\(\)/,
    "manca la pubblicazione senza rete");
  const scorrimento = html.match(/scrub\.addEventListener\("input"[\s\S]*?\n {8}\}\);/);
  assert.ok(scorrimento, "manca il gestore dello scorrimento");
  assert.match(scorrimento[0], /if \(pubblicaFotogrammaInCache\(\)\)/,
    "lo scorrimento non prova piu' la cache prima della rete");
  // Il precaricamento deve restare UNO ALLA VOLTA e distanziato, o brucia
  // l'intera finestra di richieste del servizio in un gesto.
  assert.match(html, /const CLOUD_PREFETCH_GAP_MS = \d{3,};/,
    "il precaricamento non ha piu' una pausa fra una richiesta e l'altra");
  assert.match(html, /if \(cloudPrefetchTimer \|\| cloudPrefetchBusy\) return;/,
    "il precaricamento puo' partire in parallelo con se stesso");
  // Sfrattare il fotogramma in mostra ne libererebbe l'object URL, e la
  // mappa resterebbe vuota.
  assert.match(html, /if \(candidata !== cloudPublishedKey\)/,
    "lo sfratto dalla cache puo' cancellare l'immagine in mostra");
}

console.log("3D map regression checks: OK");

// --- fluidita' sul telefono -------------------------------------------
// Misurato su telefono emulato (390x844, densita' 3, CPU rallentata sei
// volte) con quaranta scariche vive: 43,5 ms a fotogramma prima, 17,0 dopo.
// La leva e' una sola, e non e' quella che sembrava: non il numero di
// scariche (quaranta o dieci cambiava 13 fps contro 12) ne' lo shader
// volumetrico (spegnerlo adesso non sposta la mediana), ma quanti pixel
// hanno le canvas a tutto schermo, che vengono cancellate e ricaricate a
// OGNI fotogramma finche' un lampo brilla.
assert.match(html, /const soloLampi = mobile && weatherView === "satellite";/,
  "la densita' delle canvas non distingue piu' la vista satellite");
assert.match(html, /const tetto = soloLampi \? 1\.25 : \(mobile \? 2 : 2\.15\);/,
  "la densita' ridotta non e' piu' riservata al telefono in vista satellite");
// Il PC non deve cambiare: 2.15 resta 2.15, e nessun ramo lo tocca.
assert.doesNotMatch(html, /const ratio = Math\.min\(window\.devicePixelRatio \|\| 1, mobile \? [0-9.]+ : [0-9.]+\);/,
  "la densita' torna a essere decisa senza guardare la vista");
// Cambiando vista le canvas vanno rimisurate, o la densita' nuova non
// arriva mai: e' il passo che rende effettiva la riga qui sopra.
assert.match(modernUi, /resizeCanvases\(\);\s*\n\s*document\.body\.classList\.toggle\('satellite-view'/,
  "cambiando vista le canvas non vengono rimisurate: la densita' resta quella di prima");

// La maschera delle nubi si ricostruisce a ogni fotogramma satellitare, ed
// e' un blocco unico del thread. Sul telefono si dimezza il lato: a 390
// punti di schermo, 384 texel restano circa un texel per punto.
const lati = new Function('window', 'MOBILE_BREAKPOINT',
  implementazione('isMobile') + '\n' + implementazione('mascheraLato') + '\n'
  + implementazione('volumeLato')
  + '\nreturn {mascheraLato, volumeLato};');
const suTelefono = lati({ innerWidth: 390 }, 900);
const suPc = lati({ innerWidth: 1440 }, 900);
assert.ok(suTelefono.mascheraLato() < suPc.mascheraLato(),
  "la maschera delle nubi non e' piu' leggera sul telefono");
assert.equal(suPc.mascheraLato(), 768,
  "il lato della maschera sul PC e' cambiato: non doveva");
// Griglia volumetrica e maschera devono avere la STESSA scala: tenere il
// volume a 768 con la maschera a 384 non aggiunge dettaglio, lo inventa
// ingrandendo, e intanto quadruplica la texture che la GPU campiona a
// ogni fotogramma.
assert.equal(suTelefono.volumeLato(), suTelefono.mascheraLato(),
  "sul telefono il volume ingrandisce una maschera piu' piccola: dettaglio inventato e texture quadrupla");
assert.equal(suPc.volumeLato(), suPc.mascheraLato(),
  "sul PC volume e maschera non hanno piu' la stessa scala");

// Il tetto sulla taglia dell'immagine satellitare resta sulla sola
// larghezza. Provato a metterlo anche sull'altezza: sul telefono cambiava
// quasi niente (2,9 megapixel invece di 3,0) e sul PC tagliava l'immagine
// da 1920 a 1203 pixel, cioe' peggiorava proprio il caso da non toccare.
// E si verifica ESEGUENDO, non cercando una scrittura: la prima stesura di
// questa prova cercava il nome di una variabile, e una riscrittura
// equivalente le sarebbe passata sotto il naso -- provato, e infatti passava.
const taglia = (larghezza, altezza, densita) => new Function(
  'document', 'window', 'navigator',
  [html.match(/const CLOUD_MAX_SIDE = \d+;/)[0],
   implementazione('mercatorMetresX'), implementazione('mercatorMetresY'),
   implementazione('cloudRequestSize')].join('\n\n')
  + '\nreturn cloudRequestSize;')(
    { getElementById: () => ({ clientWidth: larghezza, clientHeight: altezza }) },
    { innerWidth: larghezza, innerHeight: altezza, devicePixelRatio: densita },
    { deviceMemory: 8 });
// La proprieta' esatta da difendere e' questa: la larghezza richiesta non
// deve dipendere dall'ALTEZZA dello schermo. Un tetto sull'altezza la
// farebbe dipendere, ed e' quello che tagliava il PC da 1920 a 1203.
// (Prima stesura sbagliata: avevo scelto un riquadro dove a comandare era
// il dettaglio nativo dello strumento, non il tetto, quindi la prova
// falliva sull'albero pulito. Qui si usa il dominio osservato, dove il
// tetto dello schermo comanda davvero.)
const dominioOsservato = (() => {
  const m = html.match(/const CLOUD_DOMAIN = \{ west: (-?[0-9.]+), south: (-?[0-9.]+), east: (-?[0-9.]+), north: (-?[0-9.]+) \};/);
  assert.ok(m, "manca CLOUD_DOMAIN");
  return { west: +m[1], south: +m[2], east: +m[3], north: +m[4] };
})();
const schermoAlto = taglia(1280, 860, 1)(dominioOsservato, { metres: 1000 });
const schermoBasso = taglia(1280, 300, 1)(dominioOsservato, { metres: 1000 });
assert.ok(schermoAlto.width >= 1900,
  "sul PC il tetto dello schermo non arriva piu' a 1920 pixel: chiede "
  + schermoAlto.width);
assert.equal(schermoAlto.width, schermoBasso.width,
  "la larghezza richiesta cambia con l'ALTEZZA dello schermo ("
  + schermoAlto.width + " contro " + schermoBasso.width
  + "): un tetto sull'altezza sta mordendo, ed e' quello che peggiorava il PC");
