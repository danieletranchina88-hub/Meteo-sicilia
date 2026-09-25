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
assert.match(html, /data-toggle="satclouds"[\s\S]{0,500}?<b>Satellite MTG<\/b>/,
  "manca il controllo satellitare indipendente");
assert.match(html, /data-toggle="radar"[\s\S]{0,500}?<b>Radar precipitazioni<\/b>/,
  "manca il controllo radar indipendente");
assert.match(html, /radar: showRadar,[\s\S]{0,120}?satclouds: showSatelliteClouds,/,
  "gli stati indipendenti non sono riportati nell'interfaccia");
assert.match(html, /setWeatherView\(weatherView === "satellite"/,
  "il satellite non usa la vista indipendente");
// Le nubi si riaggiornano da sole a meta' della cadenza del prodotto. Il
// giro adesso rilegge PRIMA la dichiarazione del servizio: e' quella ad
// avanzare, e senza rileggerla si continuerebbe a chiedere lo stesso istante
// all'infinito, cioe' la diretta si fermerebbe.
assert.match(
  html,
  /cloudTimer = setInterval\(function \(\) \{\s*\n\s*if \(document\.hidden\) return;\s*\n\s*aggiornaIstantiDisponibili\(false\)\.then\(function \(avanzato\) \{\s*\n\s*loadSatelliteClouds\(false\);/,
  "le nubi non si aggiornano da sole: non sarebbero in tempo reale"
);
assert.match(
  html,
  /\}, \(product\.slotMs \|\| CLOUD_SLOT_MS\) \/ 2\);/,
  "il controllo non segue piu' la cadenza del prodotto"
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
  assert.match(scorrimento[0], /pubblicaFotogrammaInCache\(\)/,
    "lo scorrimento non prova piu' la cache delle nubi prima della rete");
  assert.match(scorrimento[0], /pubblicaLightningInCache\(\)/,
    "lo scorrimento non prova la cache dei fulmini prima della rete");
  // Il precaricamento deve restare UNO ALLA VOLTA e distanziato, o brucia
  // l'intera finestra di richieste del servizio in un gesto.
  assert.match(html, /const CLOUD_PREFETCH_GAP_MS = \d{3,};/,
    "il precaricamento non ha piu' una pausa fra una richiesta e l'altra");
  assert.match(html, /if \(cloudPrefetchTimer \|\| cloudPrefetchBusy\) return;/,
    "il precaricamento puo' partire in parallelo con se stesso");
  // Sfrattare il fotogramma in mostra ne libererebbe l'object URL, e la
  // mappa resterebbe vuota.
  assert.match(html, /if \(candidata !== cloudPublishedKey\s*&&\s*candidata !== lightningPublishedKey\)/,
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

// LA SAGOMA DELLE NUBI. La regola di prima vietava del tutto di ricavare
// una maschera dal raster satellitare, per due ragioni: costava, ed era
// fisicamente ingannevole. La prima e' stata misurata e non regge -- la
// sagoma si costruisce una volta per fotogramma satellitare, cioe' ogni
// cinque o dieci minuti (23 ms a caldo), e il disegno per fotogramma passa
// da 0,10 a 0,36 ms. La seconda regge solo per una parte, e quella parte
// resta vietata qui sotto.
//
// Quello che sarebbe ingannevole e' attribuire alla rete a terra una
// grandezza che non possiede: energia ottica, radianza, footprint. Dire
// invece DOVE C'E' NUBE non e' una ricostruzione: e' la lettura della
// fotografia che l'utente sta gia' guardando sotto i fulmini. L'ampiezza
// della luce viene dalla rete (posizione, tempo, scariche vive); la sua
// FORMA viene dal satellite. Le due cose restano separate, e ognuna dice
// solo quello che sa.
// Il divieto e' sul CALCOLARLE, non sul nominarle: il renderer dichiara in
// un commento proprio quello che la rete non fornisce, e quel commento deve
// restare.
assert.doesNotMatch(html,
  /(?:const|let|var|function)\s+\w*(?:radianza|radiance|footprint)|\.\s*(?:radianza|radiance|footprint)\b/i,
  "il live attribuisce alla rete a terra una grandezza ottica che non ha");
assert.match(html, /Non fornisce geometria del canale, energia ottica,\s*\n\s*\/\/ footprint/,
  "il renderer non dichiara piu cosa la rete a terra NON misura");
// La sagoma si ricava solo dai canali dove chiaro vuol dire davvero denso.
// Sui compositi diagnostici -- fase, tipo di nube, polvere, neve -- il
// colore e' una diagnosi, e usarne la luminanza sarebbe fisica finta.
assert.match(html, /\["scene", "grey"\]\.includes\(product\.mode\)/,
  "la sagoma viene ricavata anche dai compositi diagnostici");
// E si costruisce dove l'immagine passa gia' da una canvas nostra, cioe'
// una volta per fotogramma satellitare, non a ogni lampo.
assert.doesNotMatch(implementazione('disegnaAttivita'), /costruisciMascheraNube/,
  "la sagoma viene ricostruita a ogni fotogramma di animazione");
assert.match(implementazione('publishSatelliteClouds'), /costruisciMascheraNube\(canvas, box, product\)/,
  "la sagoma non viene piu ricavata dove l'immagine e gia' in mano");

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

// updateSkyPalette a mappa INCLINATA. Il ramo che colora lo sfondo gira solo
// in prospettiva, e per mesi ha contenuto un riferimento a `onDark`, una
// variabile tolta quando il fondo e' diventato sempre la carta chiara. Finche'
// l'inclinazione era solo quella del 3D non se ne accorgeva nessuno; quando
// "Nubi in volume" ha cominciato a inclinare la mappa, il ReferenceError
// fermava attivaNubiVolumetriche prima di costruire il volume, e le nubi non
// comparivano ne' su PC ne' su telefono. La funzione qui gira davvero, in
// tutte e tre le combinazioni, con una mappa finta che ha lo sfondo.
{
  const vm = require("node:vm");
  const sorgente = implementazione("updateSkyPalette");
  for (const [show3D, showVolumeClouds] of [[true, false], [false, true], [true, true]]) {
    const dipinti = {};
    const contesto = {
      show3D, showVolumeClouds, synopticChart: false,
      mapBackgroundColour: () => "rgb(0,0,0)",
      document: { documentElement: { style: { setProperty: () => {} } } },
      map: {
        setSky: () => {},
        getLayer: (id) => id === "background" ? {} : undefined,
        setPaintProperty: (id, chiave, valore) => { dipinti[id + ":" + chiave] = valore; }
      }
    };
    vm.createContext(contesto);
    vm.runInContext(sorgente, contesto);
    assert.doesNotThrow(() => contesto.updateSkyPalette(),
      "updateSkyPalette si rompe a mappa inclinata (show3D=" + show3D
      + ", showVolumeClouds=" + showVolumeClouds + "): le nubi in volume non partono");
    assert.ok(dipinti["background:background-color"],
      "a mappa inclinata lo sfondo non viene piu' colorato");
  }
}

// L'inclinazione e' contorno: se un suo errore arriva fino a
// attivaNubiVolumetriche, il volume non viene costruito. Deve restare
// isolata, e costruisciScena deve venire dopo.
{
  const attiva = implementazione("attivaNubiVolumetriche");
  assert.match(attiva, /try \{\s*abilitaInclinazioneVolume\(true\);\s*\} catch/,
    "l'inclinazione delle nubi in volume non e' piu' isolata: un suo errore torna a bloccare il volume");
  assert.ok(attiva.indexOf("costruisciScena()") > attiva.indexOf("abilitaInclinazioneVolume(true)"),
    "costruisciScena non viene piu' dopo l'inclinazione");
}

// Le nubi in volume: i difetti che si vedevano, uno per uno, con la causa.
{
  const vm = require("node:vm");
  const contesto = { Math };
  vm.createContext(contesto);
  vm.runInContext(implementazione("limita") + implementazione("lisciaFra")
    + implementazione("spessoreDellaNube"), contesto);
  // GRADINI ALLA BASE. Lo spessore a rami saltava di chilometri lungo le
  // isolinee della cima, e la base della nube lo seguiva a scalini. Su tutta
  // la griglia di cime e coperture, un passo di dieci metri non deve mai
  // spostare lo spessore di piu' di pochi metri.
  let salto = 0, dove = "";
  for (let copertura = 0; copertura <= 1.0001; copertura += 0.02) {
    for (let cima = 0.2; cima < 16; cima += 0.01) {
      const d = Math.abs(contesto.spessoreDellaNube(cima + 0.01, copertura)
        - contesto.spessoreDellaNube(cima, copertura));
      if (d > salto) { salto = d; dove = "cima " + cima.toFixed(2) + " km, copertura " + copertura.toFixed(2); }
    }
  }
  assert.ok(salto < 0.03, "lo spessore delle nubi salta di " + (salto * 1000).toFixed(0)
    + " m in 10 m di cima (" + dove + "): la base torna a gradini");
  // Le tre morfologie restano: strato basso sottile, cirro lama, torre profonda.
  assert.ok(contesto.spessoreDellaNube(1.5, 0.8) < 1.3, "lo strato basso non e' piu' sottile");
  assert.ok(contesto.spessoreDellaNube(10, 0.25) < 1.5, "il cirro alto non e' piu' una lama");
  assert.ok(contesto.spessoreDellaNube(12, 0.95) > 6, "la torre fredda non e' piu' profonda");
}
{
  const frammento = html.slice(html.indexOf("var FRAMMENTO = ["), html.indexOf("var STESURA = ["));
  // GRANA BIANCA: l'hash per pixel faceva partire ogni raggio a caso.
  assert.doesNotMatch(frammento, /fract\(sin\(dot\(gl_FragCoord/,
    "lo scarto dei raggi e' tornato un hash casuale: la grana bianca torna");
  assert.match(frammento, /float intreccio\(vec2 f\)/, "manca lo scarto a gradiente interlacciato");
  // PUNTE: la cima calava in proporzione alla copertura e ogni cella era un cono.
  assert.doesNotMatch(frammento, /scemare/, "la cima torna a scendere a cono verso il bordo");
  // La forma delle cupole e la continuita del nucleo sono verificate
  // numericamente in test_map_3d.js (anche su GPU con --gpu).
  // STRIATURE RADIALI: il passo cresce con la distanza ma copre sempre
  // tutto il raggio con i passi rimasti.
  assert.match(frammento, /float crescita = 0\.004;/, "il passo non cresce piu' con la distanza");
  assert.match(frammento, /\(tLontano - t\) \/ float\(max\(1, uPassi - i\)\)/,
    "il raggio puo' esaurire i passi prima di attraversare la nube");
  // La sagoma viene dalla texture di fusione, letta al livello del pixel.
  assert.match(frammento, /vec4 campo = textureLod\(uCampo, uv, lodCampo\);/,
    "la texture delle nubi 3D non si legge piu' al livello del pixel");
  assert.doesNotMatch(frammento, /texture\(uRumore, q \* 3\.7/,
    "e' tornata l'ottava fine del rumore, quella che faceva i puntini");
  // Il lampo deve poter schiarire anche una nube al sole.
  assert.match(frammento, /colore = alfa \* clamp\(\(x \* \(2\.51 \* x \+ 0\.03\)\) \/ \(x \* \(2\.43 \* x \+ 0\.59\) \+ 0\.14\), 0\.0, 1\.0\);/,
    "manca la curva di risposta: il bordo d'argento torna a tagliare");
}
{
  // ANELLI, seconda causa: la quota a otto bit. Il campo va in mezza precisione.
  const pubblica = html.slice(html.indexOf("StratoVolume.prototype.pubblicaCampo"),
    html.indexOf("StratoVolume.prototype.matriceDi"));
  assert.match(pubblica, /gl\.RGBA16F/, "il campo del volume e' tornato a otto bit: le cupole si rigano");
  // QUADRETTI: la copertura letta col pixel piu' vicino.
  const superficie = implementazione("superficieSenzaNubi");
  assert.doesNotMatch(superficie, /Math\.round\(y \* campo\.altezza/,
    "la superficie torna a leggere la copertura a quadretti");
  const ottica = implementazione("integraCoperturaOttica");
  assert.match(ottica, /otticaIn\(mx, my\)/, "la maschera GeoColour torna a essere letta a quadretti");
  // Il volume dal telefono: disegnato in una texture di dimensione limitata.
  assert.match(html, /StratoVolume\.prototype\.prerender = function/,
    "il volume non si disegna piu' nella sua texture: sul telefono costa il quadruplo");
}
{
  // Le cime allargate non superano mai la cima piu' alta dei dintorni e
  // lasciano stare le nubi basse.
  const vm = require("node:vm");
  // Stessa scala del campo vero sul PC: circa 2,6 km per pixel.
  const contesto = { Math, Float32Array, DOMINIO: { ovest: 0, est: 54 * 200 / 1600 },
    CIMA_ALLARGA_KM: 14, CIMA_ALLARGA_PESO: 0.62 };
  vm.createContext(contesto);
  vm.runInContext(implementazione("limita") + implementazione("lisciaFra")
    + implementazione("sfoca") + implementazione("massimoScorrevole")
    + implementazione("allargaCime"), contesto);
  const w = 200, h = 120, quota = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const r = Math.hypot(x - 60, y - 60);
    quota[y * w + x] = Math.max(0, 13 * Math.exp(-r * r / 40));  // una torre a punta
    if (x > 150) quota[y * w + x] = 1.5;                           // uno strato basso
  }
  const fuori = contesto.allargaCime(quota, w, h);
  let massimo = 0;
  for (const v of fuori) massimo = Math.max(massimo, v);
  assert.ok(massimo <= 13 + 1e-4, "l'allargamento ha alzato la cima oltre la misura");
  assert.ok(fuori[60 * w + 64] > quota[60 * w + 64] + 1, "la torre a punta non si allarga in cupola");
  assert.ok(Math.abs(fuori[60 * w + 180] - 1.5) < 1e-4, "lo strato basso e' stato alzato");
}
console.log("nubi in volume: spessore continuo, niente grana, niente coni, niente quadretti");

// Nubi in volume dalle misure EUMETSAT: dove (maschera CLM), a che quota
// (Cloud Top Height), quanto opache. Il modulo gira davvero, su immagini
// sintetiche con i colori dei prodotti.
{
  const vm = require("node:vm");
  const inizio = html.indexOf("const NubiVolumetriche = (function");
  const corpo = html.slice(html.indexOf('"use strict";', inizio) + 13,
    html.indexOf("      // --- il volume, attraversato dai raggi", inizio));
  const contesto = { Math, Float32Array, Float64Array, Uint32Array, Uint8Array, Date,
    isMobile: () => false, costruisciMascheraNube: () => null, CLOUD_PRODUCTS: {} };
  vm.createContext(contesto);
  vm.runInContext("this.m = (function () {" + corpo + ";return { mascheraDaClm, quoteDaCth,"
    + " campoMisurato, componiCopertura, dettaglioDaGeoColour, tessituraDelCampo, CTH_SCALA,"
    + " QUOTA_SCALA_KM, classificaNubi, classificaGriglia, GENERI };})();", contesto);
  const m = contesto.m;
  const immagine = (w, h, colore) => {
    const px = new Uint8Array(w * h * 4);
    for (let k = 0; k < w * h; k++) {
      const c = typeof colore === "function" ? colore(k % w, Math.floor(k / w)) : colore;
      px.set(c, k * 4);
    }
    return px;
  };

  // La maschera: il rosso e' la frazione di nube.
  const clm = m.mascheraDaClm(new Uint8Array([
    255, 255, 255, 255,  0, 192, 0, 255,  0, 0, 255, 255,  255, 255, 255, 0,  128, 224, 128, 255]), 5);
  assert.deepEqual(Array.from(clm).map((v) => Math.round(v * 100) / 100), [1, 0, 0, 0, 0.5],
    "la maschera nubi CLM non si legge piu' come frazione di nube");

  // La quota: ogni colore della legenda torna la sua quota.
  let peggiore = 0;
  for (const [metri, r, g, b] of m.CTH_SCALA) {
    if (metri < 600) continue;  // i primi gradini sono quasi neri: indistinguibili
    const q = m.quoteDaCth(immagine(12, 12, [r, g, b, 255]), 12, 12).quota[6 * 12 + 6];
    peggiore = Math.max(peggiore, Math.abs(q - metri / 1000));
  }
  assert.ok(peggiore < 0.35, "la scala del Cloud Top Height non si decodifica: errore "
    + peggiore.toFixed(2) + " km");
  // Un colore di bordo, lontano dalla scala, non diventa una quota: si prende
  // dai vicini.
  const bordo = m.quoteDaCth(immagine(12, 12, (x) => x === 6 ? [120, 200, 240, 255] : [254, 102, 0, 255]), 12, 12);
  assert.ok(Math.abs(bordo.quota[6 * 12 + 6] - 11.76) < 0.3,
    "un pixel mescolato dal server diventa una quota inventata: " + bordo.quota[6 * 12 + 6]);

  // Il caso della segnalazione: all'alba il suolo spagnolo e' freddo e
  // l'infrarosso lo stima alto, ma la maschera dice sereno. Non deve esserci
  // nube. Accanto, uno strato basso e caldo che l'infrarosso non vede ma la
  // maschera si': deve esserci, alla quota misurata.
  const w = 40, h = 20, quanti = w * h;
  const campo = {
    larghezza: w, altezza: h,
    quota: new Float32Array(quanti).fill(0),
    copertura: new Float32Array(quanti).fill(0)
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < 20; x++) {
    campo.quota[y * w + x] = 7; campo.copertura[y * w + x] = 1;  // suolo freddo scambiato per nube
  }
  const maschera = new Float32Array(quanti);
  for (let y = 0; y < h; y++) for (let x = 20; x < w; x++) maschera[y * w + x] = 1;
  const cth = { quota: new Float32Array(quanti), valida: new Float32Array(quanti) };
  for (let y = 0; y < h; y++) for (let x = 20; x < w; x++) { cth.quota[y * w + x] = 1.2; cth.valida[y * w + x] = 1; }
  m.campoMisurato(campo, maschera, cth, { larghezza: w, altezza: h });
  assert.ok(campo.copertura[10 * w + 5] < 0.01,
    "il suolo freddo che la maschera dice sereno torna a essere nube: copertura "
    + campo.copertura[10 * w + 5].toFixed(2));
  assert.ok(campo.copertura[10 * w + 32] > 0.3,
    "lo strato basso visto dalla maschera ma non dall'infrarosso sparisce");
  assert.ok(Math.abs(campo.quota[10 * w + 32] - 1.2) < 0.3,
    "la quota non e' piu' quella misurata dal Cloud Top Height: " + campo.quota[10 * w + 32]);

  // I GENERI. Ogni nube sta alla quota del suo genere: la base non e' la
  // cima meno uno spessore -- che attaccava ogni nube a terra come una
  // montagna -- ma la quota a cui quel genere si forma.
  const scena = (cimaKm, quotaIr, conCella, irregolare) => {
    const c = { larghezza: w, altezza: h, quota: new Float32Array(quanti),
      copertura: new Float32Array(quanti).fill(1) };
    for (let k = 0; k < quanti; k++) {
      // Celle di un pixel: piu' celle nella finestra della tessitura, come i cumuli
      // veri rispetto ai dieci chilometri della finestra.
      const cx = k % w, cy = Math.floor(k / w);
      const caso = Math.abs(Math.sin(cx * 12.9898 + cy * 78.233) * 43758.5453) % 1;
      c.quota[k] = quotaIr + (irregolare ? caso * 3 - 1.5 : 0);
    }
    const tutto = new Float32Array(quanti).fill(1);
    m.campoMisurato(c, tutto, { quota: new Float32Array(quanti).fill(cimaKm), valida: tutto },
      { larghezza: w, altezza: h });
    if (conCella) c.celle = new Float32Array(quanti).fill(1);
    m.classificaNubi(c, null, null, null);
    const centro = 10 * w + 20, kc = 5 * c.classeMisura.larghezza + 10;
    // Solo una cima convettiva osservata puo' collegare l'incudine alla base.
    const base = c.chi[centro] > 0.5 ? Math.min(1, c.base[centro]) : c.base[centro];
    return { base, genere: m.GENERI[c.classe[kc]], cima: c.quota[centro],
      incudine: c.incudine[centro], pioggiaNube: c.pioggiaNube[centro],
      granuli: c.granuli[centro], veloLiscio: c.veloLiscio[centro] };
  };
  const cirro = scena(11, 3, false, false);
  assert.equal(cirro.genere, "Cirro", "un velo alto e caldo all'infrarosso non e' riconosciuto come cirro: " + cirro.genere);
  assert.ok(cirro.base > 8.5, "il cirro non sta piu' in quota: base " + cirro.base.toFixed(1) + " km");
  const cirrostrato = scena(11, 11, false, false);
  assert.ok(cirrostrato.veloLiscio > cirro.veloLiscio + 0.08,
    "il cirrostrato spesso non diventa un velo piu' disteso del cirro");
  const altocumulo = scena(4.6, 4.6, false, true);
  const altostrato = scena(4.6, 4.6, false, false);
  assert.ok(altocumulo.granuli > altostrato.granuli + 0.08,
    "gli elementi dell'altocumulo non si distinguono dal banco dell'altostrato");
  assert.ok(altostrato.veloLiscio > altocumulo.veloLiscio + 0.08,
    "l'altostrato non conserva una cima piu' distesa dell'altocumulo");
  assert.ok(altocumulo.granuli > 0.7 && altostrato.veloLiscio > 0.7,
    "a parita' di quota i segnali di forma restano troppo deboli per distinguere Ac e As nel volume");
  const rdtSenzaPicco = scena(12, 12, true, false);
  assert.notEqual(rdtSenzaPicco.genere, "Cumulonembo",
    "una cella RDT con cima uniforme inventa una torre: " + rdtSenzaPicco.genere);
  assert.ok(rdtSenzaPicco.base > 5,
    "la nube alta senza nucleo convettivo scende artificiosamente al suolo");
  {
    // Una cima fredda gia' spianata nell'IR puo' ancora avere una colonna
    // osservata con DUE indizi indipendenti: cella RDT e rovescio radar.
    const W = 512, H = 200, N = W * H, cx = 256, cy = 100;
    const nucleo = (conRdt) => {
      const c = { larghezza: W, altezza: H, quota: new Float32Array(N).fill(12),
        quotaIr: new Float32Array(N).fill(12), copertura: new Float32Array(N).fill(1),
        cancello: new Float32Array(N).fill(1), opacita: new Float32Array(N).fill(1),
        celle: new Float32Array(N), ecoRadar: new Float32Array(N) };
      for (let y = cy - 4; y <= cy + 4; y++) for (let x = cx - 4; x <= cx + 4; x++) {
        if (Math.hypot(x - cx, y - cy) > 4) continue;
        const k = y * W + x;
        c.celle[k] = conRdt ? 1 : 0;
        c.ecoRadar[k] = 54;
      }
      m.classificaGriglia(c, null, new Float32Array(N), null);
      return c.chi[cy * W + cx];
    };
    assert.ok(nucleo(true) > 0.2,
      "una cella RDT con rovescio radar confermato resta un blocco alto senza torre");
    assert.ok(nucleo(false) < 0.01,
      "un rovescio radar isolato inventa un cumulonembo sotto un fronte liscio");
  }
  // CONVEZIONE PROFONDA senza RDT: un sistema alto, freddo, opaco e con la
  // cima ribollente e un rilievo termico coerente ha un nucleo profondo e
  // l'incudine attorno. Un fronte altrettanto alto ma liscio no.
  {
    const W = 1024, H = 400, N = W * H;  // circa 4 km per pixel, come il campo vero
    const sistema = (ribollente) => {
      const c = { larghezza: W, altezza: H, quota: new Float32Array(N), copertura: new Float32Array(N).fill(1) };
      for (let k = 0; k < N; k++) {
        const x = k % W, y = Math.floor(k / W);
        const caso = Math.abs(Math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1;
        const r = Math.hypot(x - 512, y - 200);
        c.quota[k] = 11.5 + (ribollente ? caso * 3 - 1.5
          + (r < 5 ? 3.5 * Math.exp(-r * r / 16) : 0) : 0);
      }
      const tutto = new Float32Array(N).fill(1);
      m.campoMisurato(c, tutto, { quota: new Float32Array(N).fill(12), valida: tutto }, { larghezza: W, altezza: H });
      m.classificaNubi(c, null, null, null);
      const conta = new Array(11).fill(0);
      for (const v of c.classe) conta[v]++;
      return { conta, torreCentro: c.chi[200 * W + 512], torreLontano: c.chi[200 * W + 40],
        incudineVicino: c.incudine[200 * W + 520] };
    };
    const temporale = sistema(true), fronte = sistema(false);
    const cb = m.GENERI.indexOf("Cumulonembo"), incudine = m.GENERI.indexOf("Incudine");
    assert.ok(temporale.torreCentro > 0.5,
      "la cima piu' fredda di un sistema convettivo non diventa un nucleo: " + temporale.torreCentro.toFixed(2));
    assert.ok(temporale.conta[incudine] > temporale.conta[cb],
      "attorno alla torre non c'e' l'incudine: il temporale torna un altopiano pieno");
    assert.ok(temporale.incudineVicino > fronte.incudineVicino + 0.1,
      "il tetto dell'incudine non segue il sistema convettivo osservato");
    assert.equal(fronte.conta[cb] + fronte.conta[incudine], 0,
      "un fronte alto e liscio diventa un cumulonembo: le torri in fila lungo i fronti");
  }
  {
    // Un banco medio compatto con eco radar esteso sfuma la base; un'eco
    // assente non deve generare pioggia dal solo colore della nube.
    const W = 320, H = 120, N = W * H;
    const banco = (conPioggia) => {
      const c = { larghezza: W, altezza: H, quota: new Float32Array(N).fill(5.5),
        copertura: new Float32Array(N).fill(1) };
      const tutto = new Float32Array(N).fill(1);
      m.campoMisurato(c, tutto, { quota: new Float32Array(N).fill(5.5), valida: tutto },
        { larghezza: W, altezza: H });
      if (conPioggia) c.ecoRadar = new Float32Array(Math.round(W / 2) * Math.round(H / 2)).fill(32);
      m.classificaNubi(c, null, null, null);
      return c.pioggiaNube[60 * W + 160];
    };
    assert.equal(banco(false), 0, "il banco senza radar inventa precipitazioni");
    assert.ok(banco(true) > 0.08, "l'eco radar estesa non sfuma la base del nembostrato");
  }
  const strato = scena(1.4, 1.2, false, false);
  assert.equal(strato.genere, "Stratocumulo", "uno strato basso liscio di notte non e' uno stratocumulo: " + strato.genere);
  assert.ok(strato.base < 0.8, "la nube bassa liscia non sta bassa: base " + strato.base.toFixed(1) + " km");
  assert.ok(strato.granuli > 0.1, "lo stratocumulo basso perde i suoi lobi arrotondati");
  {
    // A parita' di quota, il prodotto notturno Fog / Low Clouds distingue
    // un velo di Stratus da un banco di elementi arrotondati.
    const st = { larghezza: w, altezza: h,
      quota: new Float32Array(quanti).fill(1.4),
      quotaIr: new Float32Array(quanti).fill(1.4),
      copertura: new Float32Array(quanti).fill(1),
      cancello: new Float32Array(quanti).fill(1),
      opacita: new Float32Array(quanti).fill(1),
      basseValide: new Float32Array(quanti).fill(1),
      basseAcqua: new Float32Array(quanti).fill(1) };
    m.classificaGriglia(st, null, new Float32Array(quanti), null);
    assert.equal(m.GENERI[st.classe[10 * w + 20]], "Strato",
      "il banco basso e uniforme confermato dal prodotto notturno non e' Stratus");
    assert.ok(st.veloLiscio[10 * w + 20] > strato.veloLiscio + 0.15,
      "lo Stratus non ha la base/cima piu' uniforme dello Stratocumulus");
  }
  const cumulo = scena(1.8, 1.8, false, true);
  assert.equal(cumulo.genere, "Cumulo", "una nube bassa a cima irregolare non e' un cumulo: " + cumulo.genere);
  assert.ok(Math.abs(cumulo.base - 1.0) < 0.25, "il cumulo non ha la base al livello di condensazione: "
    + cumulo.base.toFixed(2) + " km");
  assert.ok(cumulo.granuli < strato.granuli, "il cumulo isolato diventa un banco di stratocumuli");
  assert.ok(strato.granuli > 0.7 && cumulo.granuli < 0.35,
    "il banco basso e il cumulo isolato tornano a ricevere la stessa scultura");

  // Sabbia e nube in GeoColour: la sabbia del Sahara e' luminosa ma arancione.
  const pelle = { data: new Uint8Array([162, 137, 111, 255, 200, 200, 204, 255]) };
  const dettaglio = m.dettaglioDaGeoColour(pelle, { larghezza: 2, altezza: 1 }, { larghezza: 2, altezza: 1 });
  assert.ok(dettaglio[0] < 0.05, "la sabbia del Sahara conta come nube: " + dettaglio[0].toFixed(2));
  assert.ok(dettaglio[1] > 0.8, "una nube bianca non conta piu' come nube");
}
{
  // Sotto le nubi il suolo non e' un buco: si riempie dai dintorni sereni e
  // va in ombra. E lo shader prende la base dal campo e la forma dal genere.
  const vm = require("node:vm");
  const contesto = { Math, Float32Array, Uint8Array };
  vm.createContext(contesto);
  vm.runInContext(implementazione("limita") + implementazione("lisciaFra")
    + implementazione("riempiDaiDintorni"), contesto);
  const w = 16, h = 8, rgb = new Float32Array(w * h * 3), peso = new Float32Array(w * h);
  for (let k = 0; k < w * h; k++) {
    const sereno = (k % w) < 8;
    rgb.set(sereno ? [200, 40, 20] : [240, 240, 240], k * 3);  // a destra la nube
    peso[k] = sereno ? 1 : 0;
  }
  const pieno = contesto.riempiDaiDintorni(rgb, peso, w, h);
  const k = 4 * w + 13;
  assert.ok(pieno[k * 3] > 150 && pieno[k * 3 + 1] < 90,
    "sotto la nube il suolo non viene dai dintorni: " + Array.from(pieno.slice(k * 3, k * 3 + 3)).map(Math.round));
  const frammento = html.slice(html.indexOf("var FRAMMENTO = ["), html.indexOf("var STESURA = ["));
  // La texture delle nubi 3D: R cima, G densita', B base, A CAPE.
  assert.match(frammento, /float cimaKm = campo\.r \* uScalaKm;/, "la cima non viene piu' dal canale R");
  assert.match(frammento, /float baseKm = campo\.b \* uScalaKm;/, "la base non viene piu' dal canale B");
  assert.match(frammento, /if \(campo\.g < 0\.004\) return 0\.0;/, "la maschera satellitare non e' piu' esatta");
  assert.match(frammento, /conv = campo\.a \* smoothstep\(0\.3, 0\.8, campo\.g\);/, "il CAPE non entra piu' nello shader");
  // Le precedenti asserzioni su collo, copQuota e deformazione delle
  // coordinate imponevano proprio le formule responsabili dei crateri.
  // Il test del volume verifica ora limiti fisici e immagini del vero shader.

}
{
  // Il realismo: luce cercata verso il sole, scultura di Worley letta al
  // dettaglio giusto, rumore periodico senza giunture.
  const frammento = html.slice(html.indexOf("var FRAMMENTO = ["), html.indexOf("var STESURA = ["));
  assert.match(frammento, /float spessoreVersoSole\(vec3 p, float lodCampo, float lodRumore\)/,
    "manca la marcia della luce verso il sole: le nubi tornano illuminate uguali dappertutto");
  assert.match(frammento, /spessoreVersoSole\(p, lodCampo, lodRumore\)/,
    "la marcia della luce non viene piu' usata");
  assert.match(frammento, /float passa = diretta \+ 0\.38 \* uLampoPesoCopertura \* \(diffusa - diretta\);/,
    "la luce del fulmine attraversa il nucleo fitto senza attenuazione coerente");
  assert.match(frammento, /0\.022 \* migliore \* migliore/,
    "il lampo lontano illumina l'intera nube come una luce uniforme");
  assert.match(frammento, /float lodRumore = max\(-6\.0, log2\(/,
    "il rumore non si legge piu' al livello del pixel: da vicino le bolle si spianano");
  assert.match(frammento, /uFoschia/, "manca la foschia con la distanza");
  const vm = require("node:vm");
  const contesto = { Math, Float32Array };
  vm.createContext(contesto);
  vm.runInContext(implementazione("hash3") + implementazione("worley3D"), contesto);
  const lato = 24, w = contesto.worley3D(lato, 4, 1);
  let min = 1, max = 0, bordo = 0, dentro = 0, n = 0;
  for (let z = 0; z < lato; z++) for (let y = 0; y < lato; y++) {
    const riga = (z * lato + y) * lato;
    bordo += Math.abs(w[riga] - w[riga + lato - 1]);
    dentro += Math.abs(w[riga + 10] - w[riga + 11]);
    n++;
    for (let x = 0; x < lato; x++) { min = Math.min(min, w[riga + x]); max = Math.max(max, w[riga + x]); }
  }
  assert.ok(min >= 0 && max <= 1 && max - min > 0.6, "il rumore di Worley non ha piu' bolle: " + min + ".." + max);
  assert.ok(bordo / n < 2.5 * dentro / n, "il rumore di Worley ha una giuntura al bordo: si vedrebbe una riga nelle nubi");
}
{
  // Da vicino: niente grana sui fianchi, niente pareti a tende.
  const frammento = html.slice(html.indexOf("var FRAMMENTO = ["), html.indexOf("var STESURA = ["));
  assert.match(frammento, /float passoMin = min\(passoZ \/ max\(abs\(direzione\.z\), 1e-3\), uTexelCampo \* 0\.9\);/,
    "il passo non segue piu' la quota: uno strato sottile viene scavalcato");

}
{
  // Il lampo nelle nubi: colpi di ritorno, canale esteso, ombra della nube.
  const vm = require("node:vm");
  const costanti = ["COLPI_MIN", "COLPO_INTERVALLO_MS", "COLPO_DECADIMENTO_MS",
    "COLPO_CORRENTE_MS", "LAMPO_DURATA_MS"].map((n) => html.match(new RegExp("var " + n + " = [^;]+;"))[0]);
  const contesto = { Math };
  vm.createContext(contesto);
  vm.runInContext(costanti.join("\n") + implementazione("semeDellaScarica")
    + implementazione("colpiDelLampo") + implementazione("luceDelLampo"), contesto);
  const seme = (n) => contesto.semeDellaScarica(37.5, 14.2, 1727100000000, n);
  const colpi = contesto.colpiDelLampo(seme);
  assert.ok(colpi.length >= 2 && colpi.length <= 4, "un lampo a terra non ha piu' 2-4 colpi di ritorno: " + colpi.length);
  assert.deepEqual(contesto.colpiDelLampo(seme), colpi, "la stessa scarica non da' piu' gli stessi colpi");
  // Fra un colpo e l'altro la luce cala e poi risale: il tremolio.
  const t2 = colpi[1].t;
  const prima = contesto.luceDelLampo(t2 - 1, colpi), dopo = contesto.luceDelLampo(t2 + 1, colpi);
  const picco = contesto.luceDelLampo(1, colpi);
  assert.ok(prima < picco * 0.5 && dopo > prima * 1.4,
    "il secondo colpo di ritorno non riaccende il lampo: " + [picco, prima, dopo].map((v) => v.toFixed(2)));
  assert.equal(contesto.luceDelLampo(10000, colpi), 0, "il lampo non si spegne");
  const frammento = html.slice(html.indexOf("var FRAMMENTO = ["), html.indexOf("var STESURA = ["));
  assert.match(frammento, /float zc = clamp\(altKm, 0\.0, L\.z\);/,
    "la sorgente del lampo e' tornata un punto: deve essere il canale verticale");
  assert.match(frammento, /float s = clamp\(dot\(dKm, dr\), 0\.0, lung\);/,
    "mancano i rami orizzontali del canale dentro la nube");
  assert.match(frammento, /tauL \+= densita\(mix\(p, qS, 0\.2 \+ 0\.3 \* float\(k\)\), lodCampo \+ 1\.0, lodRumore \+ 1\.0, true, h2, c2, f2\);/,
    "la luce del lampo non attraversa piu' la nube vera: il nucleo fitto non fa ombra");
  assert.match(frammento, /IL CANALE SOTTO LA NUBE/, "manca il canale visibile fra la base e il suolo");
}
console.log("nubi in volume: maschera CLM, quota CTH, opacita', generi, sabbia, suolo sotto le nubi, luce e scultura, fianchi, lampi");

// Physical cloud-volume regression and optional software-GPU visual checks.
{
// Default: numeric tests of the scalar GLSL surface function, no dependencies.
// node scripts/tests/test_map_3d.js --gpu [--shader /path/to/before.glsl]
// Additional real shader renders and density sections via EGL/Mesa, NumPy, PIL.
// --gpu is mandatory during cloud shader reviews; images go to a temp folder.
const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), vm = require("node:vm");
const os = require("node:os"), {execFileSync} = require("node:child_process");
const root = path.resolve(__dirname, "../..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
function shader(name) {
  const start = html.indexOf("var " + name + " = [");
  assert.ok(start >= 0);
  const end = html.indexOf('].join("\\n");', start) + 13;
  return vm.runInNewContext(html.slice(start, end) + "\n" + name);
}
const fragment = shader("FRAMMENTO");
// IL PROMPT DELLE NUBI 3D: raggio confinato fra base e cima, Perlin per i
// vuoti, Worley sottratto con un morso che il CAPE varia, Beer-Lambert,
// powder 1 - e^(-densita' x 2), Henyey-Greenstein.
assert.match(fragment, /if \(altKm < baseKm \|\| altKm > cimaKm \+ sopra\) return 0\.0;/,
  "il raggio non e' piu' confinato fra base e cima");
// Il metodo dei giochi: forma Perlin-Worley x profilo del tipo di nube,
// tagliata dalla copertura del satellite, erosa dal dettaglio.
assert.match(fragment, /float forma = clamp\(rimappa\(f\.r, fbm \* uContrasto, 1\.0, 0\.0, 1\.0\), 0\.0, 1\.0\);/, "manca la forma Perlin-Worley");
assert.match(fragment, /float profilo = mix\(strato, cumulo, cumuliforme\);/, "manca il profilo verticale del tipo di nube");
assert.match(fragment, /float d = clamp\(rimappa\(base, 1\.0 - cop, 1\.0, 0\.0, 1\.0\), 0\.0, 1\.0\);/, "la copertura del satellite non taglia piu' la forma");
assert.match(html, /var GENERA_FORMA = \[/, "manca il generatore della forma");
assert.match(html, /function creaPannelloRegolazione\(\)/, "manca il pannello ?regola=1");
assert.match(html, /if \(!REGOLA_ATTIVA\) return r;/, "i valori salvati devono valere solo con ?regola=1");
assert.match(fragment, /vec3 verso = vec3\(uSole\.xy, uSole\.z\) \/ kmPerUnita;/,
  "le ombre non seguono piu' la geometria esagerata: da vicino spariscono");
assert.match(fragment, /t = max\(tVicino, t - passoPrima\);/, "manca l'ingresso a passi corti: torna la brina");
assert.match(html, /var ESAGERAZIONE_MINIMA = 2\.4;/, "l'esagerazione torna a spianare le nubi da vicino");
// Realismo: diffusione multipla a ottave, incudine dei cumulonembi, cavita'
// fra i lobi, accumulo dei fotogrammi a mappa ferma solo sul PC.
assert.match(fragment, /multipla \+= 0\.12 \* exp\(-tauSole \* 0\.12\) \* fase2;/, "manca la seconda ottava di diffusione multipla");
assert.match(fragment, /float incudine = smoothstep\(7\.5, 10\.0, cimaKm\)/, "manca l'incudine dei cumulonembi");
assert.match(fragment, /float scarto = fract\(intreccio\(gl_FragCoord\.xy\) \+ uFotogramma \* 0\.618034\);/,
  "lo scarto del raggio non cambia piu' da un fotogramma all'altro: l'accumulo non converge");
assert.match(html, /rumore: 48, passiLuce: 5, passi: 176, qualita: 0, accumula: 0, latoForma: 64 \}/, "il telefono deve restare leggero");
assert.match(html, /qualita: 1, accumula: 12, latoForma: 128 \}/, "sul PC manca l'accumulo dei fotogrammi");
assert.match(html, /var lampiAccesi = /, "l'accumulo spalmerebbe i lampi");
assert.match(fragment, /float morso = uErosione \* mix\(0\.5, 1\.2, convettiva\)/, "il CAPE non varia piu' il morso del Worley");
// Da vicino: il passo segue la fascia della colonna (niente trama a puntini
// sui veli), il cielo e' schermato dalla nube sopra, le ombre dei primi
// passi verso il sole vedono i lobi, e le lamine non fanno curve di livello.
assert.match(fragment, /passo = min\(passo, max\(fine, \(fascia\.y - fascia\.x\) \/ salita \* 0\.25\)\);/,
  "il passo non segue piu' lo spessore della colonna");
assert.match(fragment, /float occlusione = exp\(-sopra \* uSigma \* 0\.5\);/, "manca l'occlusione del cielo");
assert.match(fragment, /i > 1 \|\| lodRumore > 1\.5, h, c, f\)/, "le ombre non vedono piu' il dettaglio dei lobi");
assert.match(fragment, /float powder = 1\.0 - exp\(-estinzione \* 2\.0\);/, "manca il powder");
assert.match(fragment, /float beer = exp\(-tauSole\);/, "manca Beer-Lambert verso il sole");
assert.match(fragment, /float henyeyGreenstein\(float coseno, float g\)/, "manca Henyey-Greenstein");
console.log("Cloud surface numeric checks: OK");
const GPU_QA = String.raw`import ctypes as c, math, sys, time, os
from ctypes.util import find_library
import numpy as np
from PIL import Image,ImageDraw
E=c.CDLL(find_library('EGL')); G=c.CDLL(find_library('GL')); P=c.c_void_p; I=c.c_int; U=c.c_uint; F=c.c_float
E.eglGetProcAddress.argtypes=[c.c_char_p]; E.eglGetProcAddress.restype=P
p=E.eglGetProcAddress(b'eglGetPlatformDisplayEXT'); dpy=c.CFUNCTYPE(P,U,P,P)(p)(0x31DD,None,None)
E.eglInitialize.argtypes=[P,P,P]; assert E.eglInitialize(dpy,None,None)
E.eglBindAPI(0x30A2); config=P();count=I()
E.eglChooseConfig.argtypes=[P,P,P,I,P];assert E.eglChooseConfig(dpy,(I*5)(0x3033,1,0x3040,8,0x3038),c.byref(config),1,c.byref(count)) and count.value
E.eglCreateContext.argtypes=[P,P,P,P];E.eglCreateContext.restype=P
ctx=E.eglCreateContext(dpy,config,None,(I*5)(0x3098,4,0x30FB,3,0x3038))
E.eglMakeCurrent.argtypes=[P,P,P,P];assert E.eglMakeCurrent(dpy,None,None,ctx)
def gl(name,args,ret=None):
 f=getattr(G,'gl'+name);f.argtypes=args;f.restype=ret;return f
CreateShader=gl('CreateShader',[U],U);ShaderSource=gl('ShaderSource',[U,I,P,P]);CompileShader=gl('CompileShader',[U]);GetShaderiv=gl('GetShaderiv',[U,U,P]);GetShaderInfoLog=gl('GetShaderInfoLog',[U,I,P,P])
def shader(src,typ):
 s=CreateShader(typ);buf=c.c_char_p(src.encode());ShaderSource(s,1,c.byref(buf),None);CompileShader(s);ok=I();GetShaderiv(s,0x8B81,c.byref(ok));log=c.create_string_buffer(16000);GetShaderInfoLog(s,len(log),None,log);assert ok.value,log.value.decode();return s
CreateProgram=gl('CreateProgram',[],U);AttachShader=gl('AttachShader',[U,U]);LinkProgram=gl('LinkProgram',[U]);UseProgram=gl('UseProgram',[U]);GetProgramiv=gl('GetProgramiv',[U,U,P]);GetProgramInfoLog=gl('GetProgramInfoLog',[U,I,P,P]);BindAttribLocation=gl('BindAttribLocation',[U,U,c.c_char_p])
GetUniformLocation=gl('GetUniformLocation',[U,c.c_char_p],I);Uniform1f=gl('Uniform1f',[I,F]);Uniform1i=gl('Uniform1i',[I,I]);Uniform2f=gl('Uniform2f',[I,F,F]);Uniform3f=gl('Uniform3f',[I,F,F,F]);Uniform4f=gl('Uniform4f',[I,F,F,F,F]);UniformMatrix4fv=gl('UniformMatrix4fv',[I,I,c.c_ubyte,P])
GenTextures=gl('GenTextures',[I,P]);BindTexture=gl('BindTexture',[U,U]);ActiveTexture=gl('ActiveTexture',[U]);TexParameteri=gl('TexParameteri',[U,U,I]);TexImage2D=gl('TexImage2D',[U,I,I,I,I,I,U,U,P]);TexImage3D=gl('TexImage3D',[U,I,I,I,I,I,I,U,U,P]);GenerateMipmap=gl('GenerateMipmap',[U])
GenFramebuffers=gl('GenFramebuffers',[I,P]);BindFramebuffer=gl('BindFramebuffer',[U,U]);FramebufferTexture2D=gl('FramebufferTexture2D',[U,U,U,U,I]);CheckFramebufferStatus=gl('CheckFramebufferStatus',[U],U)
GenVertexArrays=gl('GenVertexArrays',[I,P]);BindVertexArray=gl('BindVertexArray',[U]);GenBuffers=gl('GenBuffers',[I,P]);BindBuffer=gl('BindBuffer',[U,U]);BufferData=gl('BufferData',[U,c.c_ssize_t,P,U]);VertexAttribPointer=gl('VertexAttribPointer',[U,I,U,c.c_ubyte,I,P]);EnableVertexAttribArray=gl('EnableVertexAttribArray',[U]);Viewport=gl('Viewport',[I,I,I,I]);DrawArrays=gl('DrawArrays',[U,I,I]);ReadPixels=gl('ReadPixels',[I,I,I,I,U,U,P]);GetError=gl('GetError',[],U)
def tex(arr,unit,dim=2,mip=True):
 a=np.ascontiguousarray(arr);t=U();GenTextures(1,c.byref(t));target=0x806F if dim==3 else 0x0DE1;ActiveTexture(0x84C0+unit);BindTexture(target,t.value)
 for k,v in [(0x2801,0x2703 if mip else 0x2601),(0x2800,0x2601),(0x2802,0x2901 if dim==3 else 0x812F),(0x2803,0x2901 if dim==3 else 0x812F)]:TexParameteri(target,k,v)
 if dim==3:TexParameteri(target,0x8072,0x2901)
 typ=0x1406 if a.dtype==np.float32 else 0x1401; internal=0x8814 if a.dtype==np.float32 else 0x8058
 if dim==3:TexImage3D(target,0,internal,a.shape[2],a.shape[1],a.shape[0],0,0x1908,typ,a.ctypes.data)
 else:TexImage2D(target,0,internal,a.shape[1],a.shape[0],0,0x1908,typ,a.ctypes.data)
 if mip:GenerateMipmap(target)
 return t
W,H=320,240
out=tex(np.zeros((H,W,4),np.float32),7,mip=False);fbo=U();GenFramebuffers(1,c.byref(fbo));BindFramebuffer(0x8D40,fbo.value);FramebufferTexture2D(0x8D40,0x8CE0,0x0DE1,out.value,0);assert CheckFramebufferStatus(0x8D40)==0x8CD5
vao=U();GenVertexArrays(1,c.byref(vao));BindVertexArray(vao.value);vbo=U();GenBuffers(1,c.byref(vbo));BindBuffer(0x8892,vbo.value);vertices=np.array([-1,-1,3,-1,-1,3],np.float32);BufferData(0x8892,vertices.nbytes,vertices.ctypes.data,0x88E4);VertexAttribPointer(0,2,0x1406,0,0,None);EnableVertexAttribArray(0)
tex(np.fromfile('/tmp/cloud_perlin.raw',np.uint8).reshape(64,64,64,4),1,3);tex(np.fromfile('/tmp/cloud_worley.raw',np.uint8).reshape(32,32,32,4),2,3)
# La forma Perlin-Worley si genera sulla GPU con lo stesso shader della pagina.
FramebufferTextureLayer=gl('FramebufferTextureLayer',[U,U,U,I,I])
gprog=CreateProgram();AttachShader(gprog,shader(open('/tmp/cloud_VERTICE.glsl').read(),0x8B31));AttachShader(gprog,shader(open('/tmp/cloud_GENERA.glsl').read(),0x8B30));BindAttribLocation(gprog,0,b'aPos');LinkProgram(gprog);UseProgram(gprog)
LF=64;tforma=U();GenTextures(1,c.byref(tforma));ActiveTexture(0x84C0+3);BindTexture(0x806F,tforma.value)
for k,v in [(0x2801,0x2703),(0x2800,0x2601),(0x2802,0x2901),(0x2803,0x2901),(0x8072,0x2901)]:TexParameteri(0x806F,k,v)
TexImage3D(0x806F,0,0x8058,LF,LF,LF,0,0x1908,0x1401,None)
gfbo=U();GenFramebuffers(1,c.byref(gfbo));BindFramebuffer(0x8D40,gfbo.value)
for zz in range(LF):
 FramebufferTextureLayer(0x8D40,0x8CE0,tforma.value,0,zz);Viewport(0,0,LF,LF);Uniform1f(GetUniformLocation(gprog,b'uStrato'),(zz+.5)/LF);DrawArrays(4,0,3)
GenerateMipmap(0x806F);BindFramebuffer(0x8D40,fbo.value)
source=open(sys.argv[1]).read();prefix=sys.argv[2];mode=sys.argv[3] if len(sys.argv)>3 else 'render'
if mode=='section':
 source=source[:source.index('void main()')]+'''void main(){
 vec3 p=vec3(0.5+vNdc.x*20.0/uCircKm,0.5,(vNdc.y+1.0)*7.0*uEsagerazione/uCircKm);
 float h,c;vec2 f;float d=densita(p,0.0,0.0,false,h,c,f);
 colorePixel=vec4(d,h,c,1.0);
 }'''
program=CreateProgram();AttachShader(program,shader(open('/tmp/cloud_VERTICE.glsl').read(),0x8B31));AttachShader(program,shader(source,0x8B30));BindAttribLocation(program,0,b'aPos');LinkProgram(program);ok=I();GetProgramiv(program,0x8B82,c.byref(ok));log=c.create_string_buffer(16000);GetProgramInfoLog(program,len(log),None,log);assert ok.value,log.value.decode();UseProgram(program)
def uf(n,*v):
 loc=GetUniformLocation(program,n.encode());[None,Uniform1f,Uniform2f,Uniform3f,Uniform4f][len(v)](loc,*v)
def ui(n,v):Uniform1i(GetUniformLocation(program,n.encode()),v)
for n,v in [('uCampo',0),('uPerlin',1),('uWorley',2),('uForma',3),('uPassiLuce',4),('uPassi',200),('uQuantiLampi',0)]:ui(n,v)
for n,v in dict(uLatoForma=LF,uScalaFormaKm=24,uScalaMacroKm=96,uCopertura=.6,uContrasto=.8,uErosione=.18,uRigonfio=1.15,uCavolfiore=.8,uOmbra=1,uAmbiente=.9,uEsposizione=.55,uQualita=1).items():uf(n,v)
unit=1/40075;esag=float(os.environ.get('CLOUD_QA_EXAGGERATION','1.6'))
for n,v in dict(uScalaKm=16,uCircKm=40075,uEsagerazione=esag,uSigma=3.6,uFaseG=.6,uForzaSole=1,uZMax=14*esag*unit,uPassoKm=.3,uPixelAngolo=.00005,uTexelCampo=40*unit/128,uPerlinKm=48,uWorleyKm=6,uLatoPerlin=64,uLatoWorley=32).items():uf(n,v)
uf('uSemenza',.3,.6,.1);uf('uDominio',.5-20*unit,.5-20*unit,.5+20*unit,.5+20*unit);uf('uSole',.5,-.4,.768);uf('uCielo',.46,.58,.78);uf('uSuolo',.26,.25,.23);uf('uColoreSole',2.6,2.5,2.34);uf('uFoschia',.72,.81,.92)
x,y=np.meshgrid(np.linspace(-20,20,128),np.linspace(-20,20,128));r=np.hypot(x,y)
def smooth(a,b,v):
 z=np.clip((v-a)/(b-a),0,1);return z*z*(3-2*z)
# La texture delle nubi 3D: R cima, G densita', B base, A convezione (CAPE).
scenes=['Cb','St','Cu','Ci','Cu0'] if mode=='render' else ['Cb','St','Cu','Cu0']
canvas=Image.new('RGB',(W*3,(H+25)*((len(scenes)+2)//3)),(18,30,44));draw=ImageDraw.Draw(canvas)
for i,kind in enumerate(scenes):
 start=time.time();field=np.zeros((128,128,4),np.float32);cover=1-smooth(16,19,r)
 if kind=='Cb':top=11.5+.8*np.exp(-(r/5)**2);base=1.0;dens=.9*cover;conv=.8
 elif kind=='St':top=1.6;base=.5;dens=.55*cover;conv=0.
 elif kind in ('Cu','Cu0'):
  dens=np.zeros_like(r)
  for cx,cy,rad in [(-8,-3,4),(2,2,5),(11,-5,3),(-3,-11,3)]:dens=np.maximum(dens,.8*(1-smooth(rad*.35,rad,np.hypot(x-cx,y-cy))))
  top=4;base=1;conv=.7 if kind=='Cu' else 0.
 elif kind=='Ci':top=10;base=8.7;dens=.25*cover;conv=0.
 field[:,:,0]=top/16;field[:,:,1]=dens;field[:,:,2]=base/16;field[:,:,3]=conv
 tex(field,0)
 theta=math.radians(20 if kind=='Cb' else 55);right=np.array([1,0,0]);up=np.array([0,-math.sin(theta),math.cos(theta)]);forward=np.array([0,-math.cos(theta),-math.sin(theta)]);mat=np.eye(4,dtype=np.float32);mat[:3,0]=right*24*unit;mat[:3,1]=up*18*unit;mat[:3,2]=forward*70*unit;mat[:3,3]=[.5,.5,(6 if kind=='Cb' else 3)*esag*unit];UniformMatrix4fv(GetUniformLocation(program,b'uInversa'),1,0,np.ascontiguousarray(mat.T).ctypes.data)
 Viewport(0,0,W,H);DrawArrays(4,0,3);pixels=np.zeros((H,W,4),np.float32);ReadPixels(0,0,W,H,0x1908,0x1406,pixels.ctypes.data);assert GetError()==0
 pixels=pixels[::-1];np.save(f'{prefix}_{kind}_{mode}.npy',pixels)
 if mode=='render':rgb=pixels[:,:,:3]+(1-pixels[:,:,3:4])*np.array([.11,.24,.36])
 else:rgb=np.repeat(pixels[:,:,:1],3,axis=2)
 im=Image.fromarray(np.uint8(np.clip(rgb,0,1)*255));im.save(f'{prefix}_{kind}_{mode}.png');canvas.paste(im,((i%3)*W,(i//3)*(H+25)+25));draw.text(((i%3)*W+10,(i//3)*(H+25)+6),kind,fill='white');print(kind,mode,round(time.time()-start,2),'s',flush=True)
canvas.save(f'{prefix}_{mode}.png')
if mode=='section':
 z=(1-(np.arange(H)+.5)/H)*14;xx=((np.arange(W)+.5)/W*2-1)*20
 cb=np.load(f'{prefix}_Cb_section.npy')[:,:,0]
 # La torre convettiva profonda e' una nube PIENA dalla base alla cima.
 core=cb[np.ix_((z>2)&(z<10),abs(xx)<2)]
 assert core.mean()>.3, f'Cb core too thin: mean density {core.mean():.3f}'
 roof=((cb>.05)*z[:,None]).max(axis=0)
 assert roof[abs(xx)<2].max()>10.3, f'Cb top far below measured CTH: {roof[abs(xx)<2].max():.2f} km'
 floor=((cb>.05)*z[:,None]+(cb<=.05)*99).min(axis=0)
 assert floor[abs(xx)<2].min()<1.6, f'Cb base far above the LCL: {floor[abs(xx)<2].min():.2f} km'
 assert np.all(cb[:,abs(xx)>19.5]<.001), 'Cloud outside the observed coverage'
 st=np.load(f'{prefix}_St_section.npy')[:,:,0]
 assert np.all(st[z>1.95]<.001) and np.all(st[z<.45]<.001), 'Stratus leaves its base-top band (cupole comprese)'
 # Il CAPE scolpisce: lo stesso cumulo in aria instabile e' eroso a cavolfiore
 # ma resta una nube.
 cu=np.load(f'{prefix}_Cu_section.npy')[:,:,0];cu0=np.load(f'{prefix}_Cu0_section.npy')[:,:,0]
 assert (cu>.05).sum()>(cu0>.05).sum()*0.3, 'CAPE erodes the cumulus away'
 assert np.abs(cu-cu0).sum()>0.02*(cu0>.05).sum(), 'CAPE has no effect on cumulus sculpture'
 print(f'GPU physical checks OK: Cb core {core.mean():.3f}; Cb cap {roof[abs(xx)<2].max():.2f} km',flush=True)

`;
if (process.argv.includes("--gpu")) {
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),"meteo-cloud-gpu-"));
  fs.writeFileSync(path.join(dir,"cloud_VERTICE.glsl"),shader("VERTICE"));
  fs.writeFileSync(path.join(dir,"cloud_GENERA.glsl"),shader("GENERA_FORMA"));
  const override=process.argv.indexOf("--shader");
  fs.writeFileSync(path.join(dir,"cloud_FRAMMENTO.glsl"),override>=0 ? fs.readFileSync(process.argv[override+1]) : fragment);
  const noise={Math,Float32Array,Uint8Array};vm.createContext(noise);
  vm.runInContext(html.slice(html.indexOf("      function hash3("),html.indexOf("      // --- il volume, attraversato dai raggi"))
    + ";this.p=texturePerlin;this.w=textureWorley;",noise);
  const perlin=noise.p(64), rgba=new Uint8Array(perlin.length*4);
  for (let k=0;k<perlin.length;k++) rgba[k*4]=perlin[k];
  fs.writeFileSync(path.join(dir,"cloud_perlin.raw"),rgba);
  fs.writeFileSync(path.join(dir,"cloud_worley.raw"),noise.w(32));
  const program=path.join(dir,"render.py");fs.writeFileSync(program,GPU_QA.replaceAll("/tmp/",dir+"/"));
  const python=process.env.CLOUD_QA_PYTHON || "python3";
  for (const mode of ["render","section"]) execFileSync(python,[program,path.join(dir,"cloud_FRAMMENTO.glsl"),path.join(dir,"cloud"),mode],{stdio:"inherit",env:{...process.env,EGL_PLATFORM:"surfaceless"}});
  execFileSync(python,[program,path.join(dir,"cloud_FRAMMENTO.glsl"),path.join(dir,"mobile"),"section"],{stdio:"inherit",env:{...process.env,EGL_PLATFORM:"surfaceless",CLOUD_QA_EXAGGERATION:"3.2"}});
  console.log("GPU images and density arrays:",dir);
} else console.log("GPU render checks: run with --gpu before publishing shader changes");

}
