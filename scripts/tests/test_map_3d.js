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
  /function projectParticle\([\s\S]*?\n\s*\}/
);
assert.ok(projectParticle, "proiezione particelle assente");
assert.match(projectParticle[0], /map\.project\(\[longitude, latitude\]\)/,
  "le particelle non usano la proiezione pubblica compatibile col terreno");

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
// Sopra un campo chiaro il bianco puro sparirebbe senza un alone.
assert.match(html, /Alone scuro sotto il tratto bianco/,
  "alone di leggibilita' delle particelle assente");

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

console.log("3D map regression checks: OK");
