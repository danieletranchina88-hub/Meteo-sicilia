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
assert.match(html, /type:\s*"canvas"[\s\S]*canvas:\s*terrainParticleCanvas/,
  "texture geografica delle particelle 3D assente");
assert.match(html, /id:\s*"terrain-particles-layer"[\s\S]*type:\s*"raster"/,
  "layer drappeggiato delle particelle assente");
assert.match(html, /source\.setCoordinates\(/,
  "le coordinate della texture vento non seguono la vista");
assert.match(html, /map\.setSky\(/, "cielo MapLibre nativo assente");
assert.match(html, /function terrainExaggerationForZoom\(/,
  "esagerazione verticale adattiva assente");
assert.match(html, /rasterCanvas\.toBlob\(/,
  "pubblicazione raster asincrona assente");

const particlesLayer = html.indexOf('id: "terrain-particles-layer"');
const labelsLayer = html.indexOf('id: "labels-layer"');
assert.ok(particlesLayer > 0 && labelsLayer > particlesLayer,
  "le particelle devono restare sotto etichette e toponimi");

console.log("3D map regression checks: OK");
