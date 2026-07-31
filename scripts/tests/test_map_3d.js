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

console.log("3D map regression checks: OK");
