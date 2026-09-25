#!/usr/bin/env node
"use strict";

// Prove del ray marcher delle nubi 3D fuse (nubi_fusione.js). Senza WebGL:
// girano sul runner standard e nella corsia rapida del workflow.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");

const root = path.resolve(__dirname, "../..");
const NubiFusione = require(path.join(root, "nubi_fusione.js"));
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");

const prove = [];
function prova(nome, fn) { prove.push([nome, fn]); }

// Un PNG RGBA con un filtro diverso per riga, come puo' scriverlo chiunque.
function pngDiProva(larghezza, altezza, pixel) {
  const riga = larghezza * 4;
  const grezzo = Buffer.alloc(altezza * (riga + 1));
  const paeth = (a, b, c) => {
    const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
    return pa <= pb && pa <= pc ? a : (pb <= pc ? b : c);
  };
  for (let y = 0; y < altezza; y++) {
    const filtro = y % 5;
    grezzo[y * (riga + 1)] = filtro;
    for (let x = 0; x < riga; x++) {
      const v = pixel[y * riga + x];
      const a = x >= 4 ? pixel[y * riga + x - 4] : 0;
      const b = y > 0 ? pixel[(y - 1) * riga + x] : 0;
      const c = x >= 4 && y > 0 ? pixel[(y - 1) * riga + x - 4] : 0;
      const predetto = [0, a, b, (a + b) >> 1, paeth(a, b, c)][filtro];
      grezzo[y * (riga + 1) + 1 + x] = (v - predetto) & 255;
    }
  }
  const pezzo = (tipo, dati) => {
    const lung = Buffer.alloc(4); lung.writeUInt32BE(dati.length);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(zlib.crc32(Buffer.concat([Buffer.from(tipo), dati])) >>> 0);
    return Buffer.concat([lung, Buffer.from(tipo), dati, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(larghezza, 0); ihdr.writeUInt32BE(altezza, 4);
  ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), pezzo("IHDR", ihdr),
    pezzo("IDAT", zlib.deflateSync(grezzo)), pezzo("IEND", Buffer.alloc(0))]);
}

prova("il PNG si decodifica senza premoltiplicare l'alfa", async () => {
  const w = 6, h = 10;
  const pixel = new Uint8Array(w * h * 4);
  for (let k = 0; k < pixel.length; k++) pixel[k] = (k * 37 + (k >> 3) * 11) & 255;
  // Nube stratiforme: convezione (A) zero ma cima, densita' e base presenti.
  for (let k = 3; k < pixel.length; k += 8) pixel[k] = 0;
  const png = pngDiProva(w, h, pixel);
  const buffer = png.buffer.slice(png.byteOffset, png.byteOffset + png.byteLength);
  const fuori = await NubiFusione.decodificaPng(buffer);
  assert.equal(fuori.larghezza, w);
  assert.equal(fuori.altezza, h);
  assert.deepEqual(Array.from(fuori.dati), Array.from(pixel));
});

prova("il dominio va nelle coordinate unitarie di MapLibre", () => {
  const d = NubiFusione.dominioUnitario({ west: 3, south: 33.7, east: 22, north: 48.9 });
  assert.ok(Math.abs(d.x0 - (183 / 360)) < 1e-12);
  assert.ok(Math.abs(d.x1 - (202 / 360)) < 1e-12);
  assert.ok(d.yNord < d.ySud, "in MapLibre y cresce verso sud");
});

prova("Perlin e Worley sono in [0,1] e periodici", () => {
  const lato = 16;
  const p = NubiFusione.perlin3D(lato, 2, 2, 5);
  const w = NubiFusione.worley3D(lato, 4, 7);
  for (const campo of [p, w]) {
    let min = 1, max = 0;
    for (const v of campo) { min = Math.min(min, v); max = Math.max(max, v); }
    assert.ok(min >= 0 && max <= 1, "fuori da [0,1]");
    assert.ok(max - min > 0.5, "rumore quasi piatto");
  }
  // Periodico: l'ultima colonna e' vicina alla prima quanto due colonne
  // adiacenti qualsiasi (nessuna giuntura).
  const salto = (a, b) => {
    let s = 0;
    for (let z = 0; z < lato; z++) for (let y = 0; y < lato; y++) {
      s += Math.abs(p[(z * lato + y) * lato + a] - p[(z * lato + y) * lato + b]);
    }
    return s / (lato * lato);
  };
  assert.ok(salto(lato - 1, 0) < 2.5 * salto(3, 4) + 0.02, "il Perlin ha una giuntura");
});

prova("lo shader fa Beer-Lambert, polvere e Henyey-Greenstein", () => {
  const f = NubiFusione.FRAMMENTO;
  assert.match(f, /float henyeyGreenstein\(float coseno, float g\)/);
  assert.match(f, /\(1\.0 - g2\) \/ \(4\.0 \* PI \* pow\(max\(1e-4, 1\.0 \+ g2 - 2\.0 \* g \* coseno\), 1\.5\)\)/);
  assert.match(f, /float powder = 1\.0 - exp\(-estinzione \* 2\.0\);/);
  assert.match(f, /float beer = exp\(-tauSole\);/);
  assert.match(f, /float trasmPasso = exp\(-estinzione \* passoKm\);/);
  // La fascia base-cima e' il bounding volume.
  assert.match(f, /if \(altKm < baseKm \|\| altKm > cimaKm\) return 0\.0;/);
  // Il CAPE modula il morso del Worley.
  assert.match(f, /float morso = mix\(0\.14, 0\.9, convettiva\);/);
  assert.match(f, /rimappa\(forma, dettaglio \* morso, 1\.0, 0\.0, 1\.0\)/);
  // Il satellite e' la maschera.
  assert.match(f, /if \(campo\.g < 0\.004\) return 0\.0;/);
  assert.ok(!/texture\(uCampo/.test(f.replace(/textureLod\(uCampo/g, "")),
    "il campo si legge sempre con un livello esplicito");
});

prova("il sole e' nel verso giusto dello spazio Mercatore", () => {
  // Mezzogiorno vero a Palermo in estate: il sole sta a sud, cioe' y > 0.
  const sole = NubiFusione.posizioneSole(Date.UTC(2026, 5, 21, 11, 7), 38.1, 13.4);
  assert.ok(sole.elevazione > 1.2);
  const v = NubiFusione.vettoreSole(sole);
  assert.ok(v[1] > 0, "a mezzogiorno il sole e' a sud: +y in Mercatore");
  assert.ok(Math.abs(Math.hypot(v[0], v[1], v[2]) - 1) < 1e-9);
  const notte = NubiFusione.luceDiScena({ elevazione: -0.3, azimut: 0 });
  assert.equal(notte.forza, 0);
});

prova("la pagina carica il modulo e offre il comando", () => {
  assert.match(html, /<script src="nubi_fusione\.js"><\/script>/);
  assert.match(html, /data-toggle="volumefusion"/);
  assert.match(html, /"strikesound", "strikehaptic", "volume", "volumefusion"/,
    "il comando deve vivere nella vista satellite");
  assert.match(html, /volumefusion: showFusedClouds,/);
  assert.match(html, /} else if \(name === "volumefusion"\) \{/);
  assert.ok(html.indexOf('src="nubi_fusione.js"') < html.indexOf("new NubiFusione.Controllo()"),
    "il modulo deve essere caricato prima di essere usato");
});

(async () => {
  let ok = true;
  for (const [nome, fn] of prove) {
    try { await fn(); console.log("PASS " + nome); }
    catch (errore) { ok = false; console.log("FALLITO " + nome + ": " + errore.message); }
  }
  if (!ok) process.exit(1);
})();
