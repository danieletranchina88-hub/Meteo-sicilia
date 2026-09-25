#!/usr/bin/env node
"use strict";

// Nubi 3D · satellite + ICON-2I: la parte del browser della fusione. Gira il
// modulo vero di index.html in un contesto isolato, senza WebGL, come le
// prove di test_map_3d.js. La piastrella di prova e' scritta dal writer
// Python (meteo_analysis/clouds/environment.py): se i due formati divergono,
// fallisce qui.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const zlib = require("node:zlib");

const root = path.resolve(__dirname, "../..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const inizio = html.indexOf("const NubiVolumetriche = (function");
const corpo = html.slice(html.indexOf('"use strict";', inizio) + 13,
  html.indexOf("      // --- il volume, attraversato dai raggi", inizio));
const contesto = { Math, Float32Array, Float64Array, Uint32Array, Uint8Array, Date, DataView,
  Number, isMobile: () => false, costruisciMascheraNube: () => null, CLOUD_PRODUCTS: {} };
vm.createContext(contesto);
vm.runInContext("this.m = (function () {" + corpo + ";return { brillanzaDaGrigio, cimaFisica,"
  + " mescolaOreAmbiente, leggiPiastrella, ambienteSulCampo, applicaQuotaFisica, classificaGriglia,"
  + " tessituraDelCampo, tessituraDeiGeneri, tessituraBaseTorre, DENSITA_GENERE_MAX,"
  + " QUOTA_SCALA_KM, BASE_CUMULO_KM, DOMINIO, GENERI };})();", contesto);
const m = contesto.m;

let ok = true;
function prova(nome, fn) {
  try { fn(); console.log("PASS " + nome); }
  catch (errore) { ok = false; console.log("FALLITO " + nome + ": " + errore.message); }
}
const vicino = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, msg + ": " + a + " invece di " + b);

prova("il grigio IR torna la temperatura della legenda EUMETView", () => {
  vicino(m.brillanzaDaGrigio(254) - 273.15, -73, 0.2, "conteggio 1");
  vicino(m.brillanzaDaGrigio(1) - 273.15, 30, 0.2, "conteggio 255");
  vicino(m.brillanzaDaGrigio(254 - 110 * 253 / 254) - 273.15, -32, 0.3, "conteggio 111");
});

prova("la cima IR risale il profilo ICON-2I e conta lo sfondamento", () => {
  vicino(m.cimaFisica(288 - 6.5 * 5, 288, 6.5, 0), 5, 1e-6, "cima a 5 km");
  const tTropo = 288 - 6.5 * 12;
  vicino(m.cimaFisica(tTropo - 7, 288, 6.5, 0), 13, 1e-6, "7 K sotto la tropopausa");
  vicino(m.cimaFisica(300, 288, 6.5, 1.2), 0, 1e-6, "suolo caldo: nessuna quota negativa");
});

const ore = (lista) => lista.map((h) => ({
  valid: new Date(Date.UTC(2026, 8, 24, h)).toISOString().replace(/\.000Z$/, "Z"),
  run: "2026-09-24T00:00:00Z", file: h + ".bin.gz"
}));
prova("l'ambiente si interpola all'istante satellitare con la regola del backend", () => {
  const t = Date.UTC(2026, 8, 24, 1, 20);
  const s = m.mescolaOreAmbiente(ore([0, 1, 2]), t);
  assert.equal(s.modo, "interpolato");
  vicino(s.peso, 1 / 3, 1e-9, "peso");
  assert.equal(s.prima.file, "1.bin.gz");
  assert.equal(m.mescolaOreAmbiente(ore([0, 1, 2]), Date.UTC(2026, 8, 23, 22)).modo, "vicino");
  assert.equal(m.mescolaOreAmbiente(ore([0, 1, 2]), Date.UTC(2026, 8, 23, 20)), null,
    "quattro ore prima del run l'ambiente non vale piu'");
  assert.equal(m.mescolaOreAmbiente(ore([0, 6]), Date.UTC(2026, 8, 24, 1)).modo, "vicino",
    "un buco di sei ore non si interpola attraverso");
  assert.equal(m.mescolaOreAmbiente([], Date.UTC(2026, 8, 24, 1)), null);
});

const grezzo = zlib.gunzipSync(fs.readFileSync(path.join(__dirname, "fixtures_cloud_env.bin.gz")));
const piastrella = m.leggiPiastrella(grezzo.buffer.slice(grezzo.byteOffset, grezzo.byteOffset + grezzo.byteLength));

prova("la piastrella scritta da Python si legge nel browser", () => {
  assert.ok(piastrella.nx > 2 && piastrella.ny > 2);
  assert.ok(piastrella.sud < piastrella.nord && piastrella.ovest < piastrella.est);
  const c = piastrella.campi;
  vicino(c.t2m[0], 293.15, 0.01, "T2m");
  vicino(c.lcl[0], 1000, 1, "LCL al mare");
  vicino(c.lcl[c.lcl.length - 1], 2500, 1, "LCL sulle montagne a nord (riga 0 = sud)");
  vicino(c.cape[c.cape.length - 1], 3200, 1, "CAPE a est");
  vicino(c.cape[0], 0, 1, "CAPE a ovest");
  vicino(c.lapse[0], 33 / 5.574, 0.01, "gradiente");
});

// Il campo del volume copre l'Europa, ICON-2I solo l'Italia.
const w = 540, h = 330;
const amb = m.ambienteSulCampo(Object.assign({ modo: "esatto", run: "x", distanza: 0 }, piastrella), w, h);
const punto = (lat, lon) => {
  const D = m.DOMINIO;
  const mercY = (l) => Math.log(Math.tan(Math.PI / 4 + l * Math.PI / 360));
  const fy = (mercY(D.nord) - mercY(lat)) / (mercY(D.nord) - mercY(D.sud));
  const fx = (lon - D.ovest) / (D.est - D.ovest);
  return Math.floor(fy * h) * w + Math.floor(fx * w);
};

prova("l'ambiente sta sul campo solo dentro il dominio ICON-2I", () => {
  const dentro = punto(41, 13), fuori = punto(55, 0);
  vicino(amb.disponibile[dentro], 1, 1e-6, "Italia centrale");
  assert.equal(amb.disponibile[fuori], 0, "Mare del Nord senza ICON-2I");
  vicino(amb.lcl[dentro], 1.0, 0.02, "LCL in km");
  assert.ok(amb.convezione[punto(41, 16)] > 0.5, "CAPE 3200 J/kg: convezione forte");
  assert.ok(amb.convezione[punto(41, 6)] < 0.05, "CAPE nullo a ovest");
  const bordo = punto(40, 3.3);
  assert.ok(amb.disponibile[bordo] > 0 && amb.disponibile[bordo] < 1, "il bordo sfuma");
});

prova("la texture e' R cima, G copertura, B base, A convezione", () => {
  const n = w * h;
  const campo = { larghezza: w, altezza: h, quota: new Float32Array(n).fill(8),
    copertura: new Float32Array(n).fill(1), base: new Float32Array(n).fill(2),
    densita: new Float32Array(n).fill(1), materia: new Float32Array(n).fill(1), ambiente: amb };
  const t = m.tessituraDelCampo(campo);
  const k = punto(41, 16);
  vicino(t[k * 4], 8 / m.QUOTA_SCALA_KM, 1e-6, "R");
  vicino(t[k * 4 + 2], 2 / m.QUOTA_SCALA_KM, 1e-6, "B");
  vicino(t[k * 4 + 3], amb.convezione[k], 1e-6, "A = convezione");
  const g = m.tessituraDeiGeneri(campo);
  assert.equal(g[k * 4 + 3], 255, "densita' del genere piena in uGeneri.a");
  delete campo.ambiente;
  assert.equal(m.tessituraDelCampo(campo)[k * 4 + 3], 0, "senza ICON-2I nessuna convezione");
});

prova("la cima IR diventa fisica dove c'e' ICON-2I", () => {
  const n = w * h;
  const pixel = new Uint8Array(n * 4);
  // Grigio di una cima a -40 C: con T2m 20 C e 5,9 K/km sono circa 10 km.
  const grigio = Math.round(254 - ((-40 + 73) * (110 / 41)) * 253 / 254);
  for (let k = 0; k < n; k++) pixel.set([grigio, grigio, grigio, 255], k * 4);
  const campo = { larghezza: w, altezza: h, quota: new Float32Array(n).fill(4),
    copertura: new Float32Array(n).fill(1), ambiente: amb };
  m.applicaQuotaFisica(campo, pixel);
  vicino(campo.quota[punto(41, 13)], 60 / (33 / 5.574), 0.4, "quota fisica in Italia");
  vicino(campo.quota[punto(55, 0)], 4, 1e-6, "fuori dominio resta la stima satellitare");
});

// La classificazione con e senza ambiente, su un banco basso a cumuli.
const classifica = (conAmbiente, cape) => {
  const W = 120, H = 60, N = W * H;
  const c = { larghezza: W, altezza: H, quota: new Float32Array(N), quotaIr: new Float32Array(N),
    copertura: new Float32Array(N).fill(1), cancello: new Float32Array(N).fill(1),
    opacita: new Float32Array(N).fill(1) };
  for (let k = 0; k < N; k++) {
    const caso = Math.abs(Math.sin((k % W) * 12.9898 + Math.floor(k / W) * 78.233) * 43758.5453) % 1;
    c.quota[k] = 2.4; c.quotaIr[k] = 2.4 + caso * 1.2 - 0.6;
  }
  if (conAmbiente) {
    c.lcl = new Float32Array(N).fill(1.8);
    c.suolo = new Float32Array(N).fill(0.6);
    c.convezione = new Float32Array(N).fill(cape);
    c.disponibile = new Float32Array(N).fill(1);
  }
  m.classificaGriglia(c, null, new Float32Array(N), null);
  const k = 30 * W + 60;
  return { base: c.base[k], torre: c.baseTorre[k], kappa: c.kappa[k], genere: m.GENERI[c.classe[k]] };
};

prova("le basi dei cumuli vengono dall'LCL di ICON-2I sopra l'orografia", () => {
  const senza = classifica(false, 0), con = classifica(true, 0.3);
  assert.ok(Math.abs(senza.torre - m.BASE_CUMULO_KM) < 1e-9, "senza ICON la torre parte da 1 km");
  vicino(con.torre, 1.8, 1e-6, "con ICON la torre parte dall'LCL");
  assert.ok(con.base > senza.base + 0.3, "la base del banco sale con l'LCL e con il suolo: "
    + con.base.toFixed(2) + " contro " + senza.base.toFixed(2));
  assert.ok(con.base >= 0.7, "nessuna base sotto il suolo");
});

prova("il CAPE sostiene i cumuli ma non li inventa", () => {
  const stabile = classifica(true, 0), instabile = classifica(true, 1);
  assert.ok(instabile.kappa > stabile.kappa + 0.1, "l'aria instabile non rende piu' cumuliforme il banco");
  assert.ok(instabile.kappa <= 1);
});

if (!ok) process.exit(1);
