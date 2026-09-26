#!/usr/bin/env node
"use strict";

// Il motore d'inferenza delle nubi, scenario per scenario: dati di ingresso
// (satellite, radar, fulmini, ICON-2I) -> tipo, fiducia, base, cima,
// morfologia, oggetti convettivi. Gira il modulo vero di index.html.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.resolve(__dirname, "../..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const inizio = html.indexOf("const NubiVolumetriche = (function");
const corpo = html.slice(html.indexOf('"use strict";', inizio) + 13,
  html.indexOf("      // --- il volume, attraversato dai raggi", inizio));
const contesto = { Math, Float32Array, Float64Array, Uint32Array, Uint8Array, Int32Array, Date, DataView,
  Number, isMobile: () => false, costruisciMascheraNube: () => null, CLOUD_PRODUCTS: {} };
vm.createContext(contesto);
vm.runInContext("this.m = (function () {" + corpo + ";return { inferisciStati, trovaTorri, TIPI_NUBE,"
  + " punteggiNube, verticaleDelTipo, costruisciVuoto, unitaX, unitaY, CIRCONFERENZA,"
  + " QUOTA_SCALA_KM, VUOTO_KM_PER_GRADINO, VUOTO_QUOTA_KM };})();", contesto);
const m = contesto.m;

let ok = true;
function prova(nome, fn) {
  try { fn(); console.log("PASS " + nome); }
  catch (errore) { ok = false; console.log("FALLITO " + nome + ": " + errore.message); }
}

// Una cella (o una griglia) con i valori dati; tutto il resto assente.
function cella(valori, w = 1, h = 1) {
  const n = w * h, ingresso = { larghezza: w, altezza: h, kmPerPixel: 4 };
  for (const [nome, v] of Object.entries(valori)) {
    ingresso[nome] = typeof v === "function" ? Float32Array.from({ length: n }, (_, k) => v(k % w, Math.floor(k / w)))
      : new Float32Array(n).fill(v);
  }
  return ingresso;
}
const sigla = (stato, k = 0) => m.TIPI_NUBE[stato.tipo[k]].sigla;
const vicino = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} invece di ${b}`);

const scenari = {
  sereno: { copertura: 0, cima: 0 },
  cumuliSparsi: { copertura: 0.6, cima: 2.0, tau: 0.45, kappa: 0.75, granuli: 0.3, convezione: 0.35, lcl: 1.0, btC: 5 },
  congestus: { copertura: 0.9, cima: 5.8, tau: 0.8, kappa: 0.85, granuli: 0.1, convezione: 0.7, lcl: 1.1, btC: -18 },
  temporale: { copertura: 1, cima: 11.5, tau: 0.95, kappa: 0.6, chi: 0.85, dbz: 52, lampi: 3, convezione: 0.9,
    lcl: 1.2, btC: -62, incudineVicina: 0 },
  organizzato: { copertura: 1, cima: 11.8, tau: 0.95, kappa: 0.5, chi: 0.8, dbz: 50, lampi: 4, convezione: 0.9,
    lcl: 1.1, btC: -65, incudineVicina: 0.7 },
  stratocumulo: { copertura: 0.95, cima: 1.6, tau: 0.6, kappa: 0.45, granuli: 0.65, convezione: 0.05, cin: 150,
    lcl: 0.8, rh850: 80, btC: 6 },
  strato: { copertura: 1, cima: 0.6, tau: 0.6, kappa: 0.05, rh850: 98, nebbia: 0.6, suolo: 0, lcl: 0.3, btC: 9 },
  nembostrato: { copertura: 1, cima: 6.2, tau: 0.92, kappa: 0.05, liscio: 0.6, pioggiaDiffusa: 0.8, rgsp: 2.5,
    rcon: 0.2, rh850: 96, rh700: 95, lcl: 0.7, btC: -28, dbz: 28 },
  cirro: { copertura: 0.5, cima: 9.5, tau: 0.2, fibra: 0.75, btC: -52, hzero: 3.2 },
  altocumulo: { copertura: 0.8, cima: 4.4, tau: 0.45, kappa: 0.65, granuli: 0.6, btC: -8 }
};

prova("1. cielo sereno: nessuna nube dove il satellite vede sereno", () => {
  const s = m.inferisciStati(cella(scenari.sereno));
  assert.equal(s.tipo[0], 0);
});

prova("2. cumuli sparsi: cumulo humilis/mediocris con base all'LCL", () => {
  const s = m.inferisciStati(cella(scenari.cumuliSparsi));
  assert.match(sigla(s), /^Cu (hum|med)$/);
  vicino(s.base[0], 1.0, 0.05, "base all'LCL");
  assert.ok(s.cumulo[0] > 0.8 && s.apertura[0] > 0.35, "elementi convettivi separati");
});

prova("3. cumulus congestus: sviluppo verticale forte, non ghiacciato", () => {
  const s = m.inferisciStati(cella(scenari.congestus));
  assert.equal(sigla(s), "Cu con");
  assert.ok(s.cima[0] - s.base[0] > 4, "spessore > 4 km");
  assert.ok(s.sviluppo[0] > 0.75);
});

prova("pioggia sotto la nube solo con una prova (radar o modello)", () => {
  const senza = m.inferisciStati(cella(Object.assign({}, scenari.congestus, { dbz: NaN, rcon: NaN, rgsp: NaN, pioggiaDiffusa: NaN })));
  assert.equal(senza.pioggia[0], 0, "il tipo da solo non fa piovere");
  const radar = m.inferisciStati(cella(Object.assign({}, scenari.congestus, { dbz: 40 })));
  assert.ok(radar.pioggia[0] > 0.1, "con 40 dBZ piove: " + radar.pioggia[0]);
  const modello = m.inferisciStati(cella(Object.assign({}, scenari.congestus, { rcon: 4 })));
  assert.ok(modello.pioggia[0] > 0.1, "con 4 mm/h convettivi del modello piove: " + modello.pioggia[0]);
});

prova("4. temporale isolato: cumulonembo con base all'LCL e cima osservata", () => {
  const s = m.inferisciStati(cella(scenari.temporale));
  assert.match(sigla(s), /^Cb (cal|cap)$/);
  assert.equal(sigla(s), "Cb cap", "cima a -62 C: glaciata, capillatus");
  vicino(s.base[0], 1.2, 0.05, "base");
  vicino(s.cima[0], 11.5, 1e-6, "cima");
  assert.ok(s.fiducia[0] > 0.5, "fiducia " + s.fiducia[0]);
});

prova("5. sistema organizzato: incus dove c'e' l'incudine attorno", () => {
  const s = m.inferisciStati(cella(scenari.organizzato));
  assert.equal(sigla(s), "Cb inc");
  const anvil = m.inferisciStati(cella({ copertura: 1, cima: 11.2, tau: 0.85, incudine: 0.8, btC: -60, fibra: 0.4 }));
  assert.equal(sigla(anvil), "Inc", "manto freddo senza nucleo: incudine");
  assert.ok(anvil.base[0] > 8, "l'incudine sta in quota: base " + anvil.base[0]);
});

prova("6. stratocumulo: celle su base comune, strato basso inibito", () => {
  const s = m.inferisciStati(cella(scenari.stratocumulo));
  assert.equal(sigla(s), "Sc");
  assert.ok(s.cumulo[0] > 0.35 && s.cumulo[0] < 0.8, "semi-cumuliforme " + s.cumulo[0]);
  assert.ok(s.cellaKm[0] > 1.5 && s.cellaKm[0] < 6, "celle di qualche km: " + s.cellaKm[0]);
  assert.ok(s.cima[0] - s.base[0] < 1.2, "strato sottile");
});

prova("7. strato: volume basso, continuo, non convettivo", () => {
  const s = m.inferisciStati(cella(scenari.strato));
  assert.equal(sigla(s), "St");
  assert.ok(s.base[0] < 0.5 && s.cumulo[0] < 0.15 && s.apertura[0] < 0.1);
});

prova("8. nembostrato: spesso, piovoso, base bassa, niente cavolfiori", () => {
  const s = m.inferisciStati(cella(scenari.nembostrato));
  assert.equal(sigla(s), "Ns");
  assert.ok(s.base[0] < 1.6, "base bassa " + s.base[0]);
  assert.ok(s.cima[0] - s.base[0] > 4, "molto spesso");
  assert.ok(s.cumulo[0] < 0.2 && s.pioggia[0] > 0.7, "stratiforme e piovoso");
});

prova("9. fronte caldo: Ci -> Cs -> As -> Ns procedendo verso il fronte", () => {
  const x = [
    { copertura: 0.5, cima: 10, tau: 0.2, fibra: 0.8, btC: -55 },
    { copertura: 1, cima: 9, tau: 0.5, liscio: 0.8, btC: -48 },
    { copertura: 1, cima: 5.5, tau: 0.65, liscio: 0.6, btC: -25, rh700: 92 },
    scenari.nembostrato
  ];
  const nomi = Object.keys(Object.assign({}, ...x));
  const ingresso = { larghezza: 4, altezza: 1, kmPerPixel: 4 };
  nomi.forEach((n) => { ingresso[n] = Float32Array.from(x.map((v) => v[n] ?? NaN)); });
  const s = m.inferisciStati(ingresso);
  assert.deepEqual([0, 1, 2, 3].map((k) => sigla(s, k)), ["Ci", "Cs", "As", "Ns"]);
});

prova("10. fronte freddo: linea di cumulonembi -> torri separate con incudini nel vento", () => {
  const w = 60, h = 30;
  const linea = (x, y) => (Math.abs(x - 30) < 2 && (y % 10) > 2 && (y % 10) < 8 ? 0.8 : 0);
  const ingresso = cella({ copertura: 1, cima: (x, y) => (linea(x, y) ? 11 : 6), chi: linea,
    incudine: 0, lcl: 1.0, u250: 30, v250: 0, shu: 20, shv: 0, lampi: (x, y) => linea(x, y) * 2, dbz: 45 }, w, h);
  const torri = m.trovaTorri(ingresso, null, null, 999);
  // Una corrente ascendente ogni ~14 km: almeno una torre per ogni tratto,
  // tutte sulla linea.
  assert.ok(torri.length >= 3 && torri.length <= 6, "torri lungo la linea: " + torri.length);
  const tratti = new Set(torri.map((t) => Math.floor(t.y / 10)));
  assert.equal(tratti.size, 3, "ogni tratto della linea ha la sua torre");
  torri.forEach((t) => assert.ok(Math.abs(t.x - 30) < 2, "torre fuori dalla linea"));
  torri.forEach((t) => {
    vicino(t.dirIncX, 1, 1e-6, "incudine verso est (vento 250 hPa da ovest)");
    assert.equal(t.fonteIncudine, "vento 250 hPa (ICON-2I)");
    assert.ok(t.inclX > 0.3, "torre inclinata dallo shear verso est");
    vicino(t.baseKm, 1.0, 1e-6, "base all'LCL");
  });
});

prova("11. cirro: fibroso, in quota, niente base al suolo", () => {
  const s = m.inferisciStati(cella(scenari.cirro));
  assert.equal(sigla(s), "Ci");
  assert.ok(s.base[0] > 5.5 && s.fibra[0] > 0.7);
});

prova("12. strati misti: altocumulo a quota media", () => {
  const s = m.inferisciStati(cella(scenari.altocumulo));
  assert.equal(sigla(s), "Ac");
  assert.ok(s.base[0] > 3 && s.cellaKm[0] < 2.5);
});

prova("incudine osservata: la direzione viene dal manto freddo, non dal modello", () => {
  const w = 80, h = 40;
  const nucleo = (x, y) => (Math.hypot(x - 20, y - 20) < 3 ? 0.9 : 0);
  // Manto freddo verso nord-est, vento del modello verso ovest.
  const manto = (x, y) => (x > 22 && x < 45 && y > 8 && y < 20 ? 0.8 : 0);
  const ingresso = cella({ copertura: 1, cima: 11, chi: nucleo, incudine: manto, lcl: 1, u250: -25, v250: 0 }, w, h);
  const [t] = m.trovaTorri(ingresso, null, null, 999);
  assert.equal(t.fonteIncudine, "osservata (manto freddo)");
  assert.ok(t.dirIncX > 0.5 && t.dirIncY < 0, "verso nord-est (y verso sud): " + t.dirIncX + "," + t.dirIncY);
});

prova("coerenza temporale: la stessa torre conserva id e seme, e lo stadio segue la cima", () => {
  const w = 40, h = 40;
  const fai = (cima, lampi) => cella({ copertura: 1, cima, chi: (x, y) => (Math.hypot(x - 20, y - 20) < 4 ? 0.9 : 0),
    lcl: 1, lampi: (x, y) => (Math.hypot(x - 20, y - 20) < 4 ? lampi : 0) }, w, h);
  const prima = m.trovaTorri(fai(9.5, 0.1), null, null, 999);
  const dopo = m.trovaTorri(fai(11, 0.5), null, prima, 10);
  assert.equal(dopo[0].id, prima[0].id);
  assert.equal(dopo[0].seme, prima[0].seme);
  assert.equal(dopo[0].ciclo, 1, "in crescita");
  const fine = m.trovaTorri(fai(9.8, 0.05), null, dopo, 10);
  assert.equal(fine[0].ciclo, 3, "in dissipazione");
  const lontano = m.trovaTorri(fai(11, 0.5), null, prima, 180);
  assert.notEqual(lontano[0].id, prima[0].id, "tre ore dopo non e' piu' la stessa torre");
});

prova("stesso istante ricalcolato: le torri conservano l'identita'", () => {
  const w = 40, h = 40;
  const ingresso = cella({ copertura: 1, cima: 11, chi: (x, y) => (Math.hypot(x - 20, y - 20) < 4 ? 0.9 : 0), lcl: 1 }, w, h);
  const prima = m.trovaTorri(ingresso, null, null, 999, null);
  const ancora = m.trovaTorri(ingresso, null, null, 999, prima);
  assert.equal(ancora[0].id, prima[0].id);
  assert.equal(ancora[0].seme, prima[0].seme);
});

prova("sistema multicella: un'area convettiva estesa ha piu' torri, non un cilindro", () => {
  const w = 60, h = 60;
  const area = (x, y) => (Math.hypot(x - 30, y - 30) < 12 ? 0.6 + 0.3 * Math.cos(x * 0.8) * Math.cos(y * 0.8) : 0);
  const torri = m.trovaTorri(cella({ copertura: 1, cima: 11, chi: area, lcl: 1 }, w, h), null, null, 999, null);
  assert.ok(torri.length >= 3, "torri: " + torri.length);
  torri.forEach((t) => assert.ok(t.raggioKm <= 9, "raggio di una sola corrente: " + t.raggioKm));
});

prova("fiducia bassa: archetipi compatibili mescolati, niente salti", () => {
  // A meta' fra strato e stratocumulo.
  const s = m.inferisciStati(cella({ copertura: 1, cima: 1.2, tau: 0.6, kappa: 0.25, granuli: 0.3, rh850: 95, lcl: 0.5 }));
  assert.ok(s.fiducia[0] < 0.7, "fiducia " + s.fiducia[0]);
  assert.ok(s.cumulo[0] > 0.05 && s.cumulo[0] < 0.5, "morfologia intermedia " + s.cumulo[0]);
});

// IL VUOTO (Nubis3, sphere tracing): la griglia della distanza non deve MAI
// promettere piu' km vuoti di quanti ce ne siano davvero, altrimenti il
// raggio salterebbe dentro una nube. Confronto a forza bruta con le nubi
// del campo (colonne fra base e cima) e con una torre.
prova("vuoto: distanza sempre per difetto", () => {
  const dom = { x0: m.unitaX(8), x1: m.unitaX(22), yNord: m.unitaY(47), ySud: m.unitaY(35) };
  const w = 140, h = 150, dati = new Float32Array(w * h * 4);
  let seme = 7;
  const caso = () => { seme = (seme * 16807) % 2147483647; return seme / 2147483647; };
  const nubi = [];
  for (let k = 0; k < 40; k++) {
    const cx = Math.floor(caso() * w), cy = Math.floor(caso() * h), r = 1 + Math.floor(caso() * 4);
    const base = 0.5 + caso() * 4, cima = base + 0.3 + caso() * 7;
    for (let y = Math.max(0, cy - r); y < Math.min(h, cy + r); y++) {
      for (let x = Math.max(0, cx - r); x < Math.min(w, cx + r); x++) {
        const i = (y * w + x) * 4;
        dati[i] = cima / m.QUOTA_SCALA_KM; dati[i + 1] = 0.6; dati[i + 2] = base / m.QUOTA_SCALA_KM;
        nubi.push([(x + 0.5) / w, (y + 0.5) / h, base, cima]);
      }
    }
  }
  const torri = { quante: 1, a: new Float32Array(96), b: new Float32Array(96), c: new Float32Array(96), d: new Float32Array(96) };
  torri.a.set([dom.x0 + 0.3 * (dom.x1 - dom.x0), dom.yNord + 0.6 * (dom.ySud - dom.yNord), 6, 12]);
  torri.b.set([1, 0.1, 0, 0.5]); torri.c.set([1, 0, 40, 0.8]);
  const v = m.costruisciVuoto(dati, w, h, null, 0, 0, torri, dom, 384);
  const circ = m.CIRCONFERENZA / 1000;
  const coseno = (yu) => 1 / Math.cosh(Math.PI * (1 - 2 * yu));
  const kmTra = (u1, v1, u2, v2) => {
    const yu = dom.yNord + (0.5 * (v1 + v2)) * (dom.ySud - dom.yNord);
    const s = circ * coseno(yu);
    return Math.hypot((u1 - u2) * (dom.x1 - dom.x0) * s, (v1 - v2) * (dom.ySud - dom.yNord) * s);
  };
  const tu = 0.3, tv = 0.6;
  let prove = 0, salti = 0;
  for (let n = 0; n < 3000; n++) {
    const u = caso(), vv = caso(), alt = caso() * 18;
    const ix = Math.min(v.nx - 1, Math.floor(u * v.nx)), iy = Math.min(v.ny - 1, Math.floor(vv * v.ny));
    const iz = Math.min(v.nz - 1, Math.floor(alt / m.VUOTO_QUOTA_KM * v.nz));
    const promessi = v.dati[(iz * v.ny + iy) * v.nx + ix] * m.VUOTO_KM_PER_GRADINO;
    if (promessi <= 0) continue;
    prove++;
    // La nube piu' vicina: colonne del campo (mezzo texel di tolleranza) e
    // il cilindro della torre (raggio 6 km, da 1 a 12 km).
    const mezzoTexel = kmTra(0, 0, 1 / w, 1 / h) / 2;
    let minimo = Infinity;
    for (const [cu, cv, base, cima] of nubi) {
      const dz = alt < base ? base - alt : alt > cima ? alt - cima : 0;
      const dh = Math.max(0, kmTra(u, vv, cu, cv) - mezzoTexel);
      minimo = Math.min(minimo, Math.hypot(dh, dz));
    }
    const dzT = alt < 1 ? 1 - alt : alt > 12 ? alt - 12 : 0;
    minimo = Math.min(minimo, Math.hypot(Math.max(0, kmTra(u, vv, tu, tv) - 6), dzT));
    assert.ok(promessi <= minimo + 1e-6, "il vuoto promette " + promessi + " km ma la nube e' a " + minimo.toFixed(2) + " " + JSON.stringify([u, vv, alt, ix, iy, iz]));
    if (promessi > 20) salti++;
  }
  assert.ok(prove > 500 && salti > 100, "il vuoto non fa risparmiare niente: " + prove + " " + salti);
});

if (!ok) process.exit(1);
console.log("Inferenza nubi: scenari OK");
