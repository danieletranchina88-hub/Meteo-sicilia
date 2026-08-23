// Prova dell'algoritmo dei centri di pressione su campi costruiti, dove la
// risposta giusta la conosco, e poi sul campo PMSL vero del run pubblicato.
const fs = require("node:fs");
const zlib = require("node:zlib");
const path = require("node:path");
const root = path.resolve(__dirname, "../..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");

function grab(name) {
  const m = html.match(new RegExp("function " + name + "\\([\\s\\S]*?\\n {6}\\}"));
  if (!m) throw new Error("manca " + name);
  return m[0];
}
const consts = ["CENTER_ANALYSIS_KM", "CENTER_SMOOTHING_KM",
                "CENTER_MAX_PROMINENCE_STEPS", "ISOBAR_INTERVAL_HPA"]
  .map((n) => html.match(new RegExp("const " + n + " = [^;]+;"))[0]).join("\n");
const src = consts + "\n" + ["smoothPressureGrid", "parabolicCenterOffset",
  "sampleBilinear", "coarsenPressureGrid", "closedContourProminence",
  "detectPressureCenters"].map(grab).join("\n");
const api = new Function("clamp", "getGrid", src +
  "\nreturn { detectPressureCenters, closedContourProminence, coarsenPressureGrid };")(
  (v, a, b) => Math.min(Math.max(v, a), b), (a) => a);

// Dominio simile a ICON-2I: 2,2 km, ma per le prove basta piu' grosso.
function makeMeta(nx, ny, dx) {
  return { nx, ny, dx, dy: dx, lo1: 3.0, la1: 48.9 };
}
function build(nx, ny, fn) {
  const g = new Float32Array(nx * ny);
  for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) g[y * nx + x] = fn(x, y);
  return g;
}
let ok = true;
function check(cond, msg) { if (!cond) { console.log("  FALLITO: " + msg); ok = false; } }

const NX = 200, NY = 160, DX = 0.1;      // ~11 km per cella
const meta = makeMeta(NX, NY, DX);

// 1) Una depressione isolata: una B, nessuna A.
console.log("1) depressione isolata");
let field = build(NX, NY, (x, y) => {
  const r2 = Math.pow(x - 100, 2) + Math.pow(y - 80, 2);
  return 1013 - 25 * Math.exp(-r2 / (2 * 30 * 30));
});
let out = api.detectPressureCenters(field, meta);
console.log("   trovati:", out.map((c) => c.kind + " " + Math.round(c.value)
  + " (prominenza " + c.prominence.toFixed(0) + " hPa, " + c.closedIsobars + " isobare)").join(", ") || "nessuno");
check(out.length === 1 && out[0].kind === "B", "attesa una sola B");
check(Math.abs(out[0].x - 100) < 6 && Math.abs(out[0].y - 80) < 6, "B fuori posto");

// 2) Una sella fra due anticicloni: due A e NESSUNA B nel colle.
// E' il caso che la versione precedente sbagliava.
console.log("2) colle fra due anticicloni");
field = build(NX, NY, (x, y) => {
  const a1 = 22 * Math.exp(-(Math.pow(x - 55, 2) + Math.pow(y - 80, 2)) / (2 * 26 * 26));
  const a2 = 22 * Math.exp(-(Math.pow(x - 145, 2) + Math.pow(y - 80, 2)) / (2 * 26 * 26));
  return 1008 + a1 + a2;
});
out = api.detectPressureCenters(field, meta);
console.log("   trovati:", out.map((c) => c.kind + " " + Math.round(c.value)).join(", ") || "nessuno");
check(out.filter((c) => c.kind === "A").length === 2, "attese due A");
check(out.filter((c) => c.kind === "B").length === 0,
  "il colle fra due anticicloni non e' una depressione");

// 3) Saccatura allungata senza centro chiuso: nessuna lettera.
console.log("3) saccatura allungata aperta");
field = build(NX, NY, (x, y) => 1013 - 14 * Math.exp(-Math.pow(x - 100, 2) / (2 * 22 * 22)));
out = api.detectPressureCenters(field, meta);
console.log("   trovati:", out.length ? out.map((c) => c.kind).join(", ") : "nessuno");
check(out.length === 0, "un asse di saccatura aperto non e' un centro");

// 4) Niente soglie assolute: una depressione a 1020 hPa dentro un promontorio
//    deve uscire come B, e un anticiclone a 1006 dentro una saccatura come A.
console.log("4) centri fuori dai valori 'tipici'");
field = build(NX, NY, (x, y) => {
  const r2 = Math.pow(x - 100, 2) + Math.pow(y - 80, 2);
  return 1032 - 12 * Math.exp(-r2 / (2 * 28 * 28));
});
out = api.detectPressureCenters(field, meta);
console.log("   depressione a " + (out[0] ? Math.round(out[0].value) : "-") + " hPa ->",
  out.map((c) => c.kind).join(",") || "nessuno");
check(out.length === 1 && out[0].kind === "B",
  "una depressione a 1020 hPa resta una depressione");
field = build(NX, NY, (x, y) => {
  const r2 = Math.pow(x - 100, 2) + Math.pow(y - 80, 2);
  return 994 + 12 * Math.exp(-r2 / (2 * 28 * 28));
});
out = api.detectPressureCenters(field, meta);
console.log("   anticiclone a " + (out[0] ? Math.round(out[0].value) : "-") + " hPa ->",
  out.map((c) => c.kind).join(",") || "nessuno");
check(out.length === 1 && out[0].kind === "A",
  "un anticiclone a 1006 hPa resta un anticiclone");

// 5) Ordinamento per prominenza: la depressione piu' profonda per prima.
console.log("5) ordinamento per prominenza");
field = build(NX, NY, (x, y) => {
  const d1 = 26 * Math.exp(-(Math.pow(x - 55, 2) + Math.pow(y - 50, 2)) / (2 * 24 * 24));
  const d2 = 8 * Math.exp(-(Math.pow(x - 150, 2) + Math.pow(y - 110, 2)) / (2 * 24 * 24));
  return 1015 - d1 - d2;
});
out = api.detectPressureCenters(field, meta);
console.log("   " + out.map((c) => c.kind + " " + Math.round(c.value)
  + " prom " + c.prominence.toFixed(0)).join(" | "));
check(out.length === 2 && out[0].prominence > out[1].prominence,
  "la struttura piu' marcata deve venire prima");

// 6) Rumore di piccola scala: non deve fabbricare centri.
console.log("6) rumore sovrapposto a un campo liscio");
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
field = build(NX, NY, (x, y) => 1013 + 0.004 * (x - 100) + 1.6 * rnd());
out = api.detectPressureCenters(field, meta);
console.log("   centri da puro rumore:", out.length);
check(out.length === 0, "il rumore non deve produrre centri");

// 7) Campo PMSL vero del run pubblicato.
console.log("7) campo PMSL reale");
const stepFile = process.env.PRESSURE_TEST_STEP || "data_weather/step_0.json.gz";
if (fs.existsSync(stepFile)) {
  const data = JSON.parse(zlib.gunzipSync(fs.readFileSync(stepFile)));
  const press = Float32Array.from(data.press.map((v) => (v === null ? NaN : v)));
  const realMeta = data.meta;
  const t0 = process.hrtime.bigint();
  out = api.detectPressureCenters(press, realMeta);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  console.log("   griglia " + realMeta.nx + "x" + realMeta.ny + " in " + ms.toFixed(0) + " ms");
  out.forEach((c) => {
    const lon = realMeta.lo1 + c.x * realMeta.dx;
    const lat = realMeta.la1 - c.y * realMeta.dy;
    console.log("   " + c.kind + " " + c.value.toFixed(1) + " hPa a "
      + lon.toFixed(1) + "E " + lat.toFixed(1) + "N · "
      + c.closedIsobars + " isobare chiuse");
  });
  check(ms < 900, "troppo lento sul campo reale (" + ms.toFixed(0) + " ms)");
  check(out.every((c) => c.closedIsobars >= 1),
    "pubblicato un centro senza isobare chiuse");
} else {
  console.log("   (nessun campo PMSL locale: prova saltata)");
}

console.log(ok ? "\nESITO: SUPERATO" : "\nESITO: DA RIVEDERE");
process.exit(ok ? 0 : 1);
