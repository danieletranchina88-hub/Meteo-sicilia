'use strict';
// Regressioni del renderer Lightning Physics.
//
// Queste prove non misurano quanto un lampo sia spettacolare: verificano che
// ogni segno abbia una grandezza osservata o un significato grafico
// dichiarato, e che il codice non ricostruisca canali, energia o molteplicità
// che le sorgenti non forniscono.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');

function implementazione(nome) {
  const re = new RegExp('      function ' + nome + '\\(');
  const inizio = html.search(re);
  assert.ok(inizio >= 0, 'manca ' + nome);
  const fine = html.indexOf('\n      }', inizio);
  assert.ok(fine > inizio, 'fine non trovata per ' + nome);
  return html.slice(inizio, fine + 8);
}

function costante(nome) {
  const trovata = html.match(new RegExp('const ' + nome + ' = ([0-9.*+ ]+);'));
  assert.ok(trovata, 'manca la costante ' + nome);
  return 'const ' + nome + ' = ' + trovata[1].trim() + ';';
}

const codice = [
  'STRIKE_LIFE_MS', 'STRIKE_MAX', 'STRIKE_PING_MS',
  'STRIKE_ACTIVITY_TAU_MS', 'STRIKE_ACTIVITY_RADIUS_KM',
  'STRIKE_CELL_LAT', 'STRIKE_CELL_LON', 'STRIKE_ACTIVITY_MAX'
].map(costante).join('\n')
  + '\nconst clamp = (v, a, b) => Math.max(a, Math.min(b, v));\n'
  + [
    'tempoScaricaMillis', 'supportoRete', 'stileEtaScarica',
    'aggregaAttivita', 'limiteSegni'
  ].map(implementazione).join('\n');

const modulo = new Function(codice
  + '\nreturn {tempoScaricaMillis,supportoRete,stileEtaScarica,'
  + 'aggregaAttivita,limiteSegni};')();

const STRIKE_LIFE = eval(costante('STRIKE_LIFE_MS').match(/= (.+);/)[1]);
const TAU = eval(costante('STRIKE_ACTIVITY_TAU_MS').match(/= (.+);/)[1]);
const MAX_CELLE = Number(costante('STRIKE_ACTIVITY_MAX').match(/= (.+);/)[1]);

let ok = true;
function prova(nome, fn) {
  try { fn(); console.log('PASS ' + nome); }
  catch (error) {
    ok = false;
    console.log('FALLITO ' + nome + ': ' + error.message);
  }
}

prova('il renderer non inventa la geometria del canale', () => {
  const nomiVietati = [
    'generaRagno', 'generaCanale', 'preparaScarica', 'puntiFilamento',
    'strokeSfumato', 'drawBoltGlyph', 'luceScarica', 'luceNube', 'luceBrace'
  ];
  for (const nome of nomiVietati) {
    assert.doesNotMatch(html, new RegExp('function ' + nome + '\\('),
      nome + ' ricostruisce ancora una grandezza non osservata');
  }
  assert.doesNotMatch(html, /\.ragno\b|\.colpi\b|\.seme\b/,
    'una scarica conserva ancora geometria o colpi sintetici');
});

prova('il live conserva solo le misure disponibili e il tempo di arrivo UI', () => {
  const add = implementazione('addLiveStrike');
  for (const campo of ['lat: lat', 'lon: lon', 'at:', 'ricevuta:', 'stazioni: stazioni']) {
    assert.match(add, new RegExp(campo.replace(/[.*+?^$\{\}()|[\]\\]/g, '\\$&')),
      'manca il campo ' + campo);
  }
  assert.doesNotMatch(add, /energia|radiance|footprint|polarita|molteplicita/,
    'il live deduce una misura che il flusso non certifica');
  assert.match(add, /duplicata/, 'manca la deduplicazione dopo una riconnessione');
});

prova('il timestamp della rete viene convertito da nanosecondi a millisecondi', () => {
  const ns = 1789770590237524700;
  assert.equal(modulo.tempoScaricaMillis(ns), Number(ns) / 1e6);
});

prova('colore e opacita raccontano soltanto eta del dato', () => {
  const nuova = modulo.stileEtaScarica(0);
  const recente = modulo.stileEtaScarica(15000);
  const media = modulo.stileEtaScarica(60000);
  const vecchia = modulo.stileEtaScarica(STRIKE_LIFE - 1);
  assert.deepEqual(nuova.colore, [255, 255, 255], 'il dato nuovo non parte neutro');
  assert.ok(recente.colore[2] > recente.colore[0],
    'la fase recente non si distingue dalla fase vecchia');
  assert.ok(media.colore[0] >= media.colore[2],
    'dopo un minuto il colore non vira verso il caldo');
  assert.ok(vecchia.alfa < media.alfa && media.alfa <= nuova.alfa,
    'lopacita non cala con leta');
});

prova('il supporto di rete e logaritmico e non viene spacciato per probabilita', () => {
  const basso = modulo.supportoRete(1);
  const medio = modulo.supportoRete(6);
  const alto = modulo.supportoRete(24);
  assert.ok(basso < medio && medio < alto, 'piu stazioni non aumentano il supporto');
  assert.equal(alto, 1, 'la scala non raggiunge il massimo dichiarato');
  assert.doesNotMatch(implementazione('supportoRete'), /confidence/i,
    'il numero di stazioni viene presentato come probabilita certificata');
});

prova('la densita usa tempo osservato, decadimento esponenziale e kernel fisso', () => {
  const adesso = 1_000_000;
  const scariche = [
    { lat: 37.5, lon: 14.2, at: adesso, stazioni: 8 },
    { lat: 37.501, lon: 14.201, at: adesso - TAU, stazioni: 5 },
    // Ricevuta ora ma osservata troppo tempo fa: non deve pesare.
    { lat: 37.5, lon: 14.2, at: adesso - STRIKE_LIFE - 1, ricevuta: adesso, stazioni: 20 }
  ];
  const celle = modulo.aggregaAttivita(scariche, adesso);
  assert.equal(celle.length, 1, 'punti nella stessa cella non vengono aggregati');
  assert.ok(Math.abs(celle[0].peso - (1 + Math.exp(-1))) < 1e-9,
    'il decadimento non segue exp(-eta/tau)');
  assert.equal(celle[0].quante, 2, 'una rilevazione scaduta entra nella densita');

  const disegno = implementazione('disegnaAttivita');
  assert.match(disegno, /STRIKE_ACTIVITY_RADIUS_KM/,
    'lalone non usa il raggio geografico dichiarato');
  assert.doesNotMatch(disegno, /Math\.sin|battito|respir/,
    'la densita pulsa con un movimento non misurato');
});

prova('la densita ha un limite di memoria e di costo', () => {
  const adesso = 2_000_000;
  const sparse = Array.from({ length: MAX_CELLE + 25 }, (_, i) => ({
    lat: 33 + i * 0.05,
    lon: 4 + i * 0.07,
    at: adesso - i,
    stazioni: 4
  }));
  assert.equal(modulo.aggregaAttivita(sparse, adesso).length, MAX_CELLE);
});

prova('lanello iniziale e dichiarato feedback di arrivo, non footprint', () => {
  const ping = implementazione('disegnaNuovoRilevamento');
  const draw = implementazione('drawLiveStrikes');
  assert.match(ping, /STRIKE_PING_MS/, 'manca la finestra breve del feedback');
  assert.match(draw, /adessoFrame - s\.ricevuta/,
    'lanello non segue il momento di ricezione del dato');
  assert.doesNotMatch(ping, /footprint|energia|radiance/,
    'il feedback UI pretende di rappresentare una misura fisica');
  assert.match(html, /non rappresenta il canale/,
    'linterfaccia non dichiara il limite del feedback');
});

prova('la dimensione della densita e geografica, i punti restano etichette', () => {
  const km = implementazione('raggioKmInPixel');
  assert.match(km, /chilometri \/ \(111\.32 \* coseno\)/,
    'i chilometri non vengono convertiti con la latitudine');
  const punto = implementazione('disegnaPuntoScarica');
  assert.doesNotMatch(punto, /STRIKE_ACTIVITY_RADIUS_KM/,
    'il punto singolo viene presentato come area fisica');
});

prova('il live scompare nel passato e non simula una storia inesistente', () => {
  const draw = implementazione('drawLiveStrikes');
  assert.match(draw, /if \(cloudTimeSelected\) return;/,
    'le rilevazioni live restano visibili su un istante storico');
});

prova('il ridisegno rapido dura solo quanto il feedback di arrivo', () => {
  const animazione = implementazione('animateStrikes');
  assert.match(animazione, /timestamp - s\.ricevuta < STRIKE_PING_MS/,
    'lanimazione veloce non e legata ai nuovi dati');
  assert.match(animazione, /\(nuovo \? 0 : 1000\)/,
    'la mappa continua a girare ad alta frequenza senza un nuovo dato');
});

prova('AFA mantiene metadati coerenti e la legenda ufficiale', () => {
  const product = html.match(/const LIGHTNING_PRODUCT = \{[\s\S]*?\n      \};/);
  assert.ok(product, 'manca LIGHTNING_PRODUCT');
  assert.match(product[0], /metres: 4500/, 'AFA dichiara una risoluzione incoerente');
  const catalogo = html.match(/lightning: \{[\s\S]*?\n        \}/);
  assert.ok(catalogo, 'manca il prodotto LI nel selettore');
  assert.match(catalogo[0], /metres: LIGHTNING_PRODUCT\.metres/,
    'lo stesso prodotto ha due risoluzioni diverse');
  assert.match(html, /"raster-resampling": "nearest"/,
    'AFA viene interpolata come una fotografia');
  assert.doesNotMatch(html, /raster-color.*satellite-lightning/,
    'AFA viene ricolorata nel browser');
});

prova('la timeline pubblica nubi e AFA dalla cache nello stesso input', () => {
  const controlli = implementazione('buildSatelliteControls');
  assert.match(controlli, /pubblicaFotogrammaInCache\(\)/,
    'le nubi non vengono pubblicate subito dalla cache');
  assert.match(controlli, /pubblicaLightningInCache\(\)/,
    'AFA aspetta ancora la rete durante lo scorrimento');
  const prefetch = implementazione('scheduleCloudPrefetch');
  assert.match(prefetch, /prodotti\.push\(LIGHTNING_PRODUCT\)/,
    'il precaricamento ignora AFA');
});

prova('gli avvisi sono secondari e spenti per impostazione iniziale', () => {
  assert.match(html, /let strikeSoundOn = false;/, 'il tick parte acceso');
  assert.match(html, /let strikeHapticOn = false;/, 'la vibrazione parte accesa');
  assert.match(html, /id="lightning-alerts" class="sat-alerts" hidden/,
    'gli avvisi occupano ancora il pannello principale');
  assert.match(html, /avviso UI · non è un tuono/,
    'il tick non dichiara di essere una notifica');
});

prova('linterfaccia distingue rete a terra e osservazione ottica', () => {
  assert.match(html, /Scariche a terra · Live/, 'manca il nome della rete a terra');
  assert.match(html, /Attività ottica · MTG LI/, 'manca il nome del dato satellitare');
  assert.match(html, /punto rilevato · colore = età/,
    'manca la legenda compatta del live');
});

console.log(ok ? 'ESITO: SUPERATO' : 'ESITO: DA RIVEDERE');
process.exitCode = ok ? 0 : 1;
