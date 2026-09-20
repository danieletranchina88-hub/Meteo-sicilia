'use strict';
// Riprendere la scheda in background non deve mostrare un'immagine vecchia.
//
// loadSatelliteClouds e loadLightning catturano il riquadro e l'istante in
// una chiusura al momento della richiesta, e la pubblicano quando l'Image()
// arriva -- un passo asincrono. Se lo schermo si spegne mentre una di queste
// richieste e' in volo, resta in volo: il browser puo' lasciarla completare
// ore dopo. Le due funzioni si rifiutano di PARTIRNE una nuova mentre
// document.hidden e' vero, ma questo non invalidava una richiesta gia'
// avviata prima: il suo onload confrontava il token con cloudToken/
// lightningToken, che restava fermo per tutto il tempo nascosto, quindi la
// vedeva ancora valida. Riaprendo il telefono si poteva vedere pubblicata
// un'immagine di ore prima -- magari di notte -- sul riquadro di quando lo
// schermo si era spento, sopra la mappa di base per il resto della vista.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');

function estraiListener() {
  const inizio = html.indexOf('document.addEventListener("visibilitychange"');
  assert.ok(inizio >= 0, 'manca il listener di visibilitychange');
  const fine = html.indexOf('\n      });', inizio);
  assert.ok(fine >= 0, 'il listener di visibilitychange non si chiude come atteso');
  return html.slice(inizio, fine + 10);
}

let ok = true;
function prova(nome, fn) {
  try { fn(); console.log('PASS ' + nome); }
  catch (error) { ok = false; console.log('FALLITO ' + nome + ': ' + error.message); }
}

prova('riprendendo la scheda il satellite e il Lightning Imager si ricaricano davvero', () => {
  const listener = estraiListener();
  const rami = listener.split('} else {');
  assert.equal(rami.length, 2, 'il listener non ha piu\' i due rami nascosto/visibile');
  const ramoVisibile = rami[1];

  assert.match(ramoVisibile, /loadSatelliteClouds\(true\)/,
    'tornando visibile non si forza un ricarico del satellite: una richiesta ' +
    'vecchia rimasta in volo durante il background resterebbe valida');
  assert.match(ramoVisibile, /if \(showLightning\) loadLightning\(true\)/,
    'tornando visibile non si forza un ricarico del Lightning Imager');
  assert.match(ramoVisibile, /if \(showRadar\) syncRadarToSatellite\(cloudSlot\(\)\)/,
    'tornando visibile il radar non si riallinea al satellite corrente');

  // Il ramo NASCOSTO non deve toccare satellite/lightning: chiuderli o
  // ricaricarli mentre nessuno guarda sprecherebbe soltanto rete.
  const ramoNascosto = rami[0];
  assert.doesNotMatch(ramoNascosto, /loadSatelliteClouds|loadLightning|syncRadarToSatellite/,
    'il ramo nascosto tocca satellite/lightning/radar: dovrebbe limitarsi a fermare fulmini e particelle');
});

prova('il ricarico forzato ignora la cache e riparte da un token nuovo', () => {
  // force=true deve saltare il controllo "serve davvero ricaricare?": e'
  // esattamente il punto, perche' senza forzare, un riquadro/istante che
  // sembra invariato (perche' letto da uno stato ormai vecchio) non
  // ripartirebbe affatto.
  const re = new RegExp('      function loadSatelliteClouds\\(');
  const inizioCloud = html.search(re);
  assert.ok(inizioCloud >= 0, 'manca loadSatelliteClouds');
  const cloud = html.slice(inizioCloud, html.indexOf('\n      }', inizioCloud) + 8);
  assert.match(cloud, /if \(!force && !cloudNeedsReload/,
    'loadSatelliteClouds(true) non salta piu\' il controllo di necessita\'');
  assert.match(cloud, /const token = \+\+cloudToken;/,
    'loadSatelliteClouds non avanza piu\' il token: una richiesta vecchia in volo resterebbe valida');

  const reLight = new RegExp('      function loadLightning\\(');
  const inizioLight = html.search(reLight);
  assert.ok(inizioLight >= 0, 'manca loadLightning');
  const lightning = html.slice(inizioLight, html.indexOf('\n      }', inizioLight) + 8);
  assert.match(lightning, /if \(!force && !lightningNeedsReload/,
    'loadLightning(true) non salta piu\' il controllo di necessita\'');
  assert.match(lightning, /const token = \+\+lightningToken;/,
    'loadLightning non avanza piu\' il token: una richiesta vecchia in volo resterebbe valida');
});

function implementazione(nome) {
  const re = new RegExp('      function ' + nome + '\\(');
  const inizio = html.search(re);
  assert.ok(inizio >= 0, 'manca ' + nome);
  return html.slice(inizio, html.indexOf('\n      }', inizio) + 8);
}

prova('"l\'ultimo disponibile" non e\' piu\' lo stesso URL a ogni richiesta', () => {
  // In diretta lo slot stimato manca spesso (la latenza di 10-20 minuti e'
  // una media, non una garanzia), e il codice ripiega su "ultimo
  // disponibile" -- un URL che OMETTE l'istante apposta, lasciando
  // scegliere al servizio. Scorrendo il passato invece ogni istante ha il
  // suo &time=... e non e' mai lo stesso URL due volte, quindi non puo'
  // arrivare dalla cache del browser un byte vecchio. "Ultimo disponibile"
  // richiesto piu' volte con lo stesso riquadro produceva pero' l'IDENTICO
  // URL ogni volta: un invito a essere messo in cache, e "l'ultimo
  // disponibile" del momento della cache poteva restare quello per ore --
  // il motivo per cui in diretta si vedeva un'immagine vecchia mentre ogni
  // istante passato restava perfetto.
  const wms = html.match(/const CLOUD_WMS = "[^"]+";/);
  assert.ok(wms, 'manca CLOUD_WMS');
  const codice = [wms[0], implementazione('mercatorMetresX'), implementazione('mercatorMetresY'),
    implementazione('cloudUrl')].join('\n\n');
  const modulo = new Function(codice + '\nreturn {cloudUrl};')();
  const box = { west: 12, east: 13, south: 37, north: 38 };
  const size = { width: 512, height: 384 };
  const product = { layer: 'mtg_fd:rgb_geocolour' };
  const slot = { iso: '2026-09-20T12:00:00.000Z' };

  const primo = modulo.cloudUrl(box, size, product, slot, true);
  // Node non aspetta da solo: senza questo le due chiamate potrebbero
  // cadere nello stesso millisecondo e la prova non proverebbe nulla.
  const fine = Date.now() + 2;
  while (Date.now() <= fine) { /* attesa attiva breve */ }
  const secondo = modulo.cloudUrl(box, size, product, slot, true);
  assert.notEqual(primo, secondo,
    'due richieste di "ultimo disponibile" con lo stesso riquadro producono ancora lo stesso URL');

  // Il ramo con l'istante esplicito e' gia' unico da solo: non deve
  // guadagnare (ne perdere) la stessa aggiunta.
  const storico1 = modulo.cloudUrl(box, size, product, slot, false);
  const storico2 = modulo.cloudUrl(box, size, product, slot, false);
  assert.equal(storico1, storico2,
    'una richiesta con istante esplicito non dovrebbe cambiare da sola fra due chiamate identiche');
  assert.match(storico1, /&time=2026-09-20T12:00:00\.000Z/,
    'l\'istante esplicito e\' sparito dall\'URL');
});

console.log(ok ? 'ESITO: SUPERATO' : 'ESITO: DA RIVEDERE');
process.exit(ok ? 0 : 1);
