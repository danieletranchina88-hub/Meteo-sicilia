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

console.log(ok ? 'ESITO: SUPERATO' : 'ESITO: DA RIVEDERE');
process.exit(ok ? 0 : 1);
