'use strict';
// La geometria e i tempi del lampo.
//
// Il disegno di un fulmine e' l'unico pezzo del sito dove "sembra giusto" non
// basta: la prima versione era una spezzata a cinque segmenti e su schermo
// usciva una riga quasi dritta. Il difetto non si vede leggendo il codice --
// il generatore c'era, e sembrava sensato -- si vede misurando la curva che
// produce. Queste prove misurano quella curva.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');

function implementazione(nome) {
  const re = new RegExp('      function ' + nome + '\\(');
  const inizio = html.search(re);
  assert.ok(inizio >= 0, 'manca ' + nome);
  return html.slice(inizio, html.indexOf('\n      }', inizio) + 8);
}

// Le costanti si leggono dal sito, non si ricopiano: se domani cambia la
// durata del leader, queste prove devono misurare quella nuova.
function costante(nome) {
  const trovata = html.match(new RegExp('const ' + nome + ' = ([0-9.]+);'));
  assert.ok(trovata, 'manca la costante ' + nome);
  return 'const ' + nome + ' = ' + trovata[1] + ';';
}

const codice = [costante('STRIKE_LEADER_MS'), costante('STRIKE_STROKE_MS')].join('\n')
  + '\n\n'
  + ['semeCasuale', 'generaCanale', 'generaRami', 'preparaScarica', 'luceScarica']
      .map(implementazione).join('\n\n');
const modulo = new Function(codice
  + '\nreturn {semeCasuale, generaCanale, generaRami, preparaScarica, luceScarica};')();

let ok = true;
function prova(nome, fn) {
  try { fn(); console.log('PASS ' + nome); }
  catch (error) { ok = false; console.log('FALLITO ' + nome + ': ' + error.message); }
}

function scarica(seme) {
  const s = { seme: seme };
  modulo.preparaScarica(s);
  return s;
}

prova('la stessa scarica da sempre lo stesso fulmine', () => {
  // Se la forma cambiasse fra un fotogramma e l'altro, il lampo tremerebbe
  // invece di brillare: e' il motivo per cui la geometria nasce da un seme
  // fisso e si calcola una volta sola.
  const a = scarica(0.37), b = scarica(0.37), c = scarica(0.38);
  assert.deepEqual(a.canale, b.canale, 'due scariche uguali danno canali diversi');
  assert.deepEqual(a.colpi, b.colpi, 'due scariche uguali danno colpi diversi');
  assert.notDeepEqual(a.canale, c.canale, 'semi diversi danno lo stesso canale');
});

prova('il canale scende sempre e arriva a terra', () => {
  for (let i = 0; i < 60; i += 1) {
    const s = scarica(i / 60);
    let precedente = -1;
    for (let k = 0; k < s.canale.length; k += 1) {
      assert.ok(s.canale[k].y >= precedente,
        'il canale risale: al punto ' + k + ' la quota torna indietro');
      precedente = s.canale[k].y;
    }
    assert.equal(s.canale[0].y, 0, 'il canale non parte dalla base della nube');
    assert.equal(s.canale[s.canale.length - 1].y, 1, 'il canale non tocca terra');
  }
});

prova('il canale e\' davvero spezzato, non una riga', () => {
  // La prova che avrebbe smascherato la prima versione. Due misure:
  // lo scarto laterale, in unita' di semilarghezza del disegno, e la
  // tortuosita', cioe' quanto il percorso e' piu' lungo della corda.
  let scartoMinimo = Infinity, tortuositaMinima = Infinity;
  for (let i = 0; i < 60; i += 1) {
    const s = scarica(i / 60);
    let scarto = 0, percorso = 0;
    for (let k = 0; k < s.canale.length; k += 1) {
      scarto = Math.max(scarto, Math.abs(s.canale[k].x));
      if (k) {
        percorso += Math.hypot(s.canale[k].x - s.canale[k - 1].x,
                               s.canale[k].y - s.canale[k - 1].y);
      }
    }
    scartoMinimo = Math.min(scartoMinimo, scarto);
    tortuositaMinima = Math.min(tortuositaMinima, percorso);
  }
  // Con larghezza 22 px sullo schermo, mezza unita' sono 11 px di scarto:
  // sotto questa soglia il fulmine torna a leggersi come una riga.
  assert.ok(scartoMinimo > 0.5,
    'il canale piu\' dritto si scosta solo di ' + scartoMinimo.toFixed(2)
    + ' semilarghezze: a schermo e\' una riga');
  assert.ok(tortuositaMinima > 1.15,
    'il percorso piu\' dritto e\' lungo ' + tortuositaMinima.toFixed(3)
    + ' volte la corda: non c\'e\' abbastanza dettaglio');
});

prova('il dettaglio c\'e\' a ogni scala, non solo nei gomiti grandi', () => {
  // La proprieta' che distingue un frattale da una zigzagata: raffinando si
  // continua a trovare struttura, con ampiezza che cala in modo geometrico.
  // Misurata come scarto quadratico dei punti dalla corda, per meta' e per
  // quarti del canale.
  const s = scarica(0.41);
  function rugosita(da, a) {
    let somma = 0, n = 0;
    for (let k = da; k <= a; k += 1) {
      const t = (s.canale[k].y - s.canale[da].y)
        / ((s.canale[a].y - s.canale[da].y) || 1);
      const corda = s.canale[da].x + t * (s.canale[a].x - s.canale[da].x);
      somma += (s.canale[k].x - corda) ** 2; n += 1;
    }
    return Math.sqrt(somma / n);
  }
  const ultimo = s.canale.length - 1;
  const intera = rugosita(0, ultimo);
  const meta = Math.max(rugosita(0, ultimo >> 1), rugosita(ultimo >> 1, ultimo));
  assert.ok(intera > 0, 'il canale non ha alcuna rugosita\'');
  assert.ok(meta > 0.02,
    'meta\' canale e\' liscia (' + meta.toFixed(4) + '): il dettaglio fine e\' sparito');
  assert.ok(meta < intera,
    'la rugosita\' non cala raffinando: non e\' un frattale, e\' rumore');
});

prova('i rami nascono dal canale e muoiono prima di toccare terra', () => {
  for (let i = 0; i < 60; i += 1) {
    const s = scarica(i / 60);
    assert.ok(s.rami.length >= 2, 'una scarica senza biforcazioni');
    for (let r = 0; r < s.rami.length; r += 1) {
      const punti = s.rami[r].punti;
      const fine = punti[punti.length - 1].y;
      // Un ramo che arriva a terra non e' un ramo: e' un secondo fulmine,
      // e disegnato cosi' sembrerebbe che la scarica sia doppia.
      assert.ok(fine < 1,
        'un ramo arriva a quota ' + fine.toFixed(3) + ', cioe\' sottoterra');
      assert.ok(punti[0].y < fine, 'un ramo risale invece di scendere');
    }
  }
});

prova('la luce sfarfalla: piu\' colpi lungo lo stesso canale', () => {
  const s = scarica(0.23);
  assert.ok(s.colpi.length >= 2,
    'un colpo solo: il lampo si accende e si spegne senza sfarfallare');
  // Campionamento fitto dell'inviluppo: si contano i massimi locali veri.
  const curva = [];
  for (let t = 0; t <= s.durata; t += 2) curva.push(modulo.luceScarica(s, t));
  let picchi = 0;
  for (let i = 1; i < curva.length - 1; i += 1) {
    if (curva[i] > curva[i - 1] && curva[i] >= curva[i + 1] && curva[i] > 0.1) picchi += 1;
  }
  assert.ok(picchi >= 2, 'la curva della luce ha ' + picchi + ' picco: non sfarfalla');
  assert.equal(modulo.luceScarica(s, -1), 0, 'la scarica illumina prima di esistere');
  assert.ok(modulo.luceScarica(s, s.durata) < 0.06,
    'alla fine della durata dichiarata la scarica sta ancora brillando: '
    + 'il ciclo di disegno smetterebbe di aggiornarla mentre e\' ancora accesa');
});

prova('prima del colpo di ritorno c\'e\' solo il leader, debole', () => {
  // E' quello che da' al lampo il suo tempo: il canale si cerca la strada,
  // poi esplode. Senza, si accende tutto insieme e sembra un disegno.
  const s = scarica(0.61);
  const durante = modulo.luceScarica(s, 40);
  const dopo = modulo.luceScarica(s, s.colpi[0].quando + 8);
  assert.equal(durante, 0, 'il leader illumina gia\' come un colpo di ritorno');
  assert.ok(dopo > 0.7, 'il colpo di ritorno non illumina (' + dopo.toFixed(2) + ')');
});

prova('il tick non puo\' diventare un ronzio', () => {
  // Una cella attiva manda decine di scariche al secondo. Senza un
  // intervallo minimo il tick smette di essere una notizia e diventa rumore
  // continuo -- cioe' il fastidio che si voleva evitare.
  const gap = html.match(/const STRIKE_SOUND_MIN_GAP_MS = (\d+);/);
  assert.ok(gap, 'manca il limite di frequenza del tick');
  assert.ok(Number(gap[1]) >= 100,
    'il tick puo\' suonare ogni ' + gap[1] + ' ms: e\' un ronzio');
  assert.match(html, /if \(adesso - strikeLastSound < STRIKE_SOUND_MIN_GAP_MS\) return;/,
    'il limite e\' dichiarato ma non applicato');
  // Secondo freno: sotto raffica il singolo colpo si abbassa, cosi' una
  // cella che scarica in continuazione diventa un picchiettio di fondo.
  assert.match(html, /const fitto = adesso - strikeLastSound < 500 \? 0\.5 : 1;/,
    'il tick suona a volume pieno anche sotto raffica');
  assert.match(html, /uscita\.gain\.value = 0\.055 \* forza \* fitto;/,
    'l\'attenuazione sotto raffica e\' calcolata ma non applicata al volume');
  // Fuori dallo schermo non si suona: un fulmine in Croazia mentre si
  // guarda la Sicilia non riguarda chi sta guardando.
  assert.match(html, /const forza = volumeScarica\(lon, lat\);\s*\n\s*if \(forza <= 0\) return;/,
    'il tick suona anche per scariche che non si vedono');
  assert.match(html, /if \(!strikeSoundOn \|\| document\.hidden\) return;/,
    'il tick suona con la scheda in secondo piano');
  // Il contesto audio nasce dentro il clic, che e' l'unico momento in cui il
  // browser lo permette.
  assert.match(html, /if \(strikeSoundOn\) ensureStrikeAudio\(\);/,
    'il contesto audio non viene creato dentro il gesto dell\'utente');
});

prova('il volume cala verso il bordo e si azzera fuori', () => {
  const volume = new Function(implementazione('volumeScarica')
    + '\nreturn volumeScarica;');
  const contesto = {
    map: {
      getBounds: () => ({ getWest: () => 10, getEast: () => 14,
                          getSouth: () => 36, getNorth: () => 40 }),
      getCenter: () => ({ lng: 12, lat: 38 })
    }
  };
  const fn = new Function('map', implementazione('volumeScarica')
    + '\nreturn volumeScarica;')(contesto.map);
  assert.equal(fn(9, 38), 0, 'una scarica fuori schermo suona lo stesso');
  assert.equal(fn(12, 41), 0, 'una scarica sopra lo schermo suona lo stesso');
  const centro = fn(12, 38), bordo = fn(13.9, 38);
  assert.ok(centro > bordo, 'il bordo suona come il centro');
  assert.ok(bordo > 0, 'il bordo dello schermo e\' gia\' muto');
  void volume;
});

console.log(ok ? 'ESITO: SUPERATO' : 'ESITO: DA RIVEDERE');
process.exitCode = ok ? 0 : 1;
