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
  // Anche le costanti scritte come prodotto (90 * 1000), che si leggono
  // meglio nel sorgente del sito.
  const trovata = html.match(new RegExp('const ' + nome + ' = ([0-9.*+ ]+);'));
  assert.ok(trovata, 'manca la costante ' + nome);
  return 'const ' + nome + ' = ' + trovata[1].trim() + ';';
}

const codice = ['STRIKE_LEADER_MS', 'STRIKE_STROKE_MS', 'STRIKE_BAGLIORE_MS',
  'STRIKE_BRACE_MS', 'CELLA_LAT', 'CELLA_LON', 'CELLA_TAU_MS', 'CELLA_MAX',
  'CELLA_RAGGIO_KM',
  'STRIKE_LIFE_MS', 'STRIKE_MAX'].map(costante).join('\n')
  + '\n\n'
  + ['semeCasuale', 'generaCanale', 'generaRami', 'preparaScarica', 'luceScarica',
     'luceNube', 'luceBrace', 'aggregaCelle', 'limiteSegni']
      .map(implementazione).join('\n\n');
const modulo = new Function(codice
  + '\nreturn {semeCasuale, generaCanale, generaRami, preparaScarica, luceScarica,'
  + ' luceNube, luceBrace, aggregaCelle, limiteSegni};')();

const CELLA_MAX_ATTESO = Number(costante('CELLA_MAX').match(/= (\d+)/)[1]);
const STRIKE_MAX_ATTESO = Number(costante('STRIKE_MAX').match(/= (\d+)/)[1]);
const STRIKE_LIFE_ATTESO = eval(costante('STRIKE_LIFE_MS').match(/= (.+);/)[1]);
const STRIKE_BRACE_FINE = eval(costante('STRIKE_BRACE_MS').match(/= (.+);/)[1]);
const CELLA_RAGGIO_ATTESO = Number(costante('CELLA_RAGGIO_KM').match(/= (\d+)/)[1]);

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

prova('la nube resta accesa oltre lo sfarfallio del canale', () => {
  // Il punto della coltre: il canale sfarfalla in decimi di secondo, la
  // massa d'aria illuminata no. Se la nube seguisse i colpi si vedrebbe
  // lampeggiare mezzo cielo a quaranta millisecondi di distanza, che e'
  // esattamente cio' che un temporale vero non fa.
  const s = scarica(0.31);
  assert.equal(modulo.luceNube(s, 0), 0, 'la nube si illumina prima del colpo');
  const alPicco = modulo.luceNube(s, s.colpi[0].quando + 60);
  assert.ok(alPicco > 0.6, 'la nube non si accende (' + alPicco.toFixed(2) + ')');
  // Fra un colpo e l'altro il canale cala molto e la nube quasi niente.
  const buco = s.colpi[0].quando + 35;
  const canaleNelBuco = modulo.luceScarica(s, buco) / modulo.luceScarica(s, s.colpi[0].quando + 6);
  const nubeNelBuco = modulo.luceNube(s, buco) / alPicco;
  assert.ok(nubeNelBuco > canaleNelBuco + 0.2,
    'la nube segue lo sfarfallio del canale invece di integrarlo');
  // E sopravvive all'ultimo colpo, senza arrivare viva a fine durata.
  assert.ok(modulo.luceNube(s, s.ultimoColpo + 200) > 0.08,
    'la nube si spegne insieme al canale');
  assert.ok(modulo.luceNube(s, s.durata) < 0.06,
    'la nube e\' ancora accesa quando il ciclo smette di aggiornarla: '
    + 'si spegnerebbe di scatto');
});

prova('le braci partono quando i colpi finiscono, e durano poco', () => {
  const s = scarica(0.52);
  assert.equal(modulo.luceBrace(s, s.ultimoColpo - 20), 0,
    'il canale si raffredda mentre sta ancora scaricando');
  assert.ok(modulo.luceBrace(s, s.ultimoColpo + 5) > 0.8, 'le braci non si accendono');
  assert.ok(modulo.luceBrace(s, s.ultimoColpo + 300) === 0,
    'le braci non si spengono mai');
  // Devono essere piu' brevi della nube: e' un filo di canale, non una massa
  // d'aria.
  const braceFinita = s.ultimoColpo + STRIKE_BRACE_FINE;
  assert.ok(modulo.luceNube(s, braceFinita) > modulo.luceBrace(s, braceFinita),
    'le braci durano quanto la nube illuminata');
});

prova('le scariche vicine fanno un nucleo solo, quelle lontane no', () => {
  const adesso = 100000;
  const vicine = [
    { lat: 41.0, lon: 14.0, nato: adesso - 1000 },
    { lat: 41.03, lon: 14.04, nato: adesso - 2000 },
    { lat: 41.01, lon: 13.97, nato: adesso - 3000 }
  ];
  assert.equal(modulo.aggregaCelle(vicine, adesso).length, 1,
    'tre scariche nello stesso temporale danno piu\' di un nucleo');
  const lontane = [
    { lat: 41.0, lon: 14.0, nato: adesso - 1000 },
    { lat: 45.5, lon: 9.2, nato: adesso - 1000 }
  ];
  assert.equal(modulo.aggregaCelle(lontane, adesso).length, 2,
    'Campania e Lombardia finiscono nello stesso nucleo');
});

prova('il nucleo si mette dove cadono i fulmini, non sulla griglia', () => {
  // Con il centro della cella di griglia due celle adiacenti darebbero due
  // bolle allineate; con la media pesata il nucleo sta sull'attivita' vera.
  const adesso = 100000;
  const punti = [];
  for (let i = 0; i < 6; i += 1) punti.push({ lat: 41.0, lon: 14.0, nato: adesso - 100 });
  punti.push({ lat: 41.02, lon: 14.03, nato: adesso - 100 });
  const nuclei = modulo.aggregaCelle(punti, adesso);
  assert.equal(nuclei.length, 1);
  assert.ok(Math.abs(nuclei[0].lat - 41.0) < 0.01,
    'il nucleo e\' a ' + nuclei[0].lat.toFixed(3) + ' invece che sul grosso delle scariche');
  assert.ok(nuclei[0].lat > 41.0, 'la media pesata non tiene conto della scarica isolata');
});

prova('il nucleo misura un ritmo, non un totale', () => {
  // E' la proprieta' che fa spegnere da sola una cella che ha smesso, senza
  // dover cancellare niente. Le eta' restano dentro la finestra che il sito
  // conserva davvero (STRIKE_LIFE_MS): provare con scariche piu' vecchie
  // misurerebbe un caso che non puo' presentarsi.
  const adesso = 500000;
  const attiva = [], spenta = [];
  for (let i = 0; i < 20; i += 1) {
    attiva.push({ lat: 41, lon: 14, nato: adesso - i * 1000 });
    // Stesso numero di scariche, ma tutte verso il fondo della finestra.
    spenta.push({ lat: 41, lon: 14, nato: adesso - 125000 - i * 1000 });
  }
  const pesoAttiva = modulo.aggregaCelle(attiva, adesso)[0].peso;
  const nucleiSpenta = modulo.aggregaCelle(spenta, adesso);
  assert.ok(pesoAttiva > 15, 'una cella che scarica adesso pesa poco: ' + pesoAttiva.toFixed(1));
  assert.ok(nucleiSpenta[0].peso < pesoAttiva / 3,
    'una cella ferma da due minuti pesa ' + nucleiSpenta[0].peso.toFixed(1)
    + ' contro ' + pesoAttiva.toFixed(1) + ': non si distingue da una attiva');
  // E quando le sue scariche escono dalla finestra deve sparire del tutto.
  const uscita = spenta.map(function (s) { return { lat: 41, lon: 14, nato: adesso - 400000 }; });
  assert.equal(modulo.aggregaCelle(uscita, adesso).length, 0,
    'una cella spenta da sette minuti compare ancora');
  // Doppio ritmo, peso circa doppio: e' quello che si vuole vedere.
  const doppia = attiva.concat(attiva.map(function (s) {
    return { lat: 41.01, lon: 14.01, nato: s.nato + 500 };
  }));
  const pesoDoppia = modulo.aggregaCelle(doppia, adesso)[0].peso;
  assert.ok(pesoDoppia > pesoAttiva * 1.7,
    'raddoppiando le scariche il nucleo non cresce');
});

prova('una linea temporalesca resta una fila di nuclei', () => {
  // Il rischio del raggruppamento per vicinanza: incatenando i vicini dei
  // vicini, trecento chilometri di celle contigue diventerebbero una sola
  // bolla enorme centrata sul nulla. Il raggio fisso lo impedisce.
  const adesso = 100000;
  const linea = [];
  for (let i = 0; i < 30; i += 1) {
    // Una fila lunga circa 300 km, una scarica ogni 10 km.
    linea.push({ lat: 41 + i * 0.09, lon: 14, nato: adesso - 1000 });
  }
  const nuclei = modulo.aggregaCelle(linea, adesso);
  assert.ok(nuclei.length >= 6,
    'trecento chilometri di temporali sono diventati ' + nuclei.length
    + ' nucleo: la fila e\' collassata in una macchia sola');
  // E nessun nucleo deve essere piu' largo del raggio dichiarato.
  for (const n of nuclei) {
    const dentro = linea.filter(function (s) {
      return Math.hypot((s.lat - n.lat) * 111, (s.lon - n.lon) * 111
        * Math.cos(n.lat * Math.PI / 180)) <= CELLA_RAGGIO_ATTESO + 0.5;
    });
    assert.ok(dentro.length > 0, 'un nucleo senza scariche vicine');
  }
});

prova('i nuclei sono ordinati per attivita\' e limitati in numero', () => {
  const adesso = 100000;
  const tante = [];
  for (let i = 0; i < 200; i += 1) {
    // Duecento celle distinte, con attivita' decrescente.
    for (let k = 0; k <= i % 7; k += 1) {
      tante.push({ lat: 36 + i * 0.5, lon: 8 + (i % 13) * 0.4, nato: adesso - 1000 });
    }
  }
  const nuclei = modulo.aggregaCelle(tante, adesso);
  assert.ok(nuclei.length <= CELLA_MAX_ATTESO,
    'disegnerei ' + nuclei.length + ' nuclei: uno per fotogramma ciascuno');
  for (let i = 1; i < nuclei.length; i += 1) {
    assert.ok(nuclei[i - 1].peso >= nuclei[i].peso,
      'i nuclei non sono in ordine di attivita\': il taglio scarterebbe i piu\' forti');
  }
});

prova('da lontano si disegnano pochi segni, da vicino tutti', () => {
  // A zoom cinque un glifo da dieci pixel copre trenta chilometri: e' la
  // ragione per cui il limite esiste.
  const largo = modulo.limiteSegni(5), medio = modulo.limiteSegni(7), stretto = modulo.limiteSegni(9);
  assert.ok(largo.quanti < medio.quanti && medio.quanti < stretto.quanti,
    'il numero di segni non cresce avvicinandosi');
  assert.ok(largo.eta < medio.eta && medio.eta < stretto.eta,
    'l\'eta\' massima del segno non cresce avvicinandosi');
  assert.ok(largo.quanti <= 30, 'da lontano si disegnano ancora ' + largo.quanti + ' segni');
  assert.equal(stretto.eta, STRIKE_LIFE_ATTESO,
    'da vicino i segni non vivono piu\' quanto la scarica');
  assert.equal(stretto.quanti, STRIKE_MAX_ATTESO,
    'da vicino si perdono ancora delle scariche');
});

prova('il segno sbiadisce sul limite vero, non su quello massimo', () => {
  // Da lontano il segno vive dodici secondi: se il dissolvere fosse
  // calcolato sui centocinquanta, sparirebbe di scatto ancora acceso.
  const disegno = html.match(/function drawLiveStrikes\(\)[\s\S]*?\n {6}\}/);
  assert.ok(disegno, 'manca il disegno delle scariche');
  assert.match(disegno[0], /const vita = 1 - eta \/ limite\.eta;/,
    'il dissolvere non segue il limite di zoom');
  assert.match(disegno[0], /const limite = limiteSegni\(map\.getZoom\(\)\);/,
    'il limite non dipende piu\' dallo zoom');
  // Si scorre dal fondo: liveStrikes e' in ordine di arrivo, e quando se ne
  // possono disegnare poche vanno tenute le piu' recenti.
  assert.match(disegno[0], /for \(let i = liveStrikes\.length - 1; i >= 0; i -= 1\)/,
    'scorrendo dall\'inizio, con il limite attivo resterebbero in mappa le '
    + 'scariche piu\' vecchie invece delle ultime arrivate');
});

prova('il nucleo si ricalcola di rado e si disegna copiando', () => {
  // Un gradiente radiale per nucleo e per fotogramma costerebbe piu' di
  // tutto il resto del disegno.
  assert.match(html, /if \(adesso - celleCalcolateA > 600\)/,
    'l\'aggregazione si rifa\' a ogni fotogramma');
  assert.match(html, /context\.drawImage\(bolla, /,
    'i nuclei non sono piu\' copie di un\'immagine pre-disegnata');
  const disegna = html.match(/function disegnaNuclei\([\s\S]*?\n {6}\}/);
  assert.ok(disegna, 'manca il disegno dei nuclei');
  assert.doesNotMatch(disegna[0], /createRadialGradient/,
    'si crea un gradiente per nucleo a ogni fotogramma');
});

prova('fra un lampo e l\'altro il nucleo continua a respirare', () => {
  // Il battito ha un periodo fra 0,7 e 1,4 secondi: a un fotogramma ogni
  // secondo e mezzo sarebbe uno scatto, non un respiro.
  const ciclo = html.match(/if \(timestamp - strikeLastDraw >= \(lampeggia \? 0 : (\d+)\)\)/);
  assert.ok(ciclo, 'manca la cadenza del ciclo di disegno');
  assert.ok(Number(ciclo[1]) <= 120,
    'fra i lampi si ridisegna ogni ' + ciclo[1] + ' ms: il respiro scatta');
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
