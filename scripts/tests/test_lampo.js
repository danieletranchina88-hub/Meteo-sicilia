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
  + ['semeCasuale', 'generaCanale', 'generaRagno', 'preparaScarica', 'luceScarica',
     'luceNube', 'luceBrace', 'aggregaCelle', 'limiteSegni']
      .map(implementazione).join('\n\n');
const modulo = new Function(codice
  + '\nreturn {semeCasuale, generaCanale, generaRagno, preparaScarica, luceScarica,'
  + ' luceNube, luceBrace, aggregaCelle, limiteSegni};')();

const CELLA_MAX_ATTESO = Number(costante('CELLA_MAX').match(/= (\d+)/)[1]);
const STRIKE_MAX_ATTESO = Number(costante('STRIKE_MAX').match(/= (\d+)/)[1]);
const STRIKE_LIFE_ATTESO = eval(costante('STRIKE_LIFE_MS').match(/= (.+);/)[1]);
const STRIKE_BRACE_FINE = eval(costante('STRIKE_BRACE_MS').match(/= (.+);/)[1]);
const CELLA_RAGGIO_ATTESO = Number(costante('CELLA_RAGGIO_KM').match(/= (\d+)/)[1]);
const STRIKE_BAGLIORE_ATTESO = Number(costante('STRIKE_BAGLIORE_MS').match(/= (\d+)/)[1]);

let ok = true;
function prova(nome, fn) {
  try { fn(); console.log('PASS ' + nome); }
  catch (error) { ok = false; console.log('FALLITO ' + nome + ': ' + error.message); }
}

// Tutto il codice che disegna il lampo, non una funzione sola: il bagliore
// e il ragno vivono in due posti diversi, e una prova che ne guardasse uno
// solo passerebbe mentre l'altro e' azzurro.
const regioneLampo = (function () {
  const da = html.indexOf('      function disegnaNubeIlluminata(');
  assert.ok(da >= 0, 'manca il disegno della nube illuminata');
  const disegno = html.match(/function drawLiveStrikes\(\)[\s\S]*?\n {6}\}/);
  assert.ok(disegno, 'manca il disegno delle scariche');
  return html.slice(da, html.indexOf(disegno[0]) + disegno[0].length);
})();

// Che cosa distingue la lavanda di un temporale vero dall'azzurro che
// copriva le nuvole. Misurato sulla fotografia di riferimento, la luce
// diffusa dalla nube ha SEMPRE il verde come canale piu' basso
// (133,115,128 / 189,168,190 / 229,217,240): e' magenta chiaro. Un azzurro
// no -- in 133,162,248 il verde sta in mezzo e il blu supera il rosso di
// centoquindici. Sono due cose diverse, e la prova deve saperle distinguere
// o finisce per vietare anche il colore giusto.
function azzurro(c) {
  if (c.b > c.r + 16) return true;
  return c.b > c.r && c.g >= Math.min(c.r, c.b);
}

function ritaglioMinimo() {
  const m = html.match(/function disegnaNubeIlluminata\([\s\S]*?\n {6}\}/);
  assert.ok(m, 'manca il disegno della nube illuminata');
  return m[0];
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
  assert.deepEqual(a.ragno, b.ragno, 'due scariche uguali danno ragni diversi');
  assert.deepEqual(a.colpi, b.colpi, 'due scariche uguali danno colpi diversi');
  assert.notDeepEqual(a.ragno, c.ragno, 'semi diversi danno lo stesso ragno');
});

prova('il fulmine e\' visto dall\'alto, non di lato', () => {
  // La cosa che questa modifica ha corretto, e che leggendo il disegno
  // vecchio sembrava giusta: una saetta verticale che sale dal punto di
  // impatto e' un fulmine visto DI LATO. Su una mappa zenitale quel canale
  // sarebbe un punto. Dall'alto si vede la nube accendersi da dentro e i
  // ragni strisciare sulla sommita', e basta.
  const disegno = html.match(/function drawLiveStrikes\(\)[\s\S]*?\n {6}\}/);
  assert.ok(disegno, 'manca il disegno delle scariche');
  assert.doesNotMatch(disegno[0], /const altezza =/,
    'il lampo ha di nuovo un\'altezza: e\' tornato a essere una veduta laterale');
  assert.doesNotMatch(disegno[0], /punto\.y - altezza/,
    'si disegna ancora sopra il punto, cioe\' di lato');
  assert.match(disegno[0], /puntiFilamento\(/,
    'i canali non strisciano piu\' sul piano');
  // E il tracciatore deve davvero ruotare attorno alla direzione, non
  // limitarsi a salire.
  const tracciante = html.match(/function puntiFilamento\([\s\S]*?\n {6}\}/);
  assert.ok(tracciante, 'manca il tracciatore dei filamenti');
  assert.match(tracciante[0], /Math\.cos\(direzione\)/,
    'il filamento non viene orientato nel piano');
});

prova('ogni filamento si allontana e non torna mai indietro', () => {
  for (let i = 0; i < 60; i += 1) {
    const s = scarica(i / 60);
    for (const filo of s.ragno) {
      let precedente = -1;
      for (let k = 0; k < filo.forma.length; k += 1) {
        assert.ok(filo.forma[k].y >= precedente,
          'il filamento torna verso il punto di partenza');
        precedente = filo.forma[k].y;
      }
      assert.equal(filo.forma[0].y, 0, 'il filamento non parte dalla scarica');
      assert.equal(filo.forma[filo.forma.length - 1].y, 1,
        'il filamento non arriva alla sua punta');
    }
  }
});

prova('i filamenti si spargono attorno al giro', () => {
  // Sorteggiare sei direzioni a caso le fa finire spesso tutte dalla stessa
  // parte, e il risultato sembra un pennello invece di una scarica. Le
  // direzioni sono quindi sparse sul giro con un po' di disordine: qui si
  // misura che nessun semicerchio resti vuoto.
  for (let i = 0; i < 60; i += 1) {
    const s = scarica(i / 60);
    const angoli = s.ragno.map((f) => f.direzione)
      .map((a) => ((a % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2))
      .sort((a, b) => a - b);
    let buco = (angoli[0] + Math.PI * 2) - angoli[angoli.length - 1];
    for (let k = 1; k < angoli.length; k += 1) {
      buco = Math.max(buco, angoli[k] - angoli[k - 1]);
    }
    assert.ok(buco < Math.PI * 1.05,
      'i filamenti lasciano scoperti ' + Math.round(buco * 180 / Math.PI)
      + ' gradi di giro: il ragno pende tutto da una parte');
  }
});

prova('il filamento e\' davvero spezzato, non un raggio dritto', () => {
  // La prova che smaschera un frattale che non si vede: due misure, lo
  // scarto laterale in unita' di semilarghezza e la tortuosita', cioe'
  // quanto il percorso e' piu' lungo della corda.
  let scartoMinimo = Infinity, tortuositaMinima = Infinity;
  for (let i = 0; i < 60; i += 1) {
    for (const filo of scarica(i / 60).ragno) {
      let scarto = 0, percorso = 0;
      for (let k = 0; k < filo.forma.length; k += 1) {
        scarto = Math.max(scarto, Math.abs(filo.forma[k].x));
        if (k) {
          percorso += Math.hypot(filo.forma[k].x - filo.forma[k - 1].x,
                                 filo.forma[k].y - filo.forma[k - 1].y);
        }
      }
      scartoMinimo = Math.min(scartoMinimo, scarto);
      tortuositaMinima = Math.min(tortuositaMinima, percorso);
    }
  }
  // La soglia e' in unita' di semilarghezza del filamento. Un terzo
  // significa che il piu' dritto dei filamenti lunghi si piega comunque di
  // una quindicina di pixel a schermo, e il piu' corto di tre su venti di
  // lunghezza: entrambi si leggono come scariche e non come raggi.
  assert.ok(scartoMinimo > 0.3,
    'il filamento piu\' dritto si scosta solo di ' + scartoMinimo.toFixed(2)
    + ' semilarghezze: a schermo e\' un raggio');
  assert.ok(tortuositaMinima > 1.05,
    'il percorso piu\' dritto e\' lungo ' + tortuositaMinima.toFixed(3)
    + ' volte la corda: non c\'e\' abbastanza dettaglio');
});

prova('il dettaglio c\'e\' a ogni scala, non solo nei gomiti grandi', () => {
  // La proprieta' che distingue un frattale da una zigzagata: raffinando si
  // continua a trovare struttura, con ampiezza che cala in modo geometrico.
  const punti = scarica(0.41).ragno[0].forma;
  function rugosita(da, a) {
    let somma = 0, n = 0;
    for (let k = da; k <= a; k += 1) {
      const t = (punti[k].y - punti[da].y) / ((punti[a].y - punti[da].y) || 1);
      const corda = punti[da].x + t * (punti[a].x - punti[da].x);
      somma += (punti[k].x - corda) ** 2; n += 1;
    }
    return Math.sqrt(somma / n);
  }
  const ultimo = punti.length - 1;
  const intera = rugosita(0, ultimo);
  const meta = Math.max(rugosita(0, ultimo >> 1), rugosita(ultimo >> 1, ultimo));
  assert.ok(intera > 0, 'il filamento non ha alcuna rugosita\'');
  assert.ok(meta > 0.01,
    'meta\' filamento e\' liscia (' + meta.toFixed(4) + '): il dettaglio fine e\' sparito');
  assert.ok(meta < intera,
    'la rugosita\' non cala raffinando: non e\' un frattale, e\' rumore');
});

prova('il canale si assottiglia e si spegne verso la punta', () => {
  // Un canale a spessore costante fino alla punta sembra uno stecco: era
  // il difetto che faceva leggere il ragno come una figura disegnata invece
  // che come una scarica. Spessore e luce devono calare col percorso.
  const sfumato = html.match(/function strokeSfumato\([\s\S]*?\n {6}\}/);
  assert.ok(sfumato, 'manca il tracciatore sfumato');
  assert.match(sfumato[0], /const calo = \(1 - t\)/,
    'lo spessore non dipende piu\' da quanto il canale si e\' allontanato');
  assert.match(sfumato[0], /context\.lineWidth = Math\.max\([0-9.]+, larghezza \* calo\)/,
    'lo spessore non cala verso la punta');
  assert.match(sfumato[0], /alfa \* calo/,
    'la luce non cala verso la punta: il canale finisce di netto');
  // E il disegno deve usarlo davvero, in tutte le passate.
  const disegno = html.match(/function drawLiveStrikes\(\)[\s\S]*?\n {6}\}/);
  assert.ok(!/context\.lineWidth = \([0-9.]+ \? /.test(disegno[0]),
    'restano passate a spessore fisso');
  const usi = (disegno[0].match(/strokeSfumato\(/g) || []).length;
  assert.ok(usi >= 3, 'il tracciatore sfumato viene usato solo ' + usi + ' volte');
});

prova('mentre il lampo brilla il simbolo non lo copre', () => {
  // Il glifo e' un'etichetta: sovrapposto alla luce vera la fa sembrare un
  // disegno. Deve comparire quando la scarica ha finito di illuminare.
  const disegno = html.match(/function drawLiveStrikes\(\)[\s\S]*?\n {6}\}/);
  assert.match(disegno[0], /if \(eta < s\.durata \* 0\.8\) continue;/,
    'il simbolo viene disegnato sopra il lampo acceso');
});

prova('i rami sono piu\' corti del filamento da cui nascono', () => {
  // Un ramo lungo quanto il suo filamento non e' una biforcazione: e' un
  // secondo fulmine, e disegnato cosi' il ragno sembra una ragnatela.
  let trovati = 0;
  for (let i = 0; i < 60; i += 1) {
    for (const filo of scarica(i / 60).ragno) {
      for (const ramo of filo.rami) {
        trovati += 1;
        assert.ok(ramo.lunghezza < filo.lunghezza * 0.6,
          'un ramo lungo ' + ramo.lunghezza.toFixed(2) + ' su un filamento da '
          + filo.lunghezza.toFixed(2));
        assert.ok(ramo.nodo > 0.2 && ramo.nodo < 0.8,
          'un ramo nasce sulla punta o sul punto della scarica');
        // Si stacca di lato: un ramo parallelo al filamento non si vede.
        const scarto = Math.abs(ramo.direzione - filo.direzione);
        assert.ok(scarto > 0.3, 'un ramo corre parallelo al filamento');
      }
    }
  }
  assert.ok(trovati > 20, 'quasi nessuna scarica ha biforcazioni: ne ho contate ' + trovati);
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

prova('i canali diagnostici non diventano una sagoma di nube', () => {
  // Fase delle nubi, tipo di nube, nebbia, polvere, neve sono RGB
  // artificiali: il rosa o il verde acceso segnalano una diagnosi (ghiaccio
  // piccolo, polvere, manto nevoso), non "piu' chiaro, piu' nube". Usarne
  // la luminanza come sagoma sarebbe fisica finta con l'aria di essere
  // vera -- illuminerebbe la nube secondo una scala che misura tutt'altro.
  const maschera = implementazione('costruisciMascheraNube');
  assert.match(maschera, /if \(!product \|\| !\["scene", "grey"\]\.includes\(product\.mode\)\) return null;/,
    'i canali diagnostici (fase, tipo, nebbia, polvere, neve) producono ancora una maschera');
  // E la tavolozza dichiarata dei prodotti deve avere davvero quella
  // proprieta', o il controllo sopra non filtra nulla.
  assert.match(html, /cloudphase: \{[\s\S]{0,200}mode: "diagnostic"/,
    'il prodotto diagnostico non dichiara piu\' la propria natura');
  assert.match(html, /geocolour: \{[\s\S]{0,200}mode: "scene"/,
    'il composito a colori naturali non e\' piu\' marcato "scene"');
});

prova('sui compositi a colore, terra e luci non contano come nube', () => {
  // Un deserto, un tetto assolato o le luci di una citta' possono essere
  // chiari quanto una sommita' di cumulo -- ma sono COLORATI, e una nube
  // no. Sul grigio dell'infrarosso non c'e' colore da giudicare: il filtro
  // vale solo sui compositi "scene".
  const maschera = implementazione('costruisciMascheraNube');
  assert.match(maschera,
    /const saturazione = \(Math\.max\(r, g, b\) - Math\.min\(r, g, b\)\) \/ Math\.max\(1, r, g, b\);/,
    'manca il calcolo della saturazione');
  assert.match(maschera,
    /const neutro = product\.mode === "scene"\s*\n\s*\? 1 - clamp\(\(saturazione - 0\.12\) \/ 0\.5, 0, 1\)\s*\n\s*: 1;/,
    'la saturazione non penalizza piu\' la nuvolosita\' sui compositi a colore');
  assert.match(maschera, /const nuvolosita = grezza \* grezza \* \(3 - 2 \* grezza\) \* neutro \* alfaSorgente;/,
    'il fattore neutro non entra piu\' nel calcolo della nuvolosita\'');
});

prova('una richiesta superata non ripubblica la sua immagine vecchia', () => {
  // Il caricamento di un\'immagine e la sua codifica in PNG sono due passi,
  // e il secondo e' asincrono (canvas.toBlob). Se nel frattempo parte ED
  // ARRIVA una richiesta piu\' nuova, la codifica piu\' lenta della vecchia
  // vince comunque la corsa: senza un controllo qui, ripubblicherebbe
  // un\'immagine superata SOPRA quella giusta gia\' in mostra, e la maschera
  // dei fulmini tornerebbe a descrivere la nube di prima.
  const pubblica = implementazione('publishSatelliteClouds');
  assert.match(pubblica, /const publicationToken = cloudToken;/,
    'la pubblicazione non tiene piu\' traccia di quale richiesta e\'');
  // Il controllo che conta e' quello dentro update(): e' il punto dove si
  // decide se mostrare davvero l\'immagine e la sua maschera, e deve essere
  // la PRIMA cosa che fa -- prima ancora del try, o l\'aggiornamento
  // partirebbe comunque.
  const update = pubblica.match(/const update = function \(url, isObjectUrl\) \{[\s\S]*?\n {8}\};/);
  assert.ok(update, 'manca la funzione che pubblica davvero il fotogramma');
  assert.match(update[0], /^const update = function \(url, isObjectUrl\) \{\s*\n\s*if \(publicationToken !== cloudToken\) \{/,
    'update() non e\' piu\' la prima cosa a controllare se e\' ancora la richiesta giusta');
  assert.match(update[0], /mascheraNube = maschera;/,
    'la maschera vera non si aggiorna piu\' dentro update()');
  // E il secondo controllo, dentro la codifica asincrona: senza questo,
  // una richiesta scartata rivelerebbe comunque l\'immagine vecchia una
  // volta finita la sua codifica in PNG.
  const dopoIlBlob = pubblica.match(/canvas\.toBlob\(function \(blob\) \{[\s\S]*?\n {12}\}, "image\/png"\);/);
  assert.ok(dopoIlBlob, 'manca la codifica asincrona del fotogramma');
  assert.match(dopoIlBlob[0], /if \(publicationToken !== cloudToken\) \{ URL\.revokeObjectURL\(url\); return; \}/,
    'la codifica finita in ritardo non controlla piu\' se la richiesta e\' ancora quella giusta');
});

prova('scorrendo fino a un istante gia\' visto, il lampo sagoma la nube giusta', () => {
  // pubblicaFotogrammaInCache ripubblica un fotogramma senza toccare la
  // rete. Se non ripristina anche la maschera che gli appartiene, il
  // bagliore resterebbe sagomato sull\'ultima immagine scaricata dal vivo
  // invece che su quella davvero in mostra dopo lo scorrimento.
  const dallaCache = implementazione('pubblicaFotogrammaInCache');
  assert.match(dallaCache, /mascheraNube = frame\.maschera \|\| null;/,
    'ripubblicando dalla cache la maschera dei fulmini non si aggiorna piu\'');
  const ricorda = implementazione('rememberCloudFrame');
  assert.match(ricorda, /maschera: maschera \|\| null/,
    'la cache non conserva piu\' la maschera insieme al fotogramma');
  // E il precaricamento in sottofondo -- quello che prepara gli istanti
  // vicini prima che l\'utente li chieda -- deve costruirla anche lui, o i
  // fotogrammi precaricati arriverebbero con la sagoma sempre assente.
  const scarica = implementazione('scaricaFotogramma');
  assert.match(scarica, /costruisciMascheraNube\(canvas, box, product\)/,
    'il precaricamento in sottofondo non costruisce piu\' la sagoma');
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

prova('la luce esce dalle nubi, non da un disco', () => {
  // Un lampo visto dall'alto non illumina un cerchio: illumina LA NUBE, e
  // la macchia luminosa ha la sagoma della sommita' nuvolosa. Un disco
  // morbido, per quanto ben sfumato, si riconosce subito come disegnato.
  // La sagoma non si inventa: l'immagine satellitare passa gia' da una
  // canvas nostra, e da quei pixel si ricava.
  assert.match(html, /maschera = costruisciMascheraNube\(canvas, box, product\);/,
    'la maschera delle nubi non viene piu\' costruita quando arriva un fotogramma');
  const maschera = html.match(/function costruisciMascheraNube\([\s\S]*?\n {6}\}/);
  assert.ok(maschera, 'manca la costruzione della maschera');
  assert.match(maschera[0], /0\.2126/,
    'la nuvolosita\' non si misura piu\' dalla luminanza percepita');
  assert.match(maschera[0], /px\[i \+ 3\] = Math\.round\(255 \* nuvolosita/,
    'la maschera non finisce nel canale alfa: non ritaglierebbe niente');
  // La funzione RESTITUISCE la maschera, non muta piu' la globale in
  // silenzio: e' quello che permette a chi chiama di deciderne le sorti
  // (applicarla solo se questa e' ancora la richiesta piu' recente).
  assert.doesNotMatch(maschera[0], /mascheraNube = /,
    'la funzione muta ancora la globale invece di restituire il risultato');
  assert.match(maschera[0], /return \{ tela: piccola,/,
    'la funzione non restituisce piu\' la maschera costruita');
  // E il ritaglio non deve poter spegnere del tutto un lampo: la posizione
  // di una scarica ha un chilometro di incertezza, e basta che cada in uno
  // squarcio fra le nubi perche' la maschera le porti via tutta la luce.
  const minimo = ritaglioMinimo().match(/tondo\((0\.[0-9]+), largo \* [0-9.]+\);/);
  assert.ok(minimo, 'non c\'e\' piu\' un minimo garantito: una scarica caduta fra '
    + 'due nubi diventerebbe invisibile');
  assert.ok(Number(minimo[1]) > 0.08 && Number(minimo[1]) < 0.45,
    'il minimo tondo vale ' + minimo[1] + ': o non garantisce niente, o pareggia '
    + 'la luce ritagliata e il ritaglio smette di dare forma');
  // E il minimo deve restare un minimo: se pareggiasse la passata
  // ritagliata, il ritaglio si limiterebbe a togliere luce invece di dare
  // forma, e il bagliore uscirebbe troppo debole per vedersi.
  const passate = (ritaglioMinimo().match(/drawImage\(telaBagliore/g) || []).length;
  assert.ok(passate >= 2,
    'la luce ritagliata sulle nubi si somma una volta sola: non e\' abbastanza '
    + 'forte da farsi leggere come sagoma');
  const ritaglio = html.match(/function disegnaNubeIlluminata\([\s\S]*?\n {6}\}/);
  assert.ok(ritaglio, 'manca il disegno della nube illuminata');
  assert.match(ritaglio[0], /globalCompositeOperation = "destination-in"/,
    'il bagliore non viene piu\' ritagliato sulla sagoma delle nubi');
  // E deve esistere la via di scampo: senza maschera, o con la mappa
  // inclinata -- dove il riquadro dell'immagine non e' piu' un rettangolo
  // sullo schermo -- si torna al bagliore tondo invece di sbagliare.
  assert.match(ritaglio[0], /map\.getPitch\(\) > 4/,
    'con la mappa inclinata il ritaglio finirebbe fuori posto');
  assert.match(ritaglio[0], /if \(!mascheraNube \|\| inclinata \|\| !showSatelliteClouds\)/,
    'senza immagine satellitare il lampo resterebbe invisibile');
});

prova('la luce del lampo e\' bianca calda, non azzurra', () => {
  // La luce che esce dalla sommita' di una nube ha attraversato chilometri
  // di ghiaccio: l'azzurro e' il colore del canale nudo a trentamila gradi,
  // ma diffuso resta un bianco appena caldo. Un lampo azzurro su una mappa
  // e' un lampo visto da vicino e al buio, non da un satellite.
  const colori = [...regioneLampo.matchAll(/rgba\((\d+),(\d+),(\d+),/g)]
    .map((m) => ({ r: +m[1], g: +m[2], b: +m[3] }));
  assert.ok(colori.length > 6, 'non trovo i colori del lampo');
  // Solo i colori CHIARI: i contorni scuri del glifo sono quasi neri, e un
  // nero ha sempre piu' blu che rosso senza per questo essere azzurro.
  const chiari = colori.filter((c) => c.r + c.g + c.b > 320);
  // Che cosa distingue la lavanda di un temporale vero dall'azzurro che
  // copriva le nuvole. Misurato sulla fotografia di riferimento, la luce
  // diffusa dalla nube ha SEMPRE il verde come canale piu' basso
  // (133,115,128 / 189,168,190 / 229,217,240): e' magenta chiaro. Un
  // azzurro no -- in 133,162,248 il verde sta in mezzo, e il blu supera il
  // rosso di centoquindici. Sono due cose diverse e la prova deve saperle
  // distinguere, o vieta anche il colore giusto.
  const freddi = chiari.filter(azzurro);
  // Gli unici colori freddi ammessi sono le due tappe piu' esterne del
  // gradiente della nube: la luce diffusa vira davvero al blu sul margine,
  // ma con un'opacita' che si conta in centesimi.
  const fringia = [...regioneLampo.matchAll(
    /g\.addColorStop\(([0-9.]+), "rgba\((\d+),(\d+),(\d+),([^"]*)"/g)]
    .filter((m) => azzurro({ r: +m[2], g: +m[3], b: +m[4] }));
  for (const m of fringia) {
    assert.ok(+m[1] >= 0.8,
      'la tinta fredda compare gia\' a ' + m[1] + ' del raggio: non e\' un '
      + 'accenno sul bordo, e\' il colore del lampo');
    const alfa = m[5].match(/\(([0-9.]+) \* a\)/);
    assert.ok(!alfa || Number(alfa[1]) <= 0.05,
      'la tinta fredda del bordo ha opacita\' ' + (alfa && alfa[1]) + ': si vede come azzurro');
  }
  assert.equal(freddi.length, fringia.length,
    'il lampo ha ' + (freddi.length - fringia.length) + ' colori chiari piu\' blu '
    + 'che rossi oltre all\'accenno sul bordo: e\' tornato azzurro ('
    + freddi.map((c) => c.r + ',' + c.g + ',' + c.b).join(' / ') + ')');
  // E il nucleo della cella, che e' la macchia piu' larga di tutte.
  const nucleo = html.match(/const NUCLEO_COLORI = \[([\s\S]*?)\];/);
  assert.ok(nucleo, 'manca la tavolozza del nucleo');
  for (const m of nucleo[1].matchAll(/\[(\d+), (\d+), (\d+)\]/g)) {
    assert.ok(+m[1] >= +m[3],
      'un nucleo con piu\' blu che rosso (' + m[1] + ',' + m[2] + ',' + m[3]
      + '): tinge di azzurro tutta la sommita\' della nube');
  }
});

prova('l\'alone illumina le nuvole invece di coprirle', () => {
  // Due modi di coprire: troppo largo e troppo a lungo. Il primo spalma una
  // tinta piatta su mezza cella, il secondo lascia la macchia sulla mappa
  // quando il lampo e' gia' finito.
  const largo = regioneLampo.match(/const largo = \((\d+) \+ (\d+) \* nube\) \* scala;/);
  assert.ok(largo, 'non trovo il raggio della nube illuminata');
  const massimo = Number(largo[1]) + Number(largo[2]);
  // Due limiti diversi, perche' sono due cose diverse. La luce RITAGLIATA
  // puo' essere larga: allargarla illumina piu' nube, non copre di piu', ed
  // e' quello che fa un lampo dentro un cumulo. Oltre una certa scala pero'
  // illuminerebbe celle che non hanno scaricato.
  assert.ok(massimo <= 140,
    'la luce ritagliata arriva a ' + massimo + ' pixel: illuminerebbe celle '
    + 'diverse da quella che ha scaricato');
  // Il bagliore TONDO invece non sa dove sia la nube, quindi deve restare
  // stretto: e' quello che prima copriva tutto.
  const ridotto = ritaglioMinimo().match(/tondo\(1, largo \* ([0-9.]+)\);/);
  assert.ok(ridotto, 'il bagliore tondo usa ancora il raggio pieno');
  assert.ok(massimo * Number(ridotto[1]) <= 90,
    'senza maschera il bagliore arriva a ' + Math.round(massimo * Number(ridotto[1]))
    + ' pixel: li\' non illumina la nube, la copre');
  // E deve spegnersi in fretta.
  const s = scarica(0.44);
  assert.ok(modulo.luceNube(s, s.ultimoColpo + 250) < 0.12,
    'un quarto di secondo dopo l\'ultimo colpo la nube e\' ancora accesa');
  assert.ok(STRIKE_BAGLIORE_ATTESO <= 420,
    'il bagliore dura ' + STRIKE_BAGLIORE_ATTESO + ' ms: resta sulla mappa');
  // Il centro puo' essere acceso, ma il bordo deve lasciar vedere la nube:
  // il gradiente non arriva mai opaco fino al margine.
  const stops = [...regioneLampo.matchAll(/g\.addColorStop\(([0-9.]+), "rgba\([^)]*?," \+ \(([0-9.]+) \* a\)/g)]
    .map((m) => ({ dove: +m[1], alfa: +m[2] }));
  assert.ok(stops.length >= 3, 'il gradiente della nube ha troppe poche tappe');
  const fuori = stops.filter((t) => t.dove >= 0.6);
  assert.ok(fuori.length && fuori.every((t) => t.alfa <= 0.12),
    'il bordo dell\'alone e\' ancora opaco: copre la nube invece di sfumarci sopra');
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

prova('la vibrazione alla scarica: un colpo breve, e non un ronzio', () => {
  // Stesso evento del tick raccontato a un altro senso, e quindi le stesse
  // guardie -- piu' una in piu', perche' un motore di vibrazione ha una sua
  // inerzia: colpi piu' fitti del suo tempo di avvio si fondono in un ronzio
  // continuo, che e' peggio del silenzio, e intanto consumano batteria.
  const gap = html.match(/const STRIKE_HAPTIC_MIN_GAP_MS = (\d+);/);
  assert.ok(gap, 'manca il limite di frequenza della vibrazione');
  const gapTick = html.match(/const STRIKE_SOUND_MIN_GAP_MS = (\d+);/);
  assert.ok(Number(gap[1]) > Number(gapTick[1]),
    'la vibrazione puo\' ripetersi fitta quanto il tick (' + gap[1] + ' ms): diventa un ronzio');
  assert.ok(Number(gap[1]) >= 300,
    'la vibrazione si ripete ogni ' + gap[1] + ' ms: troppo per un motore vero');

  // La si esegue davvero, invece di leggerla soltanto. Lo stato che conta
  // (strikeLastHaptic, l'orologio) vive DENTRO il modulo costruito qui:
  // passarlo come parametro lo renderebbe immutabile fra una chiamata e
  // l'altra, e il limite di frequenza -- che e' proprio cio' che si vuole
  // verificare -- non si potrebbe esercitare affatto.
  const banco = (opzioni) => {
    const colpi = [];
    const stato = { ora: 100000 };
    const sorgente = [
      html.match(/const STRIKE_HAPTIC_MIN_GAP_MS = \d+;/)[0],
      'let strikeHapticOn = ' + (opzioni.spento ? 'false' : 'true') + ';',
      'let strikeLastHaptic = 0;',
      'const cloudTimeSelected = ' + (opzioni.passato ? 'Date.now()' : '0') + ';',
      'const document = { hidden: ' + Boolean(opzioni.nascosto) + ' };',
      'const performance = { now: () => stato.ora };',
      'const map = { getBounds: () => ({ getWest: () => 10, getEast: () => 20,'
        + ' getSouth: () => 35, getNorth: () => 45 }),'
        + ' getCenter: () => ({ lng: 15, lat: 40 }) };',
      implementazione('volumeScarica'),
      implementazione('vibraScarica'),
    ].join('\n\n');
    const finto = {
      vibrate: opzioni.senzaMotore
        ? undefined
        : (d) => { colpi.push(d); return true; },
    };
    const mod = new Function('navigator', 'stato', 'colpi',
      sorgente + '\nreturn {vibraScarica};')(finto, stato, colpi);
    return { colpi, stato, vibra: mod.vibraScarica };
  };

  // Al centro dello schermo il colpo e' piu' lungo che al bordo.
  const vicino = banco({});
  vicino.vibra(15, 40);
  vicino.stato.ora += 5000;
  vicino.vibra(19.8, 44.8);
  assert.equal(vicino.colpi.length, 2,
    'due scariche ben distanziate non danno due colpi: ' + vicino.colpi.join(','));
  assert.ok(vicino.colpi[0] > vicino.colpi[1],
    'il colpo non si accorcia allontanandosi dal centro: ' + vicino.colpi.join(','));
  assert.ok(vicino.colpi[0] >= 8 && vicino.colpi[0] <= 30,
    'il colpo dura ' + vicino.colpi[0] + ' ms: fuori dall\'intervallo utile');

  // Sei scariche fitte: ne deve passare UNA sola.
  const raffica = banco({});
  for (let i = 0; i < 6; i += 1) { raffica.vibra(15, 40); raffica.stato.ora += 60; }
  assert.equal(raffica.colpi.length, 1,
    'sotto raffica passano ' + raffica.colpi.length + ' colpi invece di uno: e\' un ronzio');

  // Fuori schermo, spenta, nel passato, e senza motore: mai un colpo.
  for (const caso of [
    { nome: 'spenta', opz: { spento: true }, dove: [15, 40] },
    { nome: 'scheda nascosta', opz: { nascosto: true }, dove: [15, 40] },
    { nome: 'scorrendo il passato', opz: { passato: true }, dove: [15, 40] },
    { nome: 'senza motore di vibrazione', opz: { senzaMotore: true }, dove: [15, 40] },
    { nome: 'scarica fuori dallo schermo', opz: {}, dove: [40, 60] },
  ]) {
    const b = banco(caso.opz);
    b.vibra(caso.dove[0], caso.dove[1]);
    assert.equal(b.colpi.length, 0,
      'con "' + caso.nome + '" il telefono vibra lo stesso');
  }


  // Le guardie: fuori schermo, scheda nascosta, spento, e nel passato.
  const corpo = implementazione('vibraScarica');
  assert.match(corpo, /if \(!strikeHapticOn \|\| document\.hidden\) return;/,
    'la vibrazione parte anche spenta o con la scheda in secondo piano');
  assert.match(corpo, /if \(cloudTimeSelected\) return;/,
    'scorrendo il passato il telefono vibra per scariche di un\'ora fa');
  assert.match(corpo, /const forza = volumeScarica\(lon, lat\);\s*\n\s*if \(forza <= 0\) return;/,
    'vibra anche per scariche che non si vedono sullo schermo');
  assert.match(corpo, /if \(adesso - strikeLastHaptic < STRIKE_HAPTIC_MIN_GAP_MS\) return;/,
    'il limite di frequenza e\' dichiarato ma non applicato');
  assert.match(corpo, /try \{ navigator\.vibrate\(durata\); \} catch/,
    'un rifiuto del browser diventerebbe un errore invece di passare inosservato');

  // E si chiama dove si chiama il tick: stesso evento, stesso punto.
  const arrivo = implementazione('addLiveStrike');
  assert.match(arrivo, /suonaTick\([^\n]*\);\s*\n\s*vibraScarica\(lon, lat\);/,
    'la vibrazione non parte alla scarica, o parte in un punto diverso dal tick');
});

prova('dove non c\'e\' un motore di vibrazione, il comando non si mostra', () => {
  // Safari su iPhone non espone questa interfaccia, e un computer non ha
  // niente da far vibrare. Offrire un interruttore che non fa succedere
  // nulla e' peggio che non offrirlo.
  assert.match(html, /function vibrazioneDisponibile\(\) \{[\s\S]{0,200}typeof navigator\.vibrate === "function"/,
    'manca il controllo di esistenza del motore di vibrazione');
  assert.match(html, /vibraChip\.hidden = !showLiveLightning \|\| !vibrazioneDisponibile\(\);/,
    'il comando della vibrazione compare anche dove non puo\' funzionare');
  // E la scelta si ricorda, come quella del tick.
  assert.match(html, /const STRIKE_HAPTIC_KEY = "meteo\.vibrazioneScariche";/,
    'la scelta sulla vibrazione non ha una sua chiave di memoria');
  assert.match(html, /localStorage\.setItem\(STRIKE_HAPTIC_KEY, strikeHapticOn \? "1" : "0"\)/,
    'la scelta sulla vibrazione non viene ricordata');
  // Un colpo di conferma dentro il clic: e' l'unico momento in cui il
  // browser lo lascia partire di sicuro, e fa sentire com'e' fatto.
  const accensione = implementazione('setStrikeHaptic');
  assert.match(accensione, /navigator\.vibrate\(18\)/,
    'accendendo la vibrazione non si sente il colpo di conferma');
  assert.match(accensione, /navigator\.vibrate\(0\)/,
    'spegnendola non si ferma una vibrazione in corso');
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
