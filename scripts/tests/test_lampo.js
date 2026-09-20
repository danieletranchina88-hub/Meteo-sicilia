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
  assert.match(disegno[0], /if \(!strikeReducedMotion\(\) && eta < s\.durata \* 0\.8\s*\n\s*&& \(!showSatelliteClouds \|\| s\.cloudIlluminated\)\) continue;/,
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

prova('i diagnostici non diventano fotografie ma Cloud Type viene decodificato', () => {
  // I prodotti diagnostici non si leggono per luminanza come una fotografia.
  // Cloud Type e' pero' utile se i suoi tre contributi vengono interpretati
  // separatamente: quota, spessore ottico e fase.
  const maschera = implementazione('costruisciMascheraNube');
  assert.match(maschera, /if \(!product \|\| !\["scene", "grey"\]\.includes\(product\.mode\)\) return null;/,
    'un diagnostico viene ancora trattato come una fotografia');
  const volume = implementazione('costruisciVolumeNube');
  assert.match(volume, /decodificaCloudType\(tr,tg,tb\)/,
    'Cloud Type non viene decodificato canale per canale');

  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
  const decodifica = new Function('clamp', implementazione('decodificaCloudType')
    + '\nreturn decodificaCloudType;')(clamp);
  const base = decodifica(0.3, 0.25, 0.5);
  const spesso = decodifica(0.3, 0.9, 0.5);
  const alto = decodifica(0.9, 0.25, 0.5);
  const ghiaccio = decodifica(0.3, 0.25, 0.1);
  assert.ok(spesso.thickness > base.thickness && spesso.cloud > base.cloud,
    'il verde non aumenta lo spessore ottico');
  assert.ok(alto.top > base.top && Math.abs(alto.thickness - base.thickness) < 1e-9,
    'il rosso non controlla separatamente la quota');
  assert.ok(ghiaccio.phase < base.phase && Math.abs(ghiaccio.top - base.top) < 1e-9,
    'il blu non resta un contributo separato di fase');
});

prova('sui compositi a colore, terra e luci non contano come nube', () => {
  // Un deserto, un tetto assolato o le luci di una citta' possono essere
  // chiari quanto una sommita' di cumulo -- ma sono COLORATI, e una nube
  // no. Sul grigio dell'infrarosso non c'e' colore da giudicare: il filtro
  // vale solo sui compositi "scene".
  const maschera = implementazione('costruisciMascheraNube');
  assert.match(maschera,
    /const saturation = \(Math\.max\(r,g,b\) - Math\.min\(r,g,b\)\) \/ Math\.max\(1,r,g,b\);/,
    'manca il calcolo della saturazione');
  assert.match(maschera,
    /const neutral = product\.mode === "scene"\s*\n\s*\? 1 - clamp\(\(saturation - 0\.12\) \/ 0\.5, 0, 1\)\s*\n\s*: 1;/,
    'la saturazione non penalizza piu\' la nuvolosita\' sui compositi a colore');
  assert.match(maschera, /const amount = level \* level \* \(3 - 2 \* level\) \* neutral \* alpha;/,
    'il fattore neutro non entra piu\' nel calcolo della nuvolosita\'');
});

prova('lo spessore e la quota regolano la scala fisica del bagliore', () => {
  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
  const raggio = new Function('clamp', implementazione('raggioVolumeKm')
    + '\nreturn raggioVolumeKm;')(clamp);
  const sottile = raggio({ thickness: 0.04, top: 0.2, phase: 0.7 });
  const profonda = raggio({ thickness: 1, top: 1, phase: 0.1 });
  assert.ok(profonda > sottile + 15,
    'una nube profonda non diffonde sensibilmente piu\' di una sottile');
  assert.ok(sottile >= 10 && profonda <= 42,
    'il raggio esce dai limiti geografici dichiarati');
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

prova('il volume usa Cloud Type e IR dello stesso fotogramma satellitare', () => {
  const prepara = implementazione('preparaVolumeNube');
  assert.match(prepara, /if \(!showLiveLightning \|\| !showSatelliteClouds \|\| cloudTimeSelected\) return;/,
    'il volume resta attivo senza satellite o su un fotogramma storico');
  assert.match(prepara, /caricaRasterVolume\(CLOUD_PRODUCTS\.cloudtype, box, size, slot/,
    'non viene caricato Tipo di nube');
  assert.match(prepara, /caricaRasterVolume\(CLOUD_PRODUCTS\.ir105, box, size, slot/,
    'non viene caricato IR10.5');
  assert.match(prepara, /chiaveVolumeNube\(slot, box, size\)/,
    'i diagnostici non sono vincolati allo stesso istante e riquadro');
  assert.match(prepara, /volumeFrames\.size > VOLUME_CACHE_LIMIT/,
    'la cache del volume cresce senza limite');
  assert.match(prepara, /aggiornaTextureVolume\(renderer,(?:mascheraNube|volume)\)/,
    'la texture 3D viene caricata soltanto quando arriva il primo lampo');
  // screen non supera mai il bianco: sopra una nube diurna gia' chiara un
  // lampo cosi' composito sparirebbe invece di essere visibile. Il
  // compositing normale, con l'alfa quasi opaco del gradiente, e' l'unico
  // modo per restare visibile anche di giorno.
  assert.doesNotMatch(html, /#strike-light-canvas \{[\s\S]{0,150}mix-blend-mode/,
    'la luce torna a fondersi con "screen": di giorno sparirebbe sulle nubi chiare');
});

prova('un fulmine senza satellite illumina comunque la nube', () => {
  // Prima della riscrittura per il satellite, disegnaNubeIlluminata aveva un
  // ripiego tondo per quando manca la maschera, la mappa e' inclinata o non
  // c'e' il satellite: la riscrittura lo aveva tolto, e sulla mappa senza
  // satellite un fulmine non accendeva piu' nessuna nube -- la chiamata in
  // drawLiveStrikes restava, ma proiezioneNube torna sempre nullo senza
  // satellite e la funzione usciva subito con false.
  const ritaglio = implementazione('disegnaNubeIlluminata');
  assert.match(ritaglio, /if\(!projection \|\| inclinata \|\| !scarica\) \{/,
    'manca il ramo di ripiego senza satellite, mappa inclinata o scarica');
  assert.match(ritaglio, /createRadialGradient\(punto\.x,punto\.y,0,punto\.x,punto\.y,raggio\)/,
    'il ripiego non disegna piu\' un bagliore tondo');
  assert.match(ritaglio, /return true;\s*\n\s*\}\s*\n\s*const component=componenteNube/,
    'il ripiego tondo non restituisce successo prima del ritaglio sul satellite');
  assert.doesNotMatch(ritaglio, /tondo\(/,
    'e\' ricomparso un disco disegnato da una funzione a parte invece che inline');
  const disegno = implementazione('drawLiveStrikes');
  assert.match(disegno, /if \(nube > 0\.012\) \{[\s\S]*?disegnaNubeIlluminata\(context, punto, largo, nube\);/,
    'la vista senza satellite non chiama piu\' il bagliore della nube');
});

prova('il bagliore satellitare abbaglia di piu\' e la sommita\' rivelata ha piu\' rilievo', () => {
  const inizializza = implementazione('inizializzaVolumeRenderer');
  assert.match(inizializza, /float alone=exp\(-0\.5\*pow\(distanza\/\(raggio\*2\.6\),1\.35\)\);/,
    'manca l\'alone largo che imita il bloom di una sorgente sovraesposta');
  assert.match(inizializza, /float campo=nucleo\+0\.55\*alone;/,
    'l\'alone largo non contribuisce piu\' al campo luminoso');
  assert.match(inizializza, /faccia=clamp\(0\.28\+0\.95\*dot\(normale,direzione\),0\.12,1\.35\)/,
    'il contrasto fra lato illuminato e lato in ombra non e\' aumentato: la sommita\' rivelata resta piatta');
  assert.match(inizializza, /radianza\+=trasmittanza\*densita\*luceLocale\*dz\*0\.72;/,
    'l\'accumulo di radianza non e\' stato aumentato');
  assert.match(inizializza, /bagliore=clamp\(1\.0-exp\(-radianza\*4\.2\),0\.0,1\.0\)/,
    'la curva di esposizione non e\' piu\' ripida: il lampo non e\' abbagliante');
  assert.match(inizializza, /alfa=clamp\(max\(bagliore\*0\.98,autoOmbra\),0\.0,0\.985\)/,
    'l\'alfa massimo del nucleo non e\' stato alzato verso l\'opaco');
});

prova('il bordo scuro attorno al nucleo da\' contrasto anche su una nube diurna gia\' chiara', () => {
  // Un nucleo bianco sopra una nube diurna gia\' quasi bianca e\' invisibile
  // qualunque sia la sua opacita\': serve un bordo piu\' scuro dello sfondo,
  // non solo un centro piu\' chiaro. Nucleo e ombra ora si compongono nello
  // stesso pixel invece di scegliere l\'uno o l\'altro.
  const inizializza = implementazione('inizializzaVolumeRenderer');
  assert.match(inizializza, /autoOmbra=clamp\(ombra\*\(1\.0-bagliore\)\*1\.15,0\.0,0\.34\)/,
    'il bordo d\'ombra non e\' stato rinforzato');
  assert.doesNotMatch(inizializza, /if\(bagliore>=autoOmbra\*1\.4\)/,
    'nucleo e ombra tornano a essere una scelta binaria invece di comporsi');
  assert.match(inizializza, /pesoChiaro=bagliore\/\(bagliore\+autoOmbra\+0\.0001\)/,
    'manca la miscela fra nucleo chiaro e bordo scuro');
  assert.match(inizializza, /colore=mix\(scuro,chiaro,pesoChiaro\)/,
    'il colore finale non fonde piu\' chiaro e scuro insieme');
  // Misurato in un vero contesto WebGL2 (Playwright/Chromium), su una nube
  // sintetica del tutto piatta (il caso peggiore: nessuna ombra di
  // orientamento possibile): senza l'anello il nucleo composto su sfondo
  // bianco puro si scostava di 2 unita' su 255 dal bianco, invisibile.
  // Con l'anello lo scostamento arriva a 48 unita' subito fuori dal
  // nucleo. La sola ombra di orientamento non basta: serve un bordo
  // legato alla sola distanza dalla scarica.
  assert.match(inizializza, /float u=distanza\/raggio;/,
    'manca la distanza normalizzata al raggio per l\'anello');
  assert.match(inizializza, /float anello=smoothstep\(0\.55,1\.1,u\)\*\(1\.0-smoothstep\(1\.1,2\.2,u\)\)/,
    'manca l\'anello scuro legato alla sola distanza dal centro del lampo');
  assert.match(inizializza, /ombraLocale\+=fonte\.w\*anello\*sagoma\*6\.0;/,
    'l\'anello e\' troppo debole per dare contrasto su una nube diurna piatta');
});

prova('il bagliore ritagliato sul satellite e il ragno sulla mappa nuda sono piu\' luminosi', () => {
  const ritaglio = implementazione('disegnaNubeIlluminata');
  assert.match(ritaglio, /rgba\(245,249,255,0\.92\)/,
    'la seconda tappa del gradiente satellitare non e\' piu\' luminosa');
  assert.match(ritaglio, /rgba\(238,246,255,0\.58\)/,
    'la terza tappa del gradiente satellitare non e\' piu\' luminosa');
  assert.match(ritaglio, /rgba\(233,242,255,0\.16\)/,
    'la quarta tappa del gradiente satellitare non e\' piu\' luminosa');
  assert.match(ritaglio, /"242,248,255",0\.42\)/,
    'i canali interni sommersi non sono piu\' luminosi');
  assert.match(ritaglio, /globalAlpha=Math\.min\(0\.97,nube\*0\.97\)/,
    'l\'alfa finale del bagliore ritagliato non e\' stato alzato');
  const disegno = implementazione('drawLiveStrikes');
  assert.match(disegno, /tinta: "236,208,252", alfa: 0\.40/,
    'il lobo esterno del ragno sulla mappa nuda non e\' piu\' luminoso');
  assert.match(disegno, /tinta: "250,238,255", alfa: 0\.72/,
    'il lobo medio del ragno sulla mappa nuda non e\' piu\' luminoso');
});

prova('il nuovo motore integra un volume 3D e non una sfumatura 2D', () => {
  const inizializza = implementazione('inizializzaVolumeRenderer');
  assert.match(inizializza, /getContext\("webgl2"/,
    'il volume non usa WebGL2');
  assert.match(inizializza, /const int STEPS=\$\{VOLUME_GPU_STEPS\}/,
    'lo shader non riceve il numero di strati verticali');
  assert.match(inizializza, /for\(int passo=0;passo<STEPS;passo\+\+\)/,
    'manca il ray marching lungo la colonna nuvolosa');
  assert.match(inizializza, /trasmittanza\*=exp\(-densita/,
    'manca l\'assorbimento Beer-Lambert lungo la vista');
  assert.match(inizializza, /trasmissioneFonte=exp\(-densita\*distanza/,
    'la luce non viene assorbita fra il canale e la sommita\'');
  assert.match(inizializza, /henyeyGreenstein/,
    'manca la diffusione anisotropa di acqua e ghiaccio');
  assert.match(inizializza, /topKm=2\.0\+12\.5\*materiale\.g/,
    'la quota satellitare non determina la sommita\' del volume');
  assert.match(inizializza, /profondita=mix\(0\.7,7\.0,pow\(materiale\.b,0\.72\)\)/,
    'lo spessore ottico non determina la profondita\' ricostruita');
});

prova('il volume GPU riceve i quattro vincoli satellitari separati', () => {
  const texture = implementazione('aggiornaTextureVolume');
  assert.match(texture, /data\[p\*4\]=mask\.cloud\[p\]/,
    'la presenza della nube non entra nella texture fisica');
  assert.match(texture, /data\[p\*4\+1\]=mask\.top\[p\]/,
    'la quota non entra nella texture fisica');
  assert.match(texture, /data\[p\*4\+2\]=mask\.thickness\[p\]/,
    'lo spessore non entra nella texture fisica');
  assert.match(texture, /data\[p\*4\+3\]=mask\.phase\?mask\.phase\[p\]:128/,
    'la fase acqua-ghiaccio non entra nella texture fisica');
  const renderer = implementazione('disegnaVolumeGpu');
  assert.match(renderer, /domainX=\(mask\.east-mask\.west\)\*111\.32/,
    'il ray marcher non conserva le dimensioni geografiche');
  assert.match(renderer, /strikeVolumeCanvas\.width\/window\.innerWidth/,
    'la risoluzione interna viene confusa con lo zoom della mappa');
  assert.match(renderer, /gl\.enable\(gl\.SCISSOR_TEST\);gl\.scissor/,
    'il ray marcher calcola ogni pixel dello schermo invece della sola cella');
});

prova('ogni lampo volumetrico resta nella propria massa nuvolosa', () => {
  const pack = implementazione('impacchettaComponentiVolume');
  assert.match(pack, /new Uint8Array\(mask\.width\*mask\.height\*4\)/,
    'manca l\'atlante delle quattro celle simultanee');
  assert.match(pack, /\*4\+e\]=alpha/,
    'le componenti connesse finiscono nello stesso canale');
  assert.match(pack, /if\(volumeComponentAtlas&&volumeComponentAtlas\.mask===mask/,
    'l\'atlante da oltre un megabyte viene riallocato a ogni fotogramma');
  const inizializza = implementazione('inizializzaVolumeRenderer');
  assert.match(inizializza, /float sagoma=componente\(meta\.w,parti\)/,
    'lo shader non seleziona la cella associata alla sorgente');
  assert.match(inizializza, /\*sagoma/,
    'la sagoma connessa non limita l\'emissione volumetrica');
  const render = implementazione('renderVolumeLightning');
  assert.match(render, /eventi\.length=Math\.min\(VOLUME_GPU_EVENTS,eventi\.length\)/,
    'manca il limite di lavoro per fotogramma');
  assert.match(render, /componenteVolumePerScarica\(mask,strike,radius\)/,
    'il ray marcher illumina anche nubi separate');
});

prova('WebGL degrada senza perdere i fulmini sui dispositivi incompatibili', () => {
  const disegno = implementazione('drawLiveStrikes');
  assert.match(disegno, /const volumeGpu = showSatelliteClouds\s*\n\s*\? renderVolumeLightning\(adesso\)/,
    'il motore volumetrico non viene attivato sul satellite');
  assert.match(disegno, /if \(!s\.cloudIlluminated\) \{[\s\S]*?disegnaNubeIlluminata\(/,
    'manca il fallback ottico per WebGL2 assente o saturo');
  assert.match(html, /const volumeRatio = mobile \? 0\.55 : 0\.8;/,
    'il ray marcher gira a piena risoluzione anche sui telefoni');
  assert.match(html, /volume 3D Cloud Type \+ IR/,
    'l\'interfaccia non dichiara il volume ricostruito');
});

prova('la luce resta nella cella nuvolosa connessa al fulmine', () => {
  const componente = implementazione('componenteNube');
  const ritaglio = implementazione('disegnaNubeIlluminata');
  assert.match(componente, /const stack=\[/,
    'non c\'e\' una ricerca della componente nuvolosa connessa');
  assert.match(componente, /if\(ex\*ex\+ey\*ey>1\) continue;/,
    'la componente nuvolosa non e\' limitata dalla scala fisica del lampo');
  assert.match(ritaglio, /const component=componenteNube\(mask,scarica\.lon,scarica\.lat,radiusKm\);/,
    'il bagliore ignora la nube che contiene la scarica');
  assert.match(ritaglio, /clip\.width=lato;clip\.height=lato/,
    'la maschera non copre l\'intero tile del bagliore');
  assert.match(ritaglio, /globalCompositeOperation="destination-in";\s*\n\s*g\.drawImage\(clip,0,0\)/,
    'il cielo sereno fuori dalla cella non viene azzerato');
  assert.match(ritaglio, /\(p\.ne\.x-p\.nw\.x\)\/mw/,
    'la maschera non segue la proiezione orizzontale del satellite');
  assert.match(ritaglio, /\(p\.sw\.y-p\.nw\.y\)\/mh/,
    'la maschera non segue la proiezione verticale del satellite');
  assert.doesNotMatch(ritaglio, /tondo\(/,
    'e\' ricomparso un disco artificiale sopra il satellite');
});

prova('la dimensione resta geografica e cambia correttamente con lo zoom', () => {
  const disegno = implementazione('drawLiveStrikes');
  assert.match(disegno, /const radiusKm = raggioVolumeKm\(sample\);/,
    'il raggio non nasce da quota e spessore della nube');
  assert.match(disegno, /s\.lon \+ radiusKm \/ \(111\.32/,
    'il raggio in chilometri non viene riproiettato sulla mappa');
  assert.match(disegno, /Math\.hypot\(edge\.x-punto\.x, edge\.y-punto\.y\)/,
    'la scala resta fissa in pixel durante lo zoom');
  // Il minimo era 3 px: un lampo senza zoom, su una vista che inquadra
  // tutta l'Italia, si schiacciava a un punto invisibile.
  assert.match(disegno, /Math\.min\(220, Math\.max\(16,/,
    'il ripiego 2D torna a un minimo troppo piccolo per essere visto senza zoom');
});

prova('il lampo volumetrico non sparisce quando si e\' zoomati indietro', () => {
  const render = implementazione('renderVolumeLightning');
  assert.match(render, /pxPerKm=Math\.hypot\(projection\.ne\.x-projection\.nw\.x,\s*\n\s*projection\.ne\.y-projection\.nw\.y\)\/Math\.max\(0\.001,domainX\)/,
    'manca il calcolo dei pixel per chilometro alla scala attuale');
  assert.match(render, /raggioVisibile=raggioVisibileKm\(item\.radius,pxPerKm\)/,
    'il raggio inviato alla GPU non tiene conto dello zoom corrente');
  assert.match(render, /radius:raggioVisibile,phase:sample\.phase/,
    'la sorgente principale non usa il raggio corretto per lo zoom');
  assert.match(render, /radius:raggioVisibile\*0\.72,/,
    'le sorgenti sommerse dei rami non seguono lo stesso raggio corretto');
  const funzione = implementazione('raggioVisibileKm');
  assert.match(funzione, /Math\.max\(fisicoKm, minimo\)/,
    'il raggio fisico puo\' ancora restare sotto il minimo visibile sullo schermo');
  assert.match(funzione, /Math\.min\(VOLUME_GLOW_MAX_KM,/,
    'manca un tetto che eviti un lampo enorme quando si e\' zoomati molto indietro');
  const raggio = implementazione('componenteVolumePerScarica');
  assert.match(raggio, /componenteNube\(mask,scarica\.lon,scarica\.lat,radiusKm\)/,
    'la ricerca della nube connessa non usa piu\' il raggio fisico reale');
});

prova('sul satellite il lampo e\' bianco freddo e immerso, non una ragnatela viola', () => {
  const ritaglio = implementazione('disegnaNubeIlluminata');
  const colori = [...ritaglio.matchAll(/rgba\((\d+),(\d+),(\d+),/g)]
    .map((m) => [+m[1], +m[2], +m[3]]);
  assert.ok(colori.length >= 5, 'mancano le tappe cromatiche del bagliore');
  for (const colore of colori) {
    assert.ok(Math.min(...colore) >= 230,
      'il bagliore non e\' piu\' quasi bianco: ' + colore.join(','));
    assert.ok(Math.max(...colore) - Math.min(...colore) <= 25,
      'il bagliore e\' troppo saturo: ' + colore.join(','));
  }
  assert.match(ritaglio, /g\.filter="blur\(5px\)"/,
    'i canali interni sono di nuovo linee taglienti');
  const disegno = implementazione('drawLiveStrikes');
  assert.match(disegno, /if \(showSatelliteClouds\) \{[\s\S]*?disegnaNubeIlluminata\([\s\S]*?\n\s*continue;\s*\n\s*\}/,
    'sopra il satellite vengono ancora disegnati i canali esterni');
});

prova('il bagliore e\' breve, accessibile e riusa il rendering preparato', () => {
  const s = scarica(0.44);
  assert.ok(modulo.luceNube(s, s.ultimoColpo + 250) < 0.12,
    'un quarto di secondo dopo l\'ultimo colpo la nube e\' ancora accesa');
  assert.ok(STRIKE_BAGLIORE_ATTESO <= 420,
    'il bagliore dura ' + STRIKE_BAGLIORE_ATTESO + ' ms: resta sulla mappa');
  const ritaglio = implementazione('disegnaNubeIlluminata');
  assert.match(ritaglio, /if\(!scarica\.cloudLight \|\| scarica\.cloudLight\.key!==key/,
    'il tile volumetrico viene ricostruito a ogni fotogramma');
  assert.match(ritaglio, /strikeReducedMotion\(\)\) return false;/,
    'la preferenza di movimento ridotto non disattiva il flash');
  const disegno = implementazione('drawLiveStrikes');
  assert.doesNotMatch(disegno, /Math\.sin\([^\n]*eta[^\n]*\)/,
    'il simbolo continua a pulsare dopo il lampo');
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
