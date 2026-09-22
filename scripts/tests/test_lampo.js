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
  'STRIKE_CELL_LAT', 'STRIKE_CELL_LON', 'STRIKE_ACTIVITY_MAX',
  'ATTIVITA_RESPIRO_MS'
].map(costante).join('\n')
  + '\nconst clamp = (v, a, b) => Math.max(a, Math.min(b, v));\n'
  + [
    'tempoScaricaMillis', 'supportoRete', 'stileEtaScarica',
    'aggregaAttivita', 'limiteSegni', 'tinteCella', 'cellaIlluminata'
  ].map(implementazione).join('\n');

// cellaIlluminata legge prefersReducedMotion: si costruisce il modulo due
// volte, una per stato, invece di fingere una variabile modificabile.
function costruisci(motoRidotto) {
  return new Function('prefersReducedMotion', codice
    + '\nreturn {tempoScaricaMillis,supportoRete,stileEtaScarica,'
    + 'aggregaAttivita,limiteSegni,tinteCella,cellaIlluminata};')(motoRidotto);
}
const modulo = costruisci(false);
const moduloRidotto = costruisci(true);

const RESPIRO = Number(costante('ATTIVITA_RESPIRO_MS').match(/= (.+);/)[1]);
const RAGGIO_KM = Number(
  costante('STRIKE_ACTIVITY_RADIUS_KM').match(/= (.+);/)[1]
);

// Un raccoglitore di tappe: tinteCella parla a un oggetto gradiente, non a
// un numero, quindi per guardarci dentro basta fingere l'oggetto.
function tappe(forza, modulo_) {
  const lette = [];
  (modulo_ || modulo).tinteCella({
    addColorStop: function (posizione, colore) {
      const m = colore.match(
        /rgba\((\d+),(\d+),(\d+),([0-9.]+)\)/
      );
      assert.ok(m, 'tappa non leggibile: ' + colore);
      lette.push({ posizione: posizione, r: +m[1], g: +m[2], b: +m[3],
                   alfa: +m[4] });
    }
  }, forza);
  return lette;
}

// Una cella pronta da dare a cellaIlluminata, con i valori di comodo.
function cella(extra) {
  return Object.assign(
    { lat: 38, lon: 14, peso: 6, quante: 6, stazioni: 40, ultima: 0, vicine: 1 },
    extra || {}
  );
}
function luce(c, dallUltima, modulo_) {
  const adesso = 1000000;
  return (modulo_ || modulo).cellaIlluminata(
    Object.assign({}, c, { ultima: adesso - dallUltima }),
    adesso,
    function (km) { return km; }
  );
}

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

  // Il raggio geografico e' passato a cellaIlluminata, che e' dove il
  // disegno decide quanta nube illuminare.
  assert.match(implementazione('cellaIlluminata'), /STRIKE_ACTIVITY_RADIUS_KM/,
    'lalone non usa il raggio geografico dichiarato');
  assert.match(implementazione('disegnaAttivita'), /cellaIlluminata\(/,
    'il disegno non passa piu dalla decisione dichiarata');
  // La cella PULSA, e il battito e' un dato osservato: e' l'istante della
  // scarica piu' fresca che la rete ha mandato. Quello che resta vietato e'
  // inventare luce con un ritmo proprio, quindi l'unica oscillazione non
  // osservata deve essere una modulazione limitata, non un lampo.
  assert.match(implementazione('cellaIlluminata'), /cella\.ultima/,
    'il battito non viene dallistante della scarica osservata');
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

// ---- LA CELLA CHE RESPIRA ------------------------------------------

prova('la cella conosce la scarica piu fresca e i vicini che le stanno addosso', () => {
  const adesso = 2_000_000;
  const scariche = [
    { lat: 38.0, lon: 14.0, at: adesso - 50_000, stazioni: 6 },
    { lat: 38.0, lon: 14.0, at: adesso - 400, stazioni: 9 },
    // Una casella vicina, dentro il kernel: deve contarsi fra i vicini.
    { lat: 38.05, lon: 14.06, at: adesso - 3_000, stazioni: 7 },
    // Una lontanissima: non deve.
    { lat: 36.0, lon: 12.0, at: adesso - 3_000, stazioni: 7 }
  ];
  const celle = modulo.aggregaAttivita(scariche, adesso);
  const prima = celle.find((c) => Math.abs(c.lat - 38.0) < 0.02
    && Math.abs(c.lon - 14.0) < 0.02);
  assert.ok(prima, 'la cella piu attiva non e stata aggregata');
  assert.equal(prima.ultima, adesso - 400,
    'la cella non registra listante della scarica piu fresca');
  assert.equal(prima.vicine, 2,
    'il conteggio dei vicini non usa il raggio del kernel');
  const lontana = celle.find((c) => c.lat < 37);
  assert.equal(lontana.vicine, 1, 'una cella isolata risulta affollata');
});

prova('il respiro batte sulla scarica e si spegne da solo', () => {
  const c = cella();
  const colpo = luce(c, 0);
  const meta = luce(c, RESPIRO / 2);
  const dopo = luce(c, RESPIRO + 1);
  const molto = luce(c, 60_000);
  assert.ok(colpo.respiro > 0.98, 'il colpo non parte pieno');
  assert.ok(meta.respiro > 0 && meta.respiro < colpo.respiro,
    'il colpo non decade');
  assert.equal(dopo.respiro, 0, 'il colpo non si spegne entro la sua durata');
  assert.equal(molto.respiro, 0, 'una scarica vecchia respira ancora');
  assert.ok(colpo.forza > dopo.forza * 2,
    'il colpo non si distingue dal chiarore di fondo');
});

prova('il colpo non si divide fra i vicini, il chiarore di fondo si', () => {
  // E' la correzione misurata: dividendo anche il colpo, dentro un temporale
  // fitto (nove celle addosso) restava un terzo della luce, cioe' proprio
  // dove il lampo serve non si vedeva.
  //
  // Il peso e' basso apposta: con una cella molto attiva la somma arriva al
  // tetto di opacita' e il confronto misurerebbe il tetto, non la regola.
  const sola = cella({ peso: 2, vicine: 1 });
  const fitta = cella({ peso: 2, vicine: 9 });
  const fondoSola = luce(sola, 60_000).forza;
  const fondoFitta = luce(fitta, 60_000).forza;
  assert.ok(fondoFitta < fondoSola * 0.5,
    'il chiarore di fondo non si divide fra le celle che lo condividono');
  const colpoSola = luce(sola, 0).forza - fondoSola;
  const colpoFitta = luce(fitta, 0).forza - fondoFitta;
  assert.ok(colpoSola < 1 && luce(sola, 0).forza < 1,
    'il confronto sta misurando il tetto di opacita, non la regola');
  assert.ok(Math.abs(colpoSola - colpoFitta) < 1e-9,
    'il colpo viene diviso fra i vicini e sparisce dentro i temporali fitti');
});

prova('fra un colpo e laltro la cella non inventa movimento', () => {
  // Il respiro e' la sequenza delle scariche, non un'oscillazione nostra.
  // Una sinusoide qui dentro sarebbe movimento non osservato, e per giunta
  // invisibile: fra un arrivo e l'altro si ridisegna una volta al secondo.
  const impl = implementazione('cellaIlluminata');
  assert.doesNotMatch(impl, /Math\.sin|Math\.cos/,
    'la cella oscilla con un movimento non osservato');
  assert.doesNotMatch(impl, /adessoFrame|performance\.now/,
    'la luce della cella dipende dallorologio del disegno');
  // E si verifica anche eseguendo: l'unico tempo che conta e' quello
  // osservato, cioe' listante della scarica.
  const c = cella({ peso: 6, vicine: 1 });
  const a = luce(c, 60_000).forza;
  const b = luce(c, 61_000).forza;
  assert.equal(a, b, 'la luce cambia senza che sia cambiato nessun dato');
});

prova('due celle identiche rendono luce identica, ovunque siano', () => {
  // Contro la tentazione di far dipendere il disegno dalla posizione per
  // "variare": se due celle hanno la stessa attivita' e la stessa storia,
  // devono avere la stessa luce, o il disegno racconta qualcosa che i dati
  // non dicono.
  // Il confronto si fa sia a riposo sia sul colpo: con il solo riposo, una
  // dipendenza dalla posizione nascosta dentro il colpo passava inosservata.
  for (const dallUltima of [0, 200, 60_000]) {
    const a = luce(cella({ lat: 38.0, lon: 14.0 }), dallUltima).forza;
    const b = luce(cella({ lat: 45.9, lon: 9.7 }), dallUltima).forza;
    assert.equal(a, b, 'la posizione cambia la luce senza ragione fisica');
  }
  // ...mentre la storia deve cambiarla.
  assert.notEqual(luce(cella(), 0).forza, luce(cella(), 60_000).forza,
    'una scarica appena arrivata non cambia niente');
});

prova('piu attivita vuol dire piu luce e piu nube illuminata', () => {
  const debole = luce(cella({ peso: 1 }), 60_000);
  const forte = luce(cella({ peso: 20 }), 60_000);
  assert.ok(forte.intensita > debole.intensita, 'lintensita non segue il peso');
  assert.ok(forte.forza > debole.forza, 'la luce non segue lintensita');
  assert.ok(forte.raggio > debole.raggio,
    'una cella piu attiva non illumina piu nube');
  // ...ma con compressione logaritmica: venti volte le scariche non fanno
  // venti volte la luce, o la scala si schiaccia sul bianco. La firma del
  // logaritmo e' che RADDOPPIARE il peso aggiunge sempre lo stesso tanto,
  // qualunque sia il livello di partenza; con una scala lineare il secondo
  // raddoppio aggiungerebbe il doppio del primo.
  const gradino = (a, b) => luce(cella({ peso: b }), 60_000).intensita
    - luce(cella({ peso: a }), 60_000).intensita;
  // Due firme della compressione, prese ai due capi della scala.
  // In basso: tre scariche vive devono gia' valere buona parte della
  // luce, mentre una scala lineare le lascerebbe quasi al buio.
  const tre = luce(cella({ peso: 3 }), 60_000).intensita;
  assert.ok(tre > 0.3, 'la luce cresce linearmente con la densita');
  // In alto: ogni raddoppio del peso puo' aggiungere al massimo il
  // logaritmo di due, qualunque sia il livello di partenza. Una scala
  // lineare lo sfonda al primo raddoppio un po' serio.
  const TETTO = Math.log(2) / Math.log(28);
  for (const base of [3, 6, 12]) {
    const passo = gradino(base, base * 2);
    assert.ok(passo > 0.05,
      'raddoppiare il peso non muove lintensita a ' + base);
    assert.ok(passo <= TETTO + 1e-9,
      'la luce cresce linearmente con la densita');
  }
  const enorme = luce(cella({ peso: 400 }), 60_000);
  assert.ok(enorme.intensita <= 1, 'lintensita non ha un tetto');
  assert.ok(enorme.raggio < RAGGIO_KM * 3,
    'il raggio illuminato cresce senza limite');
});

prova('a moto ridotto la cella non lampeggia', () => {
  const c = cella();
  assert.equal(luce(c, 0, moduloRidotto).respiro, 0,
    'il colpo lampeggia anche con il moto ridotto');
  // La cella resta pero' VISIBILE: il moto ridotto toglie il lampeggio, non
  // il dato. Spegnerla sarebbe togliere informazione a chi ha gia' chiesto
  // meno movimento.
  assert.ok(luce(c, 0, moduloRidotto).forza > 0.05,
    'con il moto ridotto la cella sparisce invece di smettere di lampeggiare');
  assert.ok(luce(c, 0).respiro > 0.98,
    'senza moto ridotto il colpo non parte');
});

prova('la tavolozza e lavanda misurata, non bianco ne azzurro', () => {
  const stop = tappe(1);
  assert.ok(stop.length >= 5, 'la tavolozza ha troppe poche tappe');
  for (const t of stop) {
    // Il verde e' sempre il canale piu' basso: e' la cosa che la fotografia
    // dice, ed e' quello che distingue la lavanda dal grigio.
    assert.ok(t.g < t.r && t.g < t.b,
      'una tappa non e lavanda: ' + [t.r, t.g, t.b].join(','));
  }
  // Un bianco appena sporco soddisfa gia' "verde piu' basso": serve anche
  // che la tinta sia davvero satura, o il ritaglio illuminerebbe di bianco.
  // Il controllo va fatto su TUTTE le tappe intermedie, non su una sola:
  // con una sola, sbiancarne un'altra passava inosservato. Il cuore e'
  // escluso apposta -- il centro di un lampo e' bianco anche nella foto.
  const intermedie = stop.filter(
    (t) => t.posizione > 0.2 && t.posizione < 0.95
  );
  assert.ok(intermedie.length >= 3, 'la tavolozza non ha fasce intermedie');
  for (const t of intermedie) {
    const massimo = Math.max(t.r, t.g, t.b);
    const minimo = Math.min(t.r, t.g, t.b);
    assert.ok((massimo - minimo) / massimo > 0.08,
      'una fascia intermedia e praticamente bianca: '
        + [t.r, t.g, t.b].join(','));
  }
  // E si scalda mentre si spegne: nel cuore il blu supera il rosso, nella
  // fascia intermedia no. E' il contrario di quello che verrebbe da
  // disegnare, ed e' il motivo per cui la foto e stata misurata.
  assert.ok(stop[0].b > stop[0].r, 'il cuore non e violetto');
  const tiepida = stop.find((t) => t.posizione >= 0.5 && t.posizione < 0.75);
  assert.ok(tiepida && tiepida.b <= tiepida.r,
    'la coda non si scalda spegnendosi');
});

prova('la forza scala lopacita e lultima tappa sparisce', () => {
  const piena = tappe(1);
  const meta = tappe(0.5);
  for (let i = 0; i < piena.length - 1; i += 1) {
    assert.ok(Math.abs(meta[i].alfa - piena[i].alfa / 2) < 0.002,
      'lopacita non e proporzionale alla forza');
  }
  assert.equal(piena[piena.length - 1].alfa, 0,
    'il bagliore finisce di netto invece di svanire');
});

prova('il ritaglio sul contesto principale e protetto da chi disegna prima', () => {
  const disegno = implementazione('disegnaAttivita');
  const guardia = disegno.match(/const telaSgombra = [\s\S]*?;/);
  assert.ok(guardia, 'manca la guardia del ritaglio in posto');
  // destination-in cancella quello che trova: la scorciatoia vale solo se
  // su questa canvas non ha ancora disegnato nessun altro. La guardia deve
  // nominare OGNI prodotto che drawVectors disegna prima dei fulmini.
  const prima = implementazione('drawVectors')
    .split('drawLiveStrikes')[0];
  for (const nome of ['showVectors', 'showIsobars', 'showIsohypses',
                      'showIsotherms', 'showFronts']) {
    assert.match(guardia[0], new RegExp('!' + nome),
      'la guardia del ritaglio non esclude ' + nome);
  }
  assert.match(guardia[0], /weatherView === "satellite"/,
    'la guardia non si limita alla vista dove la canvas e sgombra');
  // I toponimi non hanno un interruttore: sono esclusi dalla vista stessa.
  assert.match(prima, /weatherView !== "satellite"[\s\S]*drawPlaceNames/,
    'i toponimi potrebbero disegnare prima dei fulmini in vista satellite');
  // E chi disegna dopo non va nella guardia: i punti delle scariche sono
  // nostri e vengono dopo il ritaglio.
  assert.doesNotMatch(guardia[0], /showLiveLightning/,
    'la guardia esclude un prodotto che disegna dopo il ritaglio');
});

prova('la sagoma delle nubi scarta le luci di citta per forma, non per colore', () => {
  const maschera = implementazione('costruisciMascheraNube');
  assert.match(maschera, /ruvidezza/,
    'la maschera non misura la ruvidezza locale');
  assert.match(maschera, /istogrammaRuvido/,
    'la soglia di ruvidezza non viene dallimmagine stessa');
  assert.doesNotMatch(maschera, /ruvidezza\[k\] > 0\.[0-9]+/,
    'la ruvidezza viene confrontata con una soglia fissa');
  // E deve entrare davvero nel risultato: calcolarla e non usarla passava
  // tutte le prove sopra.
  const opacita = maschera.match(/const nuvolosita = [\s\S]*?;/);
  assert.ok(opacita, 'manca il calcolo della nuvolosita');
  assert.match(opacita[0], /liscio/,
    'la ruvidezza viene calcolata ma non entra nella sagoma');
  assert.match(opacita[0], /neutro/,
    'la saturazione non entra piu nella sagoma');
  // La luminanza resta il segnale principale, e la sua soglia resta per
  // percentili: di notte le nubi di GeoColour sono grigio-azzurre scure.
  assert.match(maschera, /percentile\(0\.15\)/, 'manca il percentile del buio');
  assert.match(maschera, /percentile\(0\.9\)/, 'manca il percentile della nube piena');
  assert.match(maschera, /\["scene", "grey"\]\.includes\(product\.mode\)/,
    'la sagoma viene ricavata anche dai compositi diagnostici');
});

prova('la maschera pubblicata e sempre quella dellimmagine in mostra', () => {
  const pubblica = implementazione('publishSatelliteClouds');
  assert.match(pubblica, /mascheraNube = maschera;/,
    'la pubblicazione non aggiorna la sagoma');
  // Deve stare DENTRO update(), dopo il controllo del token: altrimenti una
  // richiesta vecchia e lenta lascia ai fulmini la nube di prima.
  const dentro = pubblica.slice(pubblica.indexOf('const update = function'));
  assert.match(dentro.slice(0, dentro.indexOf('};')), /mascheraNube = maschera;/,
    'la sagoma viene assegnata fuori dal controllo di attualita');
  assert.match(implementazione('rememberCloudFrame'), /maschera: maschera \|\| null/,
    'la cache non conserva la sagoma');
  assert.match(html, /mascheraNube = frame\.maschera \|\| null;/,
    'ripubblicando dalla cache i fulmini illuminano la nube sbagliata');
});

console.log(ok ? 'ESITO: SUPERATO' : 'ESITO: DA RIVEDERE');
process.exitCode = ok ? 0 : 1;
