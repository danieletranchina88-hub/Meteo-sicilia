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

function tabella(nome) {
  const trovata = html.match(
    new RegExp('const ' + nome + ' = \\[[\\s\\S]*?\\n      \\];')
  );
  assert.ok(trovata, 'manca la tabella ' + nome);
  return trovata[0];
}

const codice = [
  'STRIKE_LIFE_MS', 'STRIKE_MAX', 'STRIKE_PING_MS',
  'STRIKE_ACTIVITY_TAU_MS', 'STRIKE_ACTIVITY_RADIUS_KM',
  'STRIKE_CELL_LAT', 'STRIKE_CELL_LON', 'STRIKE_ACTIVITY_MAX',
  'ATTIVITA_LAMPO_MS', 'ATTIVITA_SALITA_MS', 'ATTIVITA_PICCO_MS',
  'ATTIVITA_CODA_MS', 'ATTIVITA_CODA_PESO', 'ATTIVITA_COLPI_MAX',
  'SFONDO_NOTTE', 'SFONDO_GIORNO', 'CHIARORE_LATO'
].map(costante).join('\n')
  + '\n' + tabella('TINTE_NOTTE') + '\n' + tabella('TINTE_GIORNO')
  + '\nconst clamp = (v, a, b) => Math.max(a, Math.min(b, v));\n'
  + [
    'tempoScaricaMillis', 'supportoRete', 'stileEtaScarica',
    'aggregaAttivita', 'limiteSegni', 'luceDelColpo', 'tinteCella',
    'cellaIlluminata'
  ].map(implementazione).join('\n');

// cellaIlluminata legge prefersReducedMotion: si costruisce il modulo due
// volte, una per stato, invece di fingere una variabile modificabile.
function costruisci(motoRidotto) {
  return new Function('prefersReducedMotion', codice
    + '\nreturn {tempoScaricaMillis,supportoRete,stileEtaScarica,'
    + 'aggregaAttivita,limiteSegni,luceDelColpo,tinteCella,'
    + 'cellaIlluminata};')(motoRidotto);
}
const modulo = costruisci(false);
const moduloRidotto = costruisci(true);

const LAMPO = Number(costante('ATTIVITA_LAMPO_MS').match(/= (.+);/)[1]);
const COLPI_MAX = Number(costante('ATTIVITA_COLPI_MAX').match(/= (.+);/)[1]);
const RAGGIO_KM = Number(
  costante('STRIKE_ACTIVITY_RADIUS_KM').match(/= (.+);/)[1]
);

// Un raccoglitore di tappe: tinteCella parla a un oggetto gradiente, non a
// un numero, quindi per guardarci dentro basta fingere l'oggetto.
function tappe(forza, sfondo, modulo_) {
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
  }, forza, sfondo || 0);
  return lette;
}

// Una cella pronta da dare a cellaIlluminata, con i valori di comodo.
function cella(extra) {
  return Object.assign(
    { lat: 38, lon: 14, peso: 6, quante: 6, stazioni: 40,
      ultima: 0, arrivi: [], vicine: 1 },
    extra || {}
  );
}
// `daArrivo` e' quanto tempo fa e' ARRIVATA l'ultima rilevazione: e' il
// tempo su cui lampeggia il disegno. Si puo' passare un elenco, per i lampi
// a piu' colpi di ritorno.
function luce(c, daArrivo, modulo_) {
  const frame = 1000000;
  const scarti = Array.isArray(daArrivo) ? daArrivo : [daArrivo];
  return (modulo_ || modulo).cellaIlluminata(
    Object.assign({}, c, { arrivi: scarti.map(function (d) { return frame - d; }) }),
    frame,
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

prova('la geometria del canale e dichiarata un segno, non un dato', () => {
  // La regola di prima vietava del tutto di disegnare la forma di un
  // fulmine. Adesso la forma c'e', e la regola e' piu' stretta, non piu'
  // larga: puo' esserci PURCHE' NON CODIFICHI NIENTE. Quello che resta
  // vietato e' far dipendere la geometria da una grandezza misurata, che
  // significherebbe affermare una relazione che nessuno ha misurato.
  const generatore = implementazione('generaRagno')
    + implementazione('generaCanale') + implementazione('semeDaScarica');
  for (const grandezza of ['stazioni', 'supportoRete', 'peso', 'quante',
                           'intensita', 'energia', 'corrente']) {
    assert.doesNotMatch(generatore, new RegExp('\\b' + grandezza + '\\b'),
      'la forma del canale dipende da ' + grandezza
        + ': starebbe affermando una relazione non misurata');
  }
  // E niente casualita' vera: la forma viene dall'identita' della scarica.
  assert.doesNotMatch(generatore, /Math\.random/,
    'la forma del canale e casuale: cambierebbe a ogni fotogramma');
  assert.match(implementazione('semeDaScarica'), /s\.lat|s\.lon|s\.at/,
    'il seme non viene dalla scarica stessa');
  assert.match(implementazione('ragnoDellaScarica'), /semeDaScarica/,
    'il ragno non nasce dal seme della scarica');

  // Restano vietate le grandezze ottiche, che la rete a terra non ha.
  for (const nome of ['luceScarica', 'luceNube', 'luceBrace']) {
    assert.doesNotMatch(html, new RegExp('function ' + nome + '\\('),
      nome + ' ricostruisce ancora una grandezza non osservata');
  }
  // E restano vietati i colpi di ritorno INVENTATI: quelli veri arrivano
  // dalla rete, uno per rilevazione.
  assert.doesNotMatch(html, /\.colpi\b/,
    'una scarica conserva ancora colpi sintetici');
  assert.doesNotMatch(implementazione('cellaIlluminata'), /Math\.random|semeCasuale/,
    'il ritmo del lampo viene sorteggiato invece che osservato');
});

prova('la stessa scarica disegna sempre lo stesso ragno', () => {
  // Non e' un dettaglio estetico: una forma che cambia a ogni fotogramma
  // sfarfalla, e soprattutto suggerisce che stia raccontando qualcosa che
  // varia. Non varia niente: e' un segno, e un segno sta fermo.
  const fatto = new Function('s', implementazione('semeCasuale')
    + implementazione('semeDaScarica') + implementazione('generaCanale')
    + implementazione('generaRagno') + implementazione('ragnoDellaScarica')
    + costante('RAGNO_LIVELLI')
    + '\nreturn ragnoDellaScarica(s);');
  const base = { lat: 38.12, lon: 14.37, at: 1700000000123, stazioni: 7 };
  const uno = fatto(Object.assign({}, base));
  const due = fatto(Object.assign({}, base));
  assert.deepEqual(due, uno, 'la stessa scarica cambia forma');

  // ...e una scarica diversa un ragno diverso, o sarebbero tutti uguali.
  const altra = fatto(Object.assign({}, base, { lat: 38.44 }));
  assert.notDeepEqual(altra, uno, 'scariche diverse disegnano lo stesso ragno');

  // Il numero di stazioni non tocca la forma: e' la regola sopra, verificata
  // eseguendo invece che leggendo.
  const conPiuStazioni = fatto(Object.assign({}, base, { stazioni: 31 }));
  assert.deepEqual(conPiuStazioni, uno,
    'il numero di stazioni cambia la forma del canale');

  // La cache sta sulla scarica: si genera una volta sola.
  const s = Object.assign({}, base);
  const primo = fatto(s);
  assert.ok(s.ragno, 'il ragno non viene conservato sulla scarica');
  assert.equal(fatto(s), primo, 'il ragno viene rigenerato a ogni chiamata');
});

prova('il ragno vive quanto il lampo e non un millisecondo di piu', () => {
  const disegno = implementazione('disegnaRagno');
  // Si accende con la stessa curva che illumina la nube: stesso dato,
  // stesso ritmo.
  assert.match(disegno, /luceDelColpo\(eta\)/,
    'il ragno non segue la curva del colpo di ritorno');
  assert.match(disegno, /prefersReducedMotion/,
    'il ragno lampeggia anche con il moto ridotto');
  // E il giro di disegno non lo chiama nemmeno, fuori dalla finestra.
  const giro = implementazione('drawLiveStrikes');
  assert.match(giro, /daArrivo >= ATTIVITA_LAMPO_MS\) continue;/,
    'il giro di disegno prova a disegnare ragni gia spenti');
  assert.match(giro, /adessoFrame - s\.ricevuta/,
    'il ragno non e ancorato al tempo di arrivo');
});

prova('i ragni disegnati sono pochi e distinti', () => {
  const giro = implementazione('drawLiveStrikes');
  assert.match(giro, /ragniFatti\.length >= RAGNO_MAX/,
    'non c e un tetto al numero di ragni per fotogramma');
  assert.match(giro, /< separazioneRagni\)/,
    'due ragni possono cadere uno sopra l altro');
  assert.match(giro, /separazioneRagni = RAGNO_SEPARAZIONE_PX \* scala/,
    'la distanza fra i ragni non scala con lo zoom');
  const max = Number(costante('RAGNO_MAX').match(/= (.+);/)[1]);
  assert.ok(max >= 3 && max <= 12,
    'il tetto dei ragni e fuori scala: ' + max);
  // Il diradamento scorre dal fondo, dove stanno gli arrivi piu' recenti:
  // tenere i piu' vecchi mostrerebbe lampi gia' spenti al posto di quelli
  // che stanno accadendo.
  assert.match(giro, /for \(let i = liveStrikes\.length - 1; i >= 0; i -= 1\) \{\n\s*if \(ragniFatti/,
    'il diradamento dei ragni non parte dagli arrivi piu recenti');
});

prova('il bagliore del lampo e luce, non una misura', () => {
  const bagliore = implementazione('disegnaBagliore');
  // Segue il dato: si accende e si spegne con la curva dei colpi di
  // ritorno, che sono gli arrivi veri della rete.
  const giro = implementazione('drawLiveStrikes');
  assert.match(giro, /disegnaBagliore\(\s*\n?\s*context, punto, luceDelColpo\(daArrivo\)/,
    'il bagliore non segue la curva del colpo di ritorno');
  // E non codifica niente: taglia in pixel, nessuna grandezza misurata.
  assert.match(bagliore, /BAGLIORE_NUCLEO_PX|BAGLIORE_RAGGI_PX/,
    'il bagliore non usa una taglia in pixel');
  for (const grandezza of ['stazioni', 'supportoRete', 'peso', 'quante',
                           'raggioKmInPixel']) {
    assert.doesNotMatch(bagliore, new RegExp('\\b' + grandezza + '\\b'),
      'il bagliore dipende da ' + grandezza + ': si leggerebbe come una misura');
  }
  // Di giorno cambia tinta come tutto il resto, o su una sommita' al sole
  // un nucleo bianco non si staccherebbe.
  assert.match(bagliore, /SFONDO_NOTTE|SFONDO_GIORNO/,
    'il bagliore non guarda quanto e chiara la fotografia sotto');
  assert.match(bagliore, /prefersReducedMotion|luce <= 0\.02/,
    'il bagliore non ha una soglia sotto cui non si disegna');
});

prova('il glifo del fulmine resta unetichetta leggibile ovunque', () => {
  const segno = implementazione('disegnaPuntoScarica');
  assert.match(segno, /drawBoltGlyph/, 'il segno non e il simbolo del fulmine');
  // Colore e opacita' restano l'eta' del dato, come prima del simbolo.
  assert.match(segno, /stileEtaScarica\(eta\)/,
    'il segno non racconta piu leta del dato');
  // L'anello resta il supporto di rete: e' l'unica grandezza osservata che
  // il segno porta oltre a posizione ed eta'.
  assert.match(segno, /supportoRete\(scarica\.stazioni\)/,
    'il segno ha perso il supporto di rete');
  // E deve avere il suo contorno scuro: sopra una sommita' bianca di
  // giorno un simbolo chiaro sparirebbe.
  assert.match(segno, /rgba\(3,11,17,/,
    'il glifo non ha contorno: su fondo chiaro sparisce');
  assert.match(segno, /drawBoltGlyph\([^)]*true\)[\s\S]*drawBoltGlyph\([^)]*false\)/,
    'il contorno non viene disegnato prima del pieno');
  // Il diradamento deve seguire la taglia del simbolo, o i glifi si
  // incastrano: sono piu' alti del punto tondo che sostituiscono.
  const giro = implementazione('drawLiveStrikes');
  const sep = giro.match(/const separazione = ([0-9.]+) \* scala;/);
  assert.ok(sep && Number(sep[1]) >= 14,
    'i segni possono incastrarsi uno nellaltro');
});

prova('la taglia del ragno e in pixel, quella della densita in chilometri', () => {
  // E' la riga di confine fra il segno e il dato. Un ragno grande sullo
  // schermo non vuol dire un fulmine grande sul territorio; l'alone di
  // attivita' invece e' una grandezza geografica e resta in chilometri.
  assert.match(implementazione('disegnaRagno'), /RAGNO_RAGGIO_PX/,
    'il ragno non usa una taglia in pixel');
  assert.doesNotMatch(implementazione('disegnaRagno'), /raggioKmInPixel|_KM\b/,
    'il ragno ha una taglia geografica: si leggerebbe come footprint');
  assert.match(implementazione('cellaIlluminata'), /STRIKE_ACTIVITY_RADIUS_KM/,
    'lalone di attivita ha perso la sua taglia geografica');
  assert.match(html, /lampo e filamenti = segni grafici, /,
    'la legenda non dichiara che lampo e filamenti sono segni');
  assert.match(html, /non geometria misurata/,
    'la legenda non dice che la geometria non e misurata');
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
  assert.match(implementazione('cellaIlluminata'), /cella\.arrivi/,
    'il battito non viene dalle rilevazioni della rete');
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
  assert.match(html, /scarica rilevata · colore = età · anello = stazioni/,
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
  assert.deepEqual(prima.arrivi.slice().sort(), [],
    'una scarica senza tempo di arrivo entra comunque fra i colpi');
  assert.equal(prima.vicine, 2,
    'il conteggio dei vicini non usa il raggio del kernel');
  const lontana = celle.find((c) => c.lat < 37);
  assert.equal(lontana.vicine, 1, 'una cella isolata risulta affollata');
});

prova('il colpo di ritorno sale subito e si spegne entro la sua durata', () => {
  const salita = Number(costante('ATTIVITA_SALITA_MS').match(/= (.+);/)[1]);
  assert.equal(modulo.luceDelColpo(-1), 0, 'un colpo non ancora arrivato illumina');
  assert.equal(modulo.luceDelColpo(LAMPO), 0, 'il colpo non si spegne mai');
  assert.equal(modulo.luceDelColpo(LAMPO + 500), 0, 'il colpo rinasce dopo la fine');
  assert.ok(modulo.luceDelColpo(salita) > 0.97, 'il colpo non arriva a piena luce');
  assert.ok(modulo.luceDelColpo(salita / 2) < 0.6, 'il colpo non ha salita');
  // Decrescente da quando ha toccato il massimo.
  let precedente = Infinity;
  for (let t = salita; t <= LAMPO; t += 4) {
    const v = modulo.luceDelColpo(t);
    assert.ok(v <= precedente + 1e-12, 'il colpo risale dopo il massimo, a ' + t);
    precedente = v;
  }
});

prova('un colpo ha due tempi: e quello che lo fa sfarfallare', () => {
  // Un solo tempo di spegnimento non basta. Con una curva sola, tre colpi di
  // ritorno a quaranta millisecondi l'uno dall'altro -- la spaziatura vera
  // di un lampo -- si sommano in un'unica onda che si gonfia e cala: visto
  // e misurato. Con il picco corto piu' la corrente di coda, fra un colpo e
  // l'altro la luce scende davvero e si vede il battito.
  const picco = Number(costante('ATTIVITA_PICCO_MS').match(/= (.+);/)[1]);
  const coda = Number(costante('ATTIVITA_CODA_MS').match(/= (.+);/)[1]);
  assert.ok(coda > picco * 3, 'i due tempi di spegnimento non sono distinti');
  // Il picco e' gia' sceso sotto la meta' quando arriva il colpo dopo...
  assert.ok(modulo.luceDelColpo(40) < 0.45,
    'il colpo e ancora alto quando ne arriva un altro: si sommano in unonda');
  // ...ma a tempi lunghi resta molto piu' luce di quanta ne lascerebbe il
  // solo picco: quella e' la corrente di coda, ed e' il secondo tempo.
  assert.ok(modulo.luceDelColpo(150) > 5 * Math.exp(-150 / picco),
    'a tempi lunghi non resta coda: il tempo di spegnimento e uno solo');

  // La prova vera: la somma di tre colpi deve avere DUE minimi interni.
  const somma = (t) => modulo.luceDelColpo(t) + modulo.luceDelColpo(t - 40)
    + modulo.luceDelColpo(t - 90);
  let minimi = 0;
  for (let t = 6; t < 130; t += 2) {
    if (somma(t) < somma(t - 2) && somma(t) <= somma(t + 2)) minimi += 1;
  }
  assert.equal(minimi, 2, 'i tre colpi non sfarfallano: sono ' + minimi + ' cali');
});

prova('il lampo si somma sui colpi e si spegne da solo', () => {
  const c = cella();
  const uno = luce(c, 14);
  const tre = luce(c, [14, 54, 104]);
  const finito = luce(c, LAMPO + 1);
  assert.ok(uno.lampo > 0.4, 'un colpo solo non accende la cella');
  assert.ok(tre.lampo > uno.lampo,
    'tre colpi di ritorno non fanno piu luce di uno');
  assert.equal(finito.lampo, 0, 'il lampo non si spegne entro la sua durata');
  // Il lampo si distingue in DUE modi, e servono tutti e due: e' piu'
  // luminoso, e accende piu' nube. Il solo aumento di opacita' non
  // basterebbe -- oltre una certa opacita' il lilla ha gia' sostituito la
  // nube, e aggiungerne non si vede.
  assert.ok(uno.unione > finito.unione * 1.4,
    'il lampo non si distingue dal chiarore di fondo');
  assert.ok(uno.raggio > finito.raggio * 1.25,
    'il lampo non accende piu nube del chiarore di fondo');
});

prova('il lampo batte sullARRIVO, non sullistante osservato', () => {
  // La rete pubblica una scarica con qualche secondo di ritardo. Un lampo
  // disegnato nell'istante osservato sarebbe un lampo gia' finito quando
  // arriva, cioe' un lampo che nessuno vede mai.
  const impl = implementazione('cellaIlluminata');
  assert.doesNotMatch(impl, /adessoEpoca|cella\.ultima -|- cella\.ultima/,
    'il lampo usa listante osservato invece del tempo di arrivo');
  const aggrega = implementazione('aggregaAttivita');
  assert.match(aggrega, /cella\.arrivi\.push\(s\.ricevuta\)/,
    'la cella non raccoglie i tempi di arrivo');
  // Una scarica osservata cinque secondi fa ma arrivata adesso deve
  // lampeggiare: e' il caso normale, non l'eccezione.
  const vecchia = cella({ ultima: 1000000 - 5000 });
  assert.ok(luce(vecchia, 8).lampo > 0.4,
    'una scarica arrivata in ritardo non lampeggia mai');
});

prova('i colpi tenuti sono pochi e sono i piu recenti', () => {
  const adesso = 3_000_000;
  const scariche = [];
  for (let i = 0; i < COLPI_MAX + 6; i += 1) {
    scariche.push({ lat: 38, lon: 14, at: adesso - i * 10,
                    ricevuta: 50_000 - i * 10, stazioni: 5 });
  }
  const celle = modulo.aggregaAttivita(scariche, adesso);
  assert.equal(celle.length, 1, 'le scariche non sono finite nella stessa cella');
  assert.equal(celle[0].arrivi.length, COLPI_MAX,
    'la lista dei colpi non ha un tetto');
  assert.equal(Math.max.apply(null, celle[0].arrivi), 50_000,
    'il tetto ha buttato via il colpo piu recente');
  assert.ok(Math.min.apply(null, celle[0].arrivi) >= 50_000 - (COLPI_MAX - 1) * 10,
    'sono stati tenuti colpi vecchi al posto dei recenti');
});

prova('lunione delle celle vale quello che una cella sola voleva', () => {
  // E' la legge che regge tutto il campo. Le caselle sono da cinque
  // chilometri e l'alone da quindici: dentro un temporale ogni casella ne
  // ha nove o dieci addosso. Se ognuna disegnasse la forza che vuole,
  // quello che si vede sarebbe la loro UNIONE -- dieci veli da 0,13 fanno
  // 0,75 -- cioe' una patina che non racconta niente se non la fittezza.
  //
  // La regola: l'unione di N celle sovrapposte deve valere quello che una
  // cella sola voleva. Si verifica facendo davvero il conto dell'unione.
  const sovrapposte = (alfa, n) => 1 - Math.pow(1 - alfa, n);
  for (const vicine of [1, 2, 5, 10, 30]) {
    for (const peso of [1, 6, 20]) {
      const c = luce(cella({ peso: peso, vicine: vicine }), LAMPO + 1);
      assert.ok(Math.abs(sovrapposte(c.forza, vicine) - c.unione) < 1e-9,
        'quello che si vede con ' + vicine + ' celle non e quello dichiarato: '
          + sovrapposte(c.forza, vicine).toFixed(4) + ' contro ' + c.unione.toFixed(4));
    }
  }
  // Un ammasso si vede PIU' di una cella sola, ma col logaritmo: dieci
  // caselle accese non fanno dieci volte la luce.
  const sola = luce(cella({ vicine: 1 }), LAMPO + 1).unione;
  const dieci = luce(cella({ vicine: 10 }), LAMPO + 1).unione;
  assert.ok(dieci > sola * 1.5, 'un ammasso non si vede piu di una cella sola');
  assert.ok(dieci < sola * 3, 'la luce cresce quasi col numero delle caselle');
  // ...e la singola cella dentro l'ammasso disegna molto meno, o la somma
  // scapperebbe.
  assert.ok(luce(cella({ vicine: 10 }), LAMPO + 1).forza < sola * 0.5,
    'la cella dentro un ammasso non si fa da parte');
  // Il lampo resta ben distinguibile dal chiarore anche nel fitto: era il
  // difetto della taratura precedente, che divideva solo il chiarore.
  const spenta = luce(cella({ vicine: 10 }), LAMPO + 1).unione;
  const accesa = luce(cella({ vicine: 10 }), 14).unione;
  assert.ok(accesa > spenta * 1.15,
    'dentro un temporale fitto il lampo non si distingue dal chiarore');
});

prova('fra un lampo e laltro la cella non inventa movimento', () => {
  // Il ritmo e' la sequenza delle scariche, non un'oscillazione nostra.
  // Una sinusoide qui dentro sarebbe movimento non osservato, e per giunta
  // invisibile: fra un arrivo e l'altro si ridisegna una volta al secondo.
  assert.doesNotMatch(implementazione('cellaIlluminata'), /Math\.sin|Math\.cos/,
    'la cella oscilla con un movimento non osservato');
  // E si verifica eseguendo: passati i colpi, la cella resta ferma
  // qualunque cosa faccia l'orologio del disegno.
  const c = cella({ peso: 6, vicine: 1 });
  const ferma = function (frame) {
    return modulo.cellaIlluminata(
      Object.assign({}, c, { arrivi: [] }), frame,
      function (km) { return km; }
    ).forza;
  };
  for (const frame of [1, 777, 12_345, 987_654]) {
    assert.equal(ferma(frame), ferma(0),
      'la luce cambia senza che sia cambiato nessun dato');
  }
  assert.equal(luce(c, LAMPO + 1).forza, ferma(0),
    'un colpo gia spento lascia comunque un segno');
});

prova('due celle identiche rendono luce identica, ovunque siano', () => {
  // Contro la tentazione di far dipendere il disegno dalla posizione per
  // "variare": se due celle hanno la stessa attivita' e la stessa storia,
  // devono avere la stessa luce, o il disegno racconta qualcosa che i dati
  // non dicono.
  // Il confronto si fa sia a riposo sia sul colpo: con il solo riposo, una
  // dipendenza dalla posizione nascosta dentro il colpo passava inosservata.
  for (const dallArrivo of [14, 60, LAMPO + 1]) {
    const a = luce(cella({ lat: 38.0, lon: 14.0 }), dallArrivo).forza;
    const b = luce(cella({ lat: 45.9, lon: 9.7 }), dallArrivo).forza;
    assert.equal(a, b, 'la posizione cambia la luce senza ragione fisica');
  }
  // ...mentre la storia deve cambiarla.
  assert.notEqual(luce(cella(), 14).forza, luce(cella(), LAMPO + 1).forza,
    'una scarica appena arrivata non cambia niente');
});

prova('piu attivita vuol dire piu luce e piu nube illuminata', () => {
  const debole = luce(cella({ peso: 1 }), LAMPO + 1);
  const forte = luce(cella({ peso: 20 }), LAMPO + 1);
  assert.ok(forte.intensita > debole.intensita, 'lintensita non segue il peso');
  assert.ok(forte.forza > debole.forza, 'la luce non segue lintensita');
  assert.ok(forte.raggio > debole.raggio,
    'una cella piu attiva non illumina piu nube');
  // ...ma con compressione logaritmica: venti volte le scariche non fanno
  // venti volte la luce, o la scala si schiaccia sul bianco. La firma del
  // logaritmo e' che RADDOPPIARE il peso aggiunge sempre lo stesso tanto,
  // qualunque sia il livello di partenza; con una scala lineare il secondo
  // raddoppio aggiungerebbe il doppio del primo.
  const gradino = (a, b) => luce(cella({ peso: b }), LAMPO + 1).intensita
    - luce(cella({ peso: a }), LAMPO + 1).intensita;
  // Due firme della compressione, prese ai due capi della scala.
  // In basso: tre scariche vive devono gia' valere buona parte della
  // luce, mentre una scala lineare le lascerebbe quasi al buio.
  const tre = luce(cella({ peso: 3 }), LAMPO + 1).intensita;
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
  const enorme = luce(cella({ peso: 400 }), LAMPO + 1);
  assert.ok(enorme.intensita <= 1, 'lintensita non ha un tetto');
  assert.ok(enorme.raggio < RAGGIO_KM * 3,
    'il raggio illuminato cresce senza limite');
});

prova('a moto ridotto la cella non lampeggia', () => {
  const c = cella();
  assert.equal(luce(c, [8, 48], moduloRidotto).lampo, 0,
    'il lampo batte anche con il moto ridotto');
  // La cella resta pero' VISIBILE: il moto ridotto toglie il lampeggio, non
  // il dato. Spegnerla sarebbe togliere informazione a chi ha gia' chiesto
  // meno movimento.
  assert.ok(luce(c, [8, 48], moduloRidotto).forza > 0.05,
    'con il moto ridotto la cella sparisce invece di smettere di lampeggiare');
  assert.ok(luce(c, 8).lampo > 0.4, 'senza moto ridotto il lampo non parte');
});

prova('la tavolozza notturna e lavanda misurata, non bianco ne azzurro', () => {
  const stop = tappe(1, 0);
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
  const piena = tappe(1, 0);
  const meta = tappe(0.5, 0);
  for (let i = 0; i < piena.length - 1; i += 1) {
    assert.ok(Math.abs(meta[i].alfa - piena[i].alfa / 2) < 0.002,
      'lopacita non e proporzionale alla forza');
  }
  assert.equal(piena[piena.length - 1].alfa, 0,
    'il bagliore finisce di netto invece di svanire');
});

prova('di giorno la stessa tinta si prende piu profonda', () => {
  // Di giorno la sommita' e' gia' bianca di sole: sommarci luce non cambia
  // niente, e la cella sparisce. L'unica cosa che una nube illuminata
  // lascia ancora vedere e' la tinta, quindi lo stesso viola si prende piu'
  // profondo -- non piu' luce che si somma, ma velo che tinge.
  const notte = tappe(1, 0);
  const giorno = tappe(1, 1);
  assert.equal(notte.length, giorno.length,
    'le due tavolozze non hanno le stesse tappe');

  // Su una sommita' gia' chiara quello che si vede e' la TINTA, non la
  // luminosita': schiarire un bianco non si nota, colorarlo si'. La prova
  // misura quindi la croma, non la distanza dal bianco -- misurare la
  // seconda diceva il contrario di quello che si vede, ed e' l'errore che
  // ha fatto perdere un giro a questa taratura.
  const croma = (t) => t.alfa * (Math.max(t.r, t.g, t.b) - Math.min(t.r, t.g, t.b));
  const cromaNotte = Math.max.apply(null, notte.map(croma));
  const cromaGiorno = Math.max.apply(null, giorno.map(croma));
  assert.ok(cromaGiorno > cromaNotte * 1.8,
    'di giorno la cella non tinge piu di quanto tingesse di notte');
  // La soglia assoluta e' bassa apposta. Una tavolozza diurna piu' carica
  // si vedeva di piu' a numeri (differenza in CIELAB 25 invece di 14), ma a
  // zoom stretto, dove un alone da quindici chilometri riempie lo schermo,
  // dipingeva una macchia viola compatta invece di illuminare: guardata, e
  // rifatta. Quattordici unita' di CIELAB restano il doppio delle sette che
  // darebbe la sola tavolozza notturna, e sono piu' che percepibili.
  assert.ok(cromaGiorno > 35,
    'nemmeno la tappa migliore tinge abbastanza una sommita al sole');
  // E non solo la tappa migliore: il controllo su un massimo solo lasciava
  // passare una fascia intermedia svuotata, che e' proprio dove il lampo si
  // legge -- il cuore e' piccolo, la fascia di mezzo e' quasi tutta l'area.
  for (const t of giorno.filter((t) => t.posizione < 0.7)) {
    assert.ok(croma(t) > 10,
      'una fascia diurna non tinge: posizione ' + t.posizione
        + ', croma ' + croma(t).toFixed(1));
  }

  // Resta la stessa famiglia di tinta: verde sempre il canale piu' basso.
  for (const t of giorno) {
    assert.ok(t.g < t.r && t.g < t.b,
      'la tavolozza diurna ha cambiato tinta: ' + [t.r, t.g, t.b].join(','));
  }
  // E di notte NON si usa quella diurna: una nube scura illuminata da un
  // viola profondo sembrerebbe colorata, non illuminata.
  const scure = notte.filter((t) => t.posizione < 0.4);
  for (const t of scure) {
    assert.ok(t.r > 200 && t.b > 200,
      'di notte il cuore del lampo non e luce: ' + [t.r, t.g, t.b].join(','));
  }
  // Il passaggio e' continuo e sta dove dicono le misure delle scene vere.
  const notteSoglia = Number(costante('SFONDO_NOTTE').match(/= (.+);/)[1]);
  const giornoSoglia = Number(costante('SFONDO_GIORNO').match(/= (.+);/)[1]);
  assert.ok(notteSoglia < giornoSoglia, 'le due soglie sono invertite');
  const mezzo = tappe(1, (notteSoglia + giornoSoglia) / 2);
  for (let i = 0; i < mezzo.length; i += 1) {
    const fra = (mezzo[i].r - notte[i].r) / ((giorno[i].r - notte[i].r) || 1);
    assert.ok(fra > 0.3 && fra < 0.7,
      'a meta strada la tavolozza salta invece di passare');
  }
  assert.deepEqual(tappe(1, notteSoglia - 0.2), notte,
    'sotto la soglia notturna non si usa la tavolozza notturna');
  assert.deepEqual(tappe(1, giornoSoglia + 0.2), giorno,
    'sopra la soglia diurna non si usa la tavolozza diurna');
});

prova('quale tavolozza si usa lo decide la fotografia sotto la cella', () => {
  // Non una scelta globale: sul terminatore meta' schermo e' di giorno e
  // meta' di notte, e ogni temporale deve prendere la tavolozza del posto
  // in cui sta.
  const impl = implementazione('sfondoSottoCella');
  assert.match(impl, /m\.chiarore/,
    'lo sfondo non viene dalla luminanza tenuta dalla maschera');
  assert.match(impl, /lon|lat/,
    'lo sfondo non viene letto nella posizione della cella');
  const disegnoCelle = implementazione('disegnaAttivita');
  assert.match(disegnoCelle, /sfondoSottoCella\(cella\.lon, cella\.lat\)/,
    'il disegno non chiede lo sfondo per ogni cella');
  // E deve arrivarci davvero fino alla tavolozza: calcolarlo e poi
  // scriverne un altro passava tutte le prove di lettura.
  assert.match(disegnoCelle, /accese\.push\([\s\S]*?sfondo: sfondo\b/,
    'lo sfondo calcolato non e quello che finisce nella cella accesa');
  assert.match(disegnoCelle, /tinteCella\(g, cella\.forza \* fattore, cella\.sfondo\)/,
    'la tavolozza non riceve lo sfondo della cella');
  assert.doesNotMatch(impl, /getImageData/,
    'lo sfondo rilegge i pixel a ogni fotogramma invece di usare il gia calcolato');
  // E la griglia deve restare grossolana: a piena risoluzione sarebbero
  // duecento kilobyte per fotogramma, moltiplicati per i fotogrammi in
  // cache. Serve a scegliere fra due tavolozze, non a disegnare.
  const lato = Number(costante('CHIARORE_LATO').match(/= (.+);/)[1]);
  const maschera = Number(costante('MASCHERA_LATO').match(/= (.+);/)[1]);
  assert.ok(lato <= maschera / 4,
    'la griglia della luminanza e grande quanto la maschera');
  assert.match(implementazione('costruisciMascheraNube'), /CHIARORE_LATO/,
    'la griglia della luminanza non viene ridotta');
  // La maschera deve davvero conservarla.
  assert.match(implementazione('costruisciMascheraNube'), /chiarore: chiarore/,
    'la maschera non conserva la luminanza della scena');

  // E si esegue: una maschera finta, meta' buia e meta' chiara.
  const leggi = new Function('mascheraNube',
    implementazione('sfondoSottoCella') + '\nreturn sfondoSottoCella;');
  const larghezza = 20, altezza = 10;
  const chiarore = new Uint8Array(larghezza * altezza);
  for (let y = 0; y < altezza; y += 1) {
    for (let x = 0; x < larghezza; x += 1) {
      chiarore[y * larghezza + x] = x < larghezza / 2 ? 20 : 230;
    }
  }
  const finta = leggi({ west: 10, east: 20, south: 30, north: 40,
                        chiarore: chiarore, larghezza: larghezza, altezza: altezza });
  assert.ok(finta(11, 35) < 0.15, 'la meta buia non risulta buia');
  assert.ok(finta(19, 35) > 0.85, 'la meta chiara non risulta chiara');
  assert.equal(finta(99, 35), 0, 'fuori dalla maschera lo sfondo non e neutro');
  assert.equal(leggi(null)(11, 35), 0, 'senza maschera lo sfondo non e neutro');
});

prova('il ridisegno veloce dura piu del lampo, o lo sfarfallio non si vede', () => {
  // Fra un arrivo e l'altro il renderer ridisegna una volta al secondo per
  // non consumare batteria; sale a sessanta fotogrammi solo per il tempo
  // dichiarato da STRIKE_PING_MS. Se il lampo durasse piu' di quella
  // finestra, la sua coda verrebbe campionata una volta al secondo, cioe'
  // il battito diventerebbe uno scatto.
  const ping = Number(costante('STRIKE_PING_MS').match(/= (.+);/)[1]);
  assert.ok(LAMPO < ping,
    'il lampo dura piu della finestra di ridisegno veloce: ' + LAMPO + ' contro ' + ping);
  const animazione = implementazione('animateStrikes');
  assert.match(animazione, /timestamp - s\.ricevuta < STRIKE_PING_MS/,
    'la finestra veloce non e piu ancorata al tempo di arrivo');
  // E il picco di un colpo deve essere risolvibile a sessanta fotogrammi:
  // piu' corto di un paio di fotogrammi e lo sfarfallio diventa rumore.
  const picco = Number(costante('ATTIVITA_PICCO_MS').match(/= (.+);/)[1]);
  assert.ok(picco > 16, 'il picco di un colpo e piu corto di un fotogramma');
});

prova('le celle si sovrappongono, solo il minimo garantito si somma', () => {
  // Sommare andava bene con la tavolozza quasi bianca su fondo nero. Con
  // quella diurna no: due veli viola sommati diventano bianchi, cioe'
  // proprio il colore su cui non si vedono.
  const disegno = implementazione('disegnaAttivita');
  const somme = disegno.match(/globalCompositeOperation = "lighter"/g) || [];
  assert.equal(somme.length, 1,
    'le celle si sommano ancora fra loro: sono ' + somme.length + ' passate additive');
  // L'unica additiva e' il minimo garantito, e deve venire DOPO il
  // ritaglio: sovrapporlo cancellerebbe la forma della nube con un disco.
  const coda = disegno.slice(disegno.lastIndexOf('globalCompositeOperation = "lighter"'));
  assert.match(coda, /tondo\(context, accese\[i\], 0\.2, 0\.68\)/,
    'la passata additiva non e il minimo garantito');
  assert.ok(disegno.indexOf('ritaglia(context') < disegno.lastIndexOf('lighter'),
    'il minimo garantito viene prima del ritaglio e ne cancella la forma');
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

prova('la sagoma e una forma, non un filtro sulla luce', () => {
  // La sagoma dice DOVE c'e' nube. Ma il bagliore viene moltiplicato per
  // lei, quindi senza guadagno finiva per dire anche QUANTA luce lasciar
  // passare: sulla nube media valeva un terzo, e due terzi della luce
  // sparivano senza che nessuno l'avesse deciso.
  const maschera = implementazione('costruisciMascheraNube');
  assert.match(maschera, /MASCHERA_GUADAGNO/,
    'la sagoma non ha guadagno: smorza la luce invece di darle forma');
  const guadagno = Number(costante('MASCHERA_GUADAGNO').match(/= (.+);/)[1]);
  assert.ok(guadagno > 1.5 && guadagno <= 4,
    'il guadagno della sagoma e fuori scala: ' + guadagno);
  // Deve restare una sagoma, non diventare una macchia piena: il cielo
  // sereno resta trasparente e il tetto resta uno.
  assert.match(maschera, /Math\.min\(1, MASCHERA_GUADAGNO/,
    'il guadagno puo portare la sagoma oltre lopacita piena');
  assert.match(maschera, /grezza \* grezza \* \(3 - 2 \* grezza\)/,
    'la sagoma ha perso la rampa morbida: il bordo della nube diventa netto');
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
