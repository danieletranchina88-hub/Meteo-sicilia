'use strict';
// Il decodificatore del flusso Blitzortung, provato su un frame VERO.
//
// Il servizio manda JSON compresso in LZW: i primi campi si leggono in
// chiaro, poi il testo "degrada" in caratteri con code point >= 256, che sono
// riferimenti al dizionario. E' un formato non documentato, dedotto guardando
// il flusso: un parser cosi' senza una prova sul dato autentico e' una
// scommessa, non un'implementazione.
//
// La fixture e' un frame catturato davvero dal servizio il 18/09/2026,
// salvato come sequenza di code point per non dipendere dalla codifica del
// file di prova.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const fixture = JSON.parse(
  fs.readFileSync(path.join(__dirname, 'fixtures_blitzortung.json'), 'utf8')
);

function implementazione(nome) {
  const re = new RegExp('      function ' + nome + '\\(');
  const inizio = html.search(re);
  assert.ok(inizio >= 0, 'manca ' + nome);
  return html.slice(inizio, html.indexOf('\n      }', inizio) + 8);
}

const decodifica = new Function(
  implementazione('decodeBlitzortung') + '\nreturn decodeBlitzortung;'
)();

let ok = true;
function prova(nome, fn) {
  try { fn(); console.log('PASS ' + nome); }
  catch (error) { ok = false; console.log('FALLITO ' + nome + ': ' + error.message); }
}

prova('un frame reale del servizio diventa JSON valido', () => {
  const grezzo = fixture.codepoints.map((c) => String.fromCodePoint(c)).join('');
  const chiaro = decodifica(grezzo);
  const scarica = JSON.parse(chiaro);
  assert.equal(scarica.time, fixture.atteso.time, 'orario della scarica sbagliato');
  assert.equal(scarica.lat, fixture.atteso.lat, 'latitudine sbagliata');
  assert.equal(scarica.lon, fixture.atteso.lon, 'longitudine sbagliata');
  assert.equal(scarica.sig.length, fixture.atteso.stazioni,
    'numero di stazioni rilevatrici sbagliato');
  // Il frame compresso e' piu' corto del JSON: se cosi' non fosse, vorrebbe
  // dire che non stiamo decomprimendo niente.
  assert.ok(chiaro.length > grezzo.length,
    'il testo decodificato non e\' piu\' lungo del compresso');
});

prova('il tempo arriva in nanosecondi e va portato a millisecondi', () => {
  const grezzo = fixture.codepoints.map((c) => String.fromCodePoint(c)).join('');
  const scarica = JSON.parse(decodifica(grezzo));
  const ms = scarica.time / 1e6;
  // Deve cadere in un intervallo plausibile di date, non nel 1970 o nel 3000:
  // e' il controllo che smaschera un fattore di conversione sbagliato.
  const anno = new Date(ms).getUTCFullYear();
  assert.ok(anno >= 2024 && anno <= 2100,
    'la conversione del tempo produce l\'anno ' + anno);
  assert.match(html, /function tempoScaricaMillis\([\s\S]*?numero \/ 1e6/,
    'il sito non converte piu\' i nanosecondi in millisecondi');
  assert.match(html, /tempoScaricaMillis\(payload && payload\.time\)/,
    'la rilevazione non usa la conversione del timestamp della sorgente');
});

prova('una stringa vuota non fa esplodere il decodificatore', () => {
  assert.equal(decodifica(''), '', 'stringa vuota non gestita');
});

prova('le scariche fuori dal dominio vengono scartate', () => {
  // La fixture e' una scarica sugli Stati Uniti: il flusso e' mondiale, e
  // senza filtro il sito accumulerebbe scariche che non mostrera' mai.
  assert.ok(fixture.atteso.lon < 3.0 || fixture.atteso.lon > 22.0,
    'la fixture non serve piu\' a provare il filtro del dominio');
  assert.match(html, /if \(lon < CLOUD_DOMAIN\.west \|\| lon > CLOUD_DOMAIN\.east\) return;/,
    'manca il filtro di longitudine sul dominio');
  assert.match(html, /if \(lat < CLOUD_DOMAIN\.south \|\| lat > CLOUD_DOMAIN\.north\) return;/,
    'manca il filtro di latitudine sul dominio');
});

prova('il flusso non ha passato, quindi sparisce scorrendo indietro', () => {
  const disegno = html.match(/function drawLiveStrikes\(\)[\s\S]*?\n {6}\}/);
  assert.ok(disegno, 'manca il disegno delle scariche');
  assert.match(disegno[0], /if \(cloudTimeSelected\) return;/,
    'le scariche in diretta resterebbero disegnate su un istante passato, '
    + 'dove non sono mai state osservate');
});

prova('la connessione si chiude quando nessuno sta guardando', () => {
  // Una rete di volontari non deve reggere connessioni di schede in secondo
  // piano.
  assert.match(html, /stopStrikeAnimation\(\);\s*\n\s*blitzDisconnect\(\);/,
    'la connessione resta aperta con la scheda nascosta');
  assert.match(html, /blitzRetryDelay = Math\.min\(blitzRetryDelay \* 2, 30000\);/,
    'i tentativi di riconnessione non rallentano piu\'');
});

prova('la sottoscrizione al flusso viene inviata', () => {
  // Senza questo messaggio il server accetta la connessione e poi tace.
  assert.match(html, /socket\.send\('\{"a":111\}'\)/,
    'manca la richiesta di sottoscrizione al flusso');
});

console.log(ok ? 'ESITO: SUPERATO' : 'ESITO: DA RIVEDERE');
process.exitCode = ok ? 0 : 1;
