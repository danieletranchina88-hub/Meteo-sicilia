'use strict';
// La corsia rapida del workflow.
//
// Una modifica a un CSS faceva girare l'intera pipeline: 30 minuti per
// riscaricare 1,7 GB di GRIB e rifare l'analisi dei fronti, al termine dei
// quali venivano ripubblicati dati identici a quelli gia' in linea. Adesso
// un push che tocca solo l'interfaccia salta tutto questo e si innesta sul
// commit che sta gia' su gh-pages.
//
// La scorciatoia pero' introduce due modi nuovi di sbagliare, ed e' quello
// che queste prove sorvegliano:
//
//  1. le DUE pubblicazioni possono divergere. Aggiungendo un file statico
//     alla pubblicazione completa e dimenticandolo nella corsia rapida, ogni
//     modifica di sola interfaccia lascerebbe in linea la versione vecchia di
//     quel file, e nessuno se ne accorgerebbe finche' non gira un run dei dati;
//  2. il classificatore puo' mandare in corsia rapida qualcosa che aveva
//     bisogno del ricalcolo.
//
// Le prove girano senza dipendenze, perche' devono girare anche DENTRO la
// corsia rapida, dove non c'e' ne' pip ne' apt.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const testo = fs.readFileSync(path.join(root, '.github/workflows/update_meteo.yml'), 'utf8');

let ok = true;
// Alcune prove fanno girare davvero il classificatore, che e' asincrono:
// senza attenderlo un fallimento diventerebbe un rifiuto non gestito invece
// di un esito rosso leggibile, e la suite passerebbe per sbaglio.
const inCorso = [];
function prova(nome, fn) {
  const esito = (async () => {
    try { await fn(); console.log('PASS ' + nome); }
    catch (error) { ok = false; console.log('FALLITO ' + nome + ': ' + error.message); }
  })();
  inCorso.push(esito);
}

function blocco(da, a) {
  const inizio = testo.indexOf(da);
  assert.ok(inizio >= 0, 'non trovo il blocco che inizia con: ' + da);
  const fine = a ? testo.indexOf(a, inizio) : testo.length;
  return testo.slice(inizio, fine > inizio ? fine : testo.length);
}

// --- i due elenchi di file pubblicati ------------------------------------
const deploy = blocco('      - name: Deploy dati su branch gh-pages');
const rapida = blocco('      - name: Pubblica i file statici su gh-pages',
                      '\n  tests:');

function fileDelDeploy() {
  const trovati = new Set();
  for (const riga of deploy.split('\n')) {
    const m = riga.match(/^\s*cp (?:-r )?(.+?) "\$STAGE"\/?\w*$/);
    if (!m) continue;
    for (const nome of m[1].trim().split(/\s+/)) trovati.add(nome);
  }
  return trovati;
}

function fileDellaCorsiaRapida() {
  const elenco = rapida.match(/const percorsi = \[([\s\S]*?)\];/);
  assert.ok(elenco, "la corsia rapida non dichiara piu' un elenco di percorsi");
  const trovati = new Set();
  for (const m of elenco[1].matchAll(/'([^']+)'/g)) trovati.add(m[1]);
  return trovati;
}

prova('le due pubblicazioni mettono in linea gli stessi file statici', () => {
  const completa = fileDelDeploy();
  const veloce = fileDellaCorsiaRapida();
  // I dati meteo la corsia rapida non li tocca: li eredita dal commit su cui
  // si innesta, ed e' tutto il punto della scorciatoia.
  completa.delete('data_weather');
  // Il fondo cartografico lo copia come cartella; la corsia rapida la
  // percorre file per file, perche' l'API vuole blob singoli.
  const cartografia = completa.delete('data_base');
  assert.ok(cartografia, 'la pubblicazione completa non copia piu\' data_base');
  assert.match(rapida, /fs\.readdirSync\('data_base'\)/,
    'la corsia rapida non pubblica piu\' il fondo cartografico');

  const mancanti = [...completa].filter((f) => !veloce.has(f));
  const inPiu = [...veloce].filter((f) => !completa.has(f));
  assert.equal(mancanti.length, 0,
    'la corsia rapida non pubblica: ' + mancanti.join(', ')
    + ' -- una modifica a questi file resterebbe invisibile in linea fino al '
    + 'prossimo run dei dati');
  assert.equal(inPiu.length, 0,
    'la corsia rapida pubblica file che la pubblicazione completa non mette '
    + 'in linea: ' + inPiu.join(', '));
});

prova('ogni file che manda in corsia rapida viene poi pubblicato', () => {
  // Se il classificatore accetta un file che la corsia rapida non pubblica,
  // quel push risulterebbe "fatto" senza aver cambiato niente in linea.
  const classificatore = blocco('            const INTERFACCIA = new Set([',
                                '            function rapido');
  const dichiarati = new Set();
  for (const m of classificatore.matchAll(/'([^']+)'/g)) dichiarati.add(m[1]);
  const pubblicati = fileDellaCorsiaRapida();
  const orfani = [...dichiarati].filter((f) => !pubblicati.has(f));
  assert.equal(orfani.length, 0,
    'il classificatore accetta ma la corsia rapida non pubblica: ' + orfani.join(', '));
});

prova('le prove in Python non finiscono nella corsia rapida', () => {
  // La corsia rapida non installa eccodes, netcdf, metpy: una prova in
  // Python li' dentro fallirebbe per mancanza di ambiente, non per un difetto.
  assert.match(testo, /nome\.startsWith\('scripts\/tests\/'\) && nome\.endsWith\('\.js'\)/,
    'la corsia rapida accetta prove che non sa eseguire');
  assert.doesNotMatch(rapida, /python/,
    'la corsia rapida esegue Python senza averne installato le dipendenze');
});

prova('la corsia rapida esegue tutte le prove in JavaScript', () => {
  // Con il glob non se ne puo' dimenticare nessuna: e' voluto.
  assert.match(rapida.length ? blocco("      - name: Prove dell'interfaccia", '\n      - name: Pubblica') : '',
    /for prova in scripts\/tests\/test_\*\.js/,
    'la corsia rapida non esegue piu\' tutte le prove in JavaScript');
});

prova('i due elenchi fissi di prove non dimenticano nessun file', () => {
  // Qui invece gli elenchi sono scritti a mano, in due copie: senza questa
  // guardia una prova nuova gira solo nella corsia rapida.
  const suDisco = fs.readdirSync(path.join(root, 'scripts/tests'))
    .filter((f) => /^test_.*\.js$/.test(f));
  assert.ok(suDisco.length >= 8, 'mi aspettavo almeno otto prove in JavaScript');
  const copie = testo.split('      - name: Verifica nucleo scientifico dei fronti').slice(1);
  assert.equal(copie.length, 2, 'le copie dell\'elenco delle prove non sono piu\' due');
  copie.forEach((copia, i) => {
    const blocco = copia.slice(0, copia.indexOf('\n      - name:'));
    for (const prova of suDisco) {
      assert.ok(blocco.includes('node scripts/tests/' + prova),
        'la copia ' + (i + 1) + ' dell\'elenco non esegue ' + prova);
    }
  });
});

prova('la pipeline completa e la corsia rapida si escludono a vicenda', () => {
  assert.match(testo,
    /needs\.ambito\.outputs\.solo_interfaccia != 'true'/,
    'la pipeline completa gira anche quando ha gia\' pubblicato la corsia rapida');
  // Se il job che classifica si rompe, i dati devono aggiornarsi lo stesso.
  assert.match(testo, /!cancelled\(\) && github\.event_name != 'pull_request'/,
    'un guasto nella classificazione fermerebbe anche gli aggiornamenti dei dati');
  assert.match(testo,
    /if: github\.event_name == 'push' && needs\.ambito\.outputs\.solo_interfaccia == 'true'/,
    'la corsia rapida non e\' piu\' condizionata alla classificazione');
  // Il default prudente: 'ambito' produce una stringa, e qualunque valore
  // diverso da 'true' manda alla pipeline completa.
  assert.match(testo, /core\.setOutput\('solo_interfaccia', String\(esito\)\)/,
    'la classificazione non produce piu\' una stringa confrontabile');
});

prova('la corsia rapida non forza mai gh-pages', () => {
  // Forzare qui vorrebbe dire cancellare i dati che un run appena concluso
  // ha pubblicato, e lasciare in linea un sito senza previsioni.
  assert.doesNotMatch(rapida, /force:\s*true/,
    'la corsia rapida forza il ramo e puo\' cancellare i dati di un run');
  assert.match(rapida, /parents: \[padre\]/,
    'il commit della corsia rapida non si innesta piu\' sulla punta di gh-pages');
  assert.match(rapida, /v\.path === 'data_weather'/,
    'manca la guardia che impedisce di pubblicare un sito senza dati');
});

// --- il classificatore, fatto girare davvero -----------------------------
function classificatore() {
  const corpo = blocco('            const INTERFACCIA = new Set([',
                       "            const esito = await decidi();");
  const sorgente = corpo.split('\n').map((r) => r.replace(/^ {12}/, '')).join('\n');
  return new Function('github', 'context', 'core',
    sorgente + '\nreturn decidi;');
}

function esegui(evento, prima, file, stato) {
  const core = { info: () => {}, setOutput: () => {}, setFailed: () => {} };
  const context = {
    eventName: evento, sha: 'bbbb', repo: { owner: 'x', repo: 'y' },
    payload: { before: prima }
  };
  const github = { rest: { repos: { compareCommitsWithBasehead: async () => {
    if (file === 'errore') throw new Error('rete');
    return { data: { status: stato || 'ahead',
      files: file.map((f) => ({ filename: f })) } };
  } } } };
  return classificatore()(github, context, core)();
}

prova('un ritocco al CSS prende la corsia rapida', async () => {
  assert.equal(await esegui('push', 'aaaa', ['modern-ui.css']), true);
  assert.equal(await esegui('push', 'aaaa', ['index.html', 'modern-ui.js']), true);
  assert.equal(await esegui('push', 'aaaa',
    ['index.html', 'scripts/tests/test_lampo.js', 'data_base/coast.json']), true);
});

prova('tutto cio\' che cambia i dati passa dalla pipeline completa', async () => {
  const casi = [
    ['scripts/process_data.py', 'il produttore dei dati'],
    ['scripts/front_engine.py', 'il motore dei fronti'],
    ['requirements.txt', 'le dipendenze'],
    ['scripts/tests/test_front_engine.py', 'una prova in Python'],
    ['.github/workflows/update_meteo.yml', 'il workflow stesso'],
    ['scripts/build_local_downscaling.py', 'il downscaling']
  ];
  for (const [file, che] of casi) {
    assert.equal(await esegui('push', 'aaaa', ['index.html', file]), false,
      che + ' (' + file + ') finirebbe in corsia rapida');
  }
});

prova('nel dubbio si fa la pipeline completa', async () => {
  assert.equal(await esegui('schedule', undefined, ['index.html']), false,
    'un run programmato non deve saltare il ricalcolo');
  assert.equal(await esegui('workflow_dispatch', undefined, ['index.html']), false,
    'un run lanciato a mano non deve saltare il ricalcolo');
  assert.equal(await esegui('push', '0000000000000000000000000000000000000000',
    ['index.html']), false, 'un push senza base nota');
  assert.equal(await esegui('push', 'aaaa', 'errore'), false,
    'una chiamata fallita al confronto');
  assert.equal(await esegui('push', 'aaaa', []), false,
    'un confronto che non elenca file');
  assert.equal(await esegui('push', 'aaaa',
    Array.from({ length: 300 }, (_, i) => 'index.html'), 'ahead'), false,
    'un elenco troncato a 300 file');
  assert.equal(await esegui('push', 'aaaa', ['index.html'], 'diverged'), false,
    'un confronto fra rami divergenti');
  assert.equal(await esegui('push', 'aaaa', ['index.html'], 'behind'), false,
    'un confronto all\'indietro');
});

// --- il pubblicatore, fatto girare davvero -------------------------------
// La parte sottile e' il ciclo di ritentativi: un run dei dati puo' forzare
// gh-pages mentre la corsia rapida lavora, e il commit va rifatto sul nuovo
// genitore invece di essere imposto sopra i dati appena pubblicati.
function pubblicatore(finto) {
  const corpo = blocco("            const fs = require('fs');",
                       '\n  tests:');
  const sorgente = corpo.split('\n').map((r) => r.replace(/^ {12}/, '')).join('\n');
  // github-script avvolge il corpo in una funzione asincrona e la attende:
  // qui va restituita la promessa, o la prova finirebbe prima del codice
  // che deve misurare, e passerebbe sempre.
  return new Function('github', 'context', 'core', 'require',
    'return (async () => {\n' + sorgente + '\n})();')(
      finto.github, finto.context, finto.core, finto.require);
}

function ambienteFinto(opzioni) {
  const opts = opzioni || {};
  const registro = { commit: [], ref: [], fallita: '' };
  let puntaCorrente = opts.punta || 'c1';
  let rifiutiRimasti = opts.rifiuti || 0;
  const github = { rest: { git: {
    createBlob: async () => ({ data: { sha: 'blob' } }),
    getRef: async () => ({ data: { object: { sha: puntaCorrente } } }),
    getCommit: async ({ commit_sha }) => ({ data: { tree: { sha: 'albero-' + commit_sha } } }),
    getTree: async () => ({ data: { tree: opts.senzaDati
      ? [{ path: 'index.html' }] : [{ path: 'index.html' }, { path: 'data_weather' }] } }),
    createTree: async ({ base_tree }) => ({ data: {
      sha: opts.identico ? base_tree : 'albero-nuovo' } }),
    createCommit: async ({ parents, tree }) => {
      registro.commit.push({ padre: parents[0], albero: tree });
      return { data: { sha: 'nuovo-' + registro.commit.length } };
    },
    updateRef: async ({ sha }) => {
      if (rifiutiRimasti > 0) {
        rifiutiRimasti -= 1;
        // Il run dei dati ha spostato la punta sotto di noi.
        puntaCorrente = 'c2';
        throw new Error('Update is not a fast forward');
      }
      registro.ref.push(sha);
    }
  } } };
  return { registro, finto: {
    github, context: { repo: { owner: 'x', repo: 'y' } },
    core: { info: () => {}, setFailed: (m) => { registro.fallita = m; } },
    require: (nome) => nome === 'fs' ? {
      readFileSync: () => Buffer.from('contenuto'),
      readdirSync: () => ['coast.json', 'borders.json']
    } : require(nome)
  } };
}

prova('la pubblicazione normale fa un commit solo, innestato sulla punta', async () => {
  const a = ambienteFinto();
  await pubblicatore(a.finto);
  assert.equal(a.registro.commit.length, 1, 'commit creati: ' + a.registro.commit.length);
  assert.equal(a.registro.commit[0].padre, 'c1', 'il commit non si innesta sulla punta');
  assert.deepEqual(a.registro.ref, ['nuovo-1'], 'il ramo non punta al commit nuovo');
  assert.equal(a.registro.fallita, '');
});

prova('contenuto identico: nessun commit', async () => {
  // Ripubblicare file identici sporcherebbe la storia di gh-pages a ogni
  // push che non cambia niente in linea.
  const a = ambienteFinto({ identico: true });
  await pubblicatore(a.finto);
  assert.equal(a.registro.commit.length, 0, 'ha creato un commit vuoto');
  assert.equal(a.registro.ref.length, 0, 'ha spostato il ramo senza motivo');
});

prova('se un run dei dati sposta gh-pages, si rifa\' il commit sul nuovo padre', async () => {
  const a = ambienteFinto({ rifiuti: 1 });
  await pubblicatore(a.finto);
  assert.equal(a.registro.commit.length, 2, 'non ha ritentato');
  assert.equal(a.registro.commit[0].padre, 'c1');
  assert.equal(a.registro.commit[1].padre, 'c2',
    'il secondo tentativo non si innesta sulla punta nuova: imporrebbe '
    + 'l\'interfaccia sopra i dati appena pubblicati');
  assert.deepEqual(a.registro.ref, ['nuovo-2']);
  assert.equal(a.registro.fallita, '');
});

prova('un ramo senza dati non viene pubblicato', async () => {
  // Innestarsi su un gh-pages senza data_weather darebbe un sito di sola
  // interfaccia, senza previsioni, e sembrerebbe un successo.
  const a = ambienteFinto({ senzaDati: true });
  await pubblicatore(a.finto);
  assert.equal(a.registro.commit.length, 0, 'ha pubblicato un sito senza dati');
  assert.match(a.registro.fallita, /data_weather/,
    'non spiega perche\' si e\' fermato');
});

Promise.all(inCorso).then(() => {
  console.log(ok ? 'ESITO: SUPERATO' : 'ESITO: DA RIVEDERE');
  process.exitCode = ok ? 0 : 1;
});
