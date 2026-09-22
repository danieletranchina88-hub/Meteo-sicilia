'use strict';
// Cima ottica: le nubi ricostruite come volume.
//
// Queste prove non giudicano se una nube e' bella. Verificano tre cose, che
// sono quelle su cui la pagina puo' mentire:
//
//  1. che il campo venga dal satellite e non da una soglia inventata -- in
//     particolare che lo ZERO sia LOCALE. Con un riferimento unico su un
//     dominio che va dal Sahara al Baltico, la Scandinavia notturna, che e'
//     piu' fredda di un cumulo mediterraneo, diventava un altopiano alto
//     dieci chilometri. E' successo davvero, e la prova 'il fondo di cielo
//     sereno segue la latitudine' e' li' perche' non succeda piu';
//  2. che quello che NON e' misurato sia dichiarato -- lo spessore, il
//     ribollire, l'esagerazione -- e che l'HUD lo dica;
//  3. che non si torni al terreno spostato. Le nubi disegnate come rilievo
//     di un terreno vengono fuori a punta: montagne di roccia, non nubi.
//     Se un giorno ricompare setTerrain in questa pagina, qui si spegne.
//
// Girano senza dipendenze, perche' devono girare anche nella corsia rapida.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'cima-ottica.html'), 'utf8');

const corpo = html.match(/<script>\n([\s\S]*?)\n    <\/script>/);
assert.ok(corpo, 'non trovo il corpo dello script di cima-ottica.html');

// La pagina si tira indietro da sola quando non c'e' il DOM: qui la si fa
// girare come una libreria e si prende quello che serve.
const esporta = [
  'DOMINIO', 'STRATI', 'QUOTA_SCALA_KM', 'QUOTA_GINOCCHIO_KM', 'QUOTA_OLTRE_KM',
  'SPESSORE_MINIMO_KM', 'SPESSORE_FRAZIONE', 'SPESSORE_MASSIMO_KM',
  'FONDO_BLOCCO_PX', 'FONDO_PENDENZA_MAX', 'FONDO_EROSIONI', 'NUBE_SOGLIA_GRIGIO',
  'LAMPO_QUOTA_CARICA_KM', 'LAMPO_SOTTO_LA_CIMA_MIN_KM', 'LAMPO_DIFFUSIONE_KM',
  'LAMPO_COLPI_MIN', 'LAMPO_COLPI_MAX', 'LAMPO_VELOCITA_KM_S', 'LAMPO_PESO_COPERTURA',
  'quotaDiCarica', 'sequenzaDiColpi', 'luceDellaSequenza',
  'LAMPO_RAGGIO_GRUPPO_KM', 'LAMPI_MAX', 'LAMPO_FRA_COLPI_MS', 'ESAGERAZIONE_INIZIALE', 'ESAGERAZIONE_GRANA',
  'RUMORE_LATO', 'SIGMA_PER_KM', 'SOLE_MINIMO_GRADI', 'LUCE_CONVENZIONALE',
  'limita', 'mercX', 'mercY', 'latDaMercY', 'unitaX', 'unitaY', 'sfoca',
  'fondoDiCieloSereno', 'mediana3', 'posizioneSole', 'slotPiuRecente',
  'riquadroMercatore', 'misuraRaster', 'indirizzo', 'campoDelleNubi',
  'tessituraDelCampo', 'tessituraDiRumore', 'lampiDaAccumulo',
  'riunisciPerVicinanza', 'luceDelLampo', 'cellaPiuViva', 'superficieSenzaNubi',
  'FRAMMENTO', 'VERTICE', 'inverti4', 'StratoVolume'
];
const modulo = new Function(corpo[1] + '\nreturn {' + esporta.join(', ') + '};')();

let ok = true;
function prova(nome, fn) {
  try { fn(); console.log('PASS ' + nome); }
  catch (errore) { ok = false; console.log('FALLITO ' + nome + ': ' + errore.message); }
}

// --- il dominio ----------------------------------------------------------
prova('il dominio copre Europa e Mediterraneo', () => {
  const d = modulo.DOMINIO;
  const dentro = (lon, lat) =>
    lon > d.ovest && lon < d.est && lat > d.sud && lat < d.nord;
  assert.ok(dentro(13.4, 37.5), 'la Sicilia e\' fuori dal dominio');
  assert.ok(dentro(-3.7, 40.4), 'Madrid e\' fuori dal dominio');
  assert.ok(dentro(18.1, 59.3), 'Stoccolma e\' fuori dal dominio');
  assert.ok(dentro(28.9, 41.0), 'Istanbul e\' fuori dal dominio');
  // Oltre i sessanta gradi MTG guarda troppo di sbieco: il limite e' voluto
  // e sta scritto nel codice, non va allargato per sbaglio.
  assert.ok(d.nord <= 64, 'il dominio sale dove MTG non guarda piu\' dritto');
});

prova('le sorgenti sono quelle dichiarate, con i loro slot', () => {
  assert.equal(modulo.STRATI.forma.layer, 'mtg_fd:ir105_hrfi');
  assert.equal(modulo.STRATI.pelle.layer, 'mtg_fd:rgb_geocolour');
  assert.equal(modulo.STRATI.lampi.layer, 'mtg_fd:li_afa');
  // FCI ogni dieci minuti, LI ogni cinque: sono le cadenze vere del servizio.
  assert.equal(modulo.STRATI.forma.slotMs, 10 * 60 * 1000);
  assert.equal(modulo.STRATI.lampi.slotMs, 5 * 60 * 1000);
  assert.ok(modulo.STRATI.lampi.ritardoMin < modulo.STRATI.forma.ritardoMin,
    'il Lightning Imager esce prima di FCI, non dopo');
});

prova('l\'indirizzo WMS chiede il riquadro giusto e lo slot', () => {
  const misura = modulo.misuraRaster(512);
  const slot = modulo.slotPiuRecente(modulo.STRATI.forma, Date.UTC(2026, 8, 22, 12, 0));
  const url = modulo.indirizzo(modulo.STRATI.forma, misura, slot, false);
  assert.ok(url.includes('srs=EPSG%3A3857') || url.includes('srs=EPSG:3857'),
    'il riquadro non e\' in Mercatore sferico');
  assert.ok(url.includes('time=' + slot.iso), 'manca lo slot richiesto');
  assert.ok(url.includes('width=512'), 'manca la larghezza');
  const senza = modulo.indirizzo(modulo.STRATI.forma, misura, slot, true);
  assert.ok(!senza.includes('time='),
    'la rete di sicurezza deve chiedere l\'ultima immagine senza tempo');
});

// --- geometria -----------------------------------------------------------
prova('Mercatore e unita' + '\' di MapLibre tornano indietro', () => {
  for (const lat of [-45, 0, 29, 45.5, 62]) {
    assert.ok(Math.abs(modulo.latDaMercY(modulo.mercY(lat)) - lat) < 1e-6,
      'mercY non e\' invertibile a ' + lat);
  }
  // unitaY e' la stessa proiezione in [0,1] con y verso sud.
  assert.ok(modulo.unitaY(62) < modulo.unitaY(29), 'la y unitaria non cresce verso sud');
  assert.ok(Math.abs(modulo.unitaX(0) - 0.5) < 1e-9);
});

// --- il sole -------------------------------------------------------------
prova('il sole sta dove deve stare', () => {
  const gradi = (r) => r * 180 / Math.PI;
  // Solstizio di giugno, mezzogiorno solare sul meridiano di Greenwich:
  // all'equatore il sole sta a 90 meno la declinazione, cioe' 66,6 gradi.
  const equatore = modulo.posizioneSole(Date.UTC(2026, 5, 21, 12, 0), 0, 0);
  assert.ok(Math.abs(gradi(equatore.elevazione) - 66.6) < 1.5,
    'elevazione al solstizio sbagliata: ' + gradi(equatore.elevazione).toFixed(1));
  // Allo stesso istante, al Circolo Polare il sole non tramonta.
  const polare = modulo.posizioneSole(Date.UTC(2026, 5, 21, 0, 0), 68, 20);
  assert.ok(gradi(polare.elevazione) > 0,
    'a mezzanotte al Circolo Polare in giugno il sole deve stare su');
  // Mezzanotte d'inverno in Italia: ben sotto l'orizzonte.
  const notte = modulo.posizioneSole(Date.UTC(2026, 0, 15, 0, 0), 41.9, 12.5);
  assert.ok(gradi(notte.elevazione) < -40,
    'a mezzanotte d\'inverno il sole non e\' sotto l\'orizzonte');
});

prova('sotto l\'orizzonte la luce diventa convenzionale e lo dice', () => {
  const strato = new modulo.StratoVolume();
  strato.sole = modulo.posizioneSole(Date.UTC(2026, 0, 15, 0, 0), 41.9, 12.5);
  const notturna = strato.luceDiScena();
  assert.equal(notturna.vera, false, 'di notte non si puo\' dire che illumini il sole');
  assert.ok(notturna.forza > 0,
    'un volume senza luce e\' nero: la convenzione serve, purche\' dichiarata');
  assert.equal(Math.round(notturna.elevazione * 180 / Math.PI),
    modulo.LUCE_CONVENZIONALE.elevazioneGradi);
  strato.sole = modulo.posizioneSole(Date.UTC(2026, 5, 21, 12, 0), 41.9, 12.5);
  assert.equal(strato.luceDiScena().vera, true,
    'con il sole alto si deve illuminare col sole vero');
});

// --- il campo ------------------------------------------------------------
// Un'immagine finta: il grigio cresce da 'fondo' e una macchia fredda in
// mezzo. Serve a vedere cosa fa campoDelleNubi senza chiamare EUMETSAT.
function immagine(larghezza, altezza, f) {
  const dati = new Uint8ClampedArray(larghezza * altezza * 4);
  for (let y = 0; y < altezza; y++) {
    for (let x = 0; x < larghezza; x++) {
      const g = Math.max(0, Math.min(255, Math.round(f(x, y) * 255)));
      const k = (y * larghezza + x) * 4;
      dati[k] = dati[k + 1] = dati[k + 2] = g;
      dati[k + 3] = 255;
    }
  }
  return dati;
}

prova('una macchia fredda diventa una nube alta, il fondo resta a terra', () => {
  const L = 256, A = 256;
  const dati = immagine(L, A, (x, y) => {
    const d = Math.hypot(x - 128, y - 128);
    return d < 40 ? 0.85 : 0.18;
  });
  const campo = modulo.campoDelleNubi(dati, L, A);
  const q = (x, y) => campo.quota[y * L + x];
  assert.ok(q(128, 128) > 8, 'la macchia fredda non sale: ' + q(128, 128).toFixed(2));
  assert.ok(q(20, 20) < 1.2, 'il fondo caldo non e\' a terra: ' + q(20, 20).toFixed(2));
  assert.ok(campo.copertura[128 * L + 128] > 0.9, 'la macchia non risulta coperta');
  assert.ok(campo.copertura[20 * L + 20] < 0.2, 'il fondo caldo risulta coperto');
});

prova('il fondo di cielo sereno segue la latitudine', () => {
  // LA REGRESSIONE IMPORTANTE. Nessuna nube: solo una superficie che si
  // raffredda andando a nord, come succede di notte fra il Sahara e il
  // Baltico. Con una soglia unica il nord diventava un altopiano; con il
  // fondo locale deve restare piatto come il sud.
  // Il gradiente e' due volte piu' ripido di quello vero: fra Sahara e
  // Baltico la superficie notturna cambia di circa un centesimo di grigio
  // ogni blocco, qui di due.
  const L = 512, A = 512;
  const dati = immagine(L, A, (x, y) => 0.62 - 0.30 * (y / A));
  const campo = modulo.campoDelleNubi(dati, L, A);
  let massimo = 0;
  for (let k = 0; k < campo.quota.length; k++) massimo = Math.max(massimo, campo.quota[k]);
  assert.ok(massimo < 2.5,
    'una superficie fredda senza nubi sale a ' + massimo.toFixed(1)
    + ' km: il fondo non e\' locale');
  // E il tetto alla pendenza dev\'essere piu\' largo del gradiente vero.
  assert.ok(modulo.FONDO_PENDENZA_MAX > 0.30 / (A / modulo.FONDO_BLOCCO_PX),
    'il tetto alla pendenza taglia anche i gradienti di superficie');
});

prova('una superficie mossa non diventa una velatura', () => {
  // Cielo limpido, ma la superficie non e' uniforme: mare, terra, una citta'
  // accesa. Senza la soglia, ognuna di queste differenze diventava un velo di
  // nube steso sull'Europa -- si vedeva a occhio nella resa a nadir.
  const L = 256, A = 256;
  const dati = immagine(L, A, (x, y) => {
    if (x > 150 && x < 190 && y > 90 && y < 130) return 0.30;   // terra piu' fredda
    if (x > 60 && x < 70 && y > 60 && y < 70) return 0.33;      // una citta'
    return 0.26 + 0.02 * Math.sin(x * 0.4) * Math.cos(y * 0.35);
  });
  const campo = modulo.campoDelleNubi(dati, L, A);
  let coperti = 0;
  for (let k = 0; k < campo.copertura.length; k++) if (campo.copertura[k] > 0.25) coperti++;
  const frazione = coperti / campo.copertura.length;
  assert.ok(frazione < 0.02,
    'il ' + (frazione * 100).toFixed(0) + '% di una scena limpida risulta coperto');
  assert.ok(modulo.NUBE_SOGLIA_GRIGIO > 0, 'manca la soglia sotto la quale non c\'e\' nube');
});

prova('il fondo locale non cancella una nube larga', () => {
  // L'altra faccia: erodendo troppo, una copertura estesa verrebbe presa per
  // superficie e sparirebbe. Meta' scena coperta deve restare coperta.
  const L = 256, A = 256;
  const dati = immagine(L, A, (x, y) => (y < A / 2 ? 0.86 : 0.16));
  const campo = modulo.campoDelleNubi(dati, L, A);
  assert.ok(campo.quota[40 * L + 128] > 7,
    'meta\' scena coperta viene presa per superficie: '
    + campo.quota[40 * L + 128].toFixed(2));
  // Il centro della copertura dista dal bordo piu' di un paio di blocchi:
  // senza abbastanza giri di erosione il chiaro di cielo non ci arriva.
  assert.ok(modulo.FONDO_EROSIONI >= 6,
    'l\'erosione non arriva piu\' al centro di una copertura larga');
});

prova('la scala si comprime invece di tagliare', () => {
  // Con un taglio secco tutte le sommita' piu' fredde finiscono alla stessa
  // quota e le celle diventano altipiani con la cima piatta.
  const L = 128, A = 128;
  const dati = immagine(L, A, (x, y) => {
    const d = Math.hypot(x - 64, y - 64);
    if (d < 10) return 1.0;
    if (d < 26) return 0.8;
    return 0.12;
  });
  const campo = modulo.campoDelleNubi(dati, L, A);
  const cuore = campo.quota[64 * L + 64];
  const corona = campo.quota[64 * L + 82];
  assert.ok(cuore > corona + 0.4,
    'il cuore non svetta sulla corona: ' + cuore.toFixed(2) + ' vs ' + corona.toFixed(2));
  assert.ok(cuore <= modulo.QUOTA_GINOCCHIO_KM + modulo.QUOTA_OLTRE_KM + 1e-6,
    'la compressione non tiene il fondoscala');
});

prova('la mediana toglie il sale e pepe senza spianare il bordo', () => {
  const L = 32, A = 32;
  const campo = new Float32Array(L * A);
  for (let y = 0; y < A; y++) {
    for (let x = 0; x < L; x++) campo[y * L + x] = x < 16 ? 0 : 10;
  }
  campo[8 * L + 8] = 99;   // il pixel isolato
  const fuori = modulo.mediana3(campo, L, A);
  assert.equal(fuori[8 * L + 8], 0, 'il pixel isolato sopravvive alla mediana');
  assert.equal(fuori[20 * L + 10], 0, 'la mediana sposta il lato freddo');
  assert.equal(fuori[20 * L + 22], 10, 'la mediana sposta il lato caldo');
});

// --- la texture del volume ------------------------------------------------
prova('lo spessore e\' una regola dichiarata e non sfonda la base', () => {
  const L = 8, A = 8;
  const campo = {
    larghezza: L, altezza: A,
    quota: new Float32Array(L * A), copertura: new Float32Array(L * A)
  };
  for (let k = 0; k < L * A; k++) {
    campo.quota[k] = (k % 16) * 1.1;
    campo.copertura[k] = 0.8;
  }
  const dati = modulo.tessituraDelCampo(campo);
  for (let k = 0; k < L * A; k++) {
    const cima = dati[k * 4] / 255 * modulo.QUOTA_SCALA_KM;
    const spessore = dati[k * 4 + 2] / 255 * modulo.QUOTA_SCALA_KM;
    assert.ok(spessore <= cima + 1e-6,
      'la base finisce sottoterra: spessore ' + spessore.toFixed(2)
      + ' su cima ' + cima.toFixed(2));
    assert.ok(spessore <= modulo.SPESSORE_MASSIMO_KM + 1e-6,
      'lo spessore supera il massimo dichiarato');
  }
  // La regola e' monotona: una nube piu' alta e' anche piu' profonda.
  assert.ok(modulo.SPESSORE_FRAZIONE > 0, 'lo spessore non dipende piu\' dalla cima');
});

prova('il fondoscala tiene dentro la scala della quota', () => {
  assert.ok(modulo.QUOTA_SCALA_KM >= modulo.QUOTA_GINOCCHIO_KM + modulo.QUOTA_OLTRE_KM,
    'le sommita\' compresse escono dal fondoscala della texture');
});

// --- il rumore ------------------------------------------------------------
prova('il rumore e\' periodico e sta in mezzo', () => {
  const lato = 16;
  const dati = modulo.tessituraDiRumore(lato);
  assert.equal(dati.length, lato * lato * lato);
  let somma = 0, minimo = 255, massimo = 0;
  for (const v of dati) { somma += v; minimo = Math.min(minimo, v); massimo = Math.max(massimo, v); }
  const media = somma / dati.length;
  assert.ok(media > 60 && media < 195, 'il rumore e\' sbilanciato: media ' + media.toFixed(0));
  assert.ok(massimo - minimo > 60, 'il rumore e\' piatto');
});

// --- i lampi --------------------------------------------------------------
function accumulo(larghezza, altezza, punti) {
  const dati = new Uint8ClampedArray(larghezza * altezza * 4);
  punti.forEach(([px, py, raggio]) => {
    for (let y = py - raggio; y <= py + raggio; y++) {
      for (let x = px - raggio; x <= px + raggio; x++) {
        if (x < 0 || y < 0 || x >= larghezza || y >= altezza) continue;
        if (Math.hypot(x - px, y - py) > raggio) continue;
        const k = (y * larghezza + x) * 4;
        dati[k] = 255; dati[k + 3] = 255;
      }
    }
  });
  return dati;
}

prova('dall\'accumulo escono le aree accese, non di piu\'', () => {
  const L = 200, A = 160;
  const dati = accumulo(L, A, [[40, 40, 5], [150, 110, 3]]);
  // Un pixel solo: rumore di ricampionamento, non un lampo.
  const solo = (100 * L + 100) * 4;
  dati[solo] = 255; dati[solo + 3] = 255;
  const trovati = modulo.lampiDaAccumulo(dati, L, A);
  assert.equal(trovati.length, 2, 'aree trovate: ' + trovati.length);
  const ordinati = trovati.slice().sort((a, b) => a.x - b.x);
  assert.ok(Math.abs(ordinati[0].x - 40) < 1.5 && Math.abs(ordinati[0].y - 40) < 1.5,
    'il baricentro della prima area e\' spostato');
  assert.ok(ordinati[0].area > ordinati[1].area, 'le aree non sono confrontabili');
  // Posizione geografica coerente col riquadro.
  const d = modulo.DOMINIO;
  trovati.forEach((l) => {
    assert.ok(l.lon > d.ovest && l.lon < d.est, 'longitudine fuori dominio');
    assert.ok(l.lat > d.sud && l.lat < d.nord, 'latitudine fuori dominio');
    assert.ok(l.ux >= 0 && l.ux <= 1 && l.uy >= 0 && l.uy <= 1,
      'le unita\' di MapLibre escono da [0,1]');
  });
});

prova('un accumulo vuoto non produce nessun lampo', () => {
  const L = 64, A = 64;
  assert.equal(modulo.lampiDaAccumulo(new Uint8ClampedArray(L * A * 4), L, A).length, 0,
    'senza dato si inventa un lampo');
});

prova('i lampi si riuniscono per vicinanza', () => {
  const finto = (lon, lat, area) => ({
    lon, lat, area, mx: modulo.mercX(lon), my: modulo.mercY(lat),
    ux: modulo.unitaX(lon), uy: modulo.unitaY(lat)
  });
  const insiemi = modulo.riunisciPerVicinanza([
    finto(15.0, 37.0, 30), finto(15.2, 37.1, 20),   // stessa cella
    finto(2.0, 48.0, 40)                            // mille chilometri piu' in la'
  ], modulo.LAMPO_RAGGIO_GRUPPO_KM);
  assert.equal(insiemi.length, 2, 'gruppi: ' + insiemi.length);
  assert.equal(insiemi.find((g) => g.length === 2).length, 2);
});

prova('l\'inviluppo del lampo ha picco e coda e finisce', () => {
  assert.equal(modulo.luceDelLampo(-5), 0, 'un lampo illumina prima di accendersi');
  assert.equal(modulo.luceDelLampo(100000), 0, 'un lampo non si spegne mai');
  const picco = modulo.luceDelLampo(36);
  assert.ok(picco > 0.6, 'il picco e\' troppo debole: ' + picco.toFixed(2));
  const coda = modulo.luceDelLampo(190);
  assert.ok(coda > 0.02 && coda < picco * 0.5,
    'manca la corrente continua dopo il colpo: ' + coda.toFixed(3));
  // Monotona in discesa dopo il picco: niente rimbalzi.
  for (let t = 60; t < 260; t += 10) {
    assert.ok(modulo.luceDelLampo(t) >= modulo.luceDelLampo(t + 10) - 1e-9,
      'l\'inviluppo risale a ' + t + ' ms');
  }
});

// --- la superficie --------------------------------------------------------
prova('sotto la nube non si disegna una superficie che il satellite non vede', () => {
  const misura = { larghezza: 8, altezza: 8 };
  const pelle = { data: new Uint8ClampedArray(8 * 8 * 4) };
  pelle.data.fill(255);
  const campo = {
    larghezza: 8, altezza: 8,
    copertura: new Float32Array(64), quota: new Float32Array(64)
  };
  for (let k = 0; k < 64; k++) campo.copertura[k] = k < 32 ? 0.95 : 0.0;
  modulo.superficieSenzaNubi(pelle, misura, campo);
  assert.ok(pelle.data[3] < 40, 'la superficie resta opaca sotto una nube spessa');
  assert.equal(pelle.data[(60 * 4) + 3], 255, 'col cielo sereno la superficie sparisce');
});

// --- dove guardare --------------------------------------------------------
prova('la cella piu\' viva sta dove ci sono i lampi', () => {
  const L = 200, A = 200;
  const campo = { larghezza: L, altezza: A, quota: new Float32Array(L * A) };
  for (let y = 0; y < A; y++) {
    for (let x = 0; x < L; x++) {
      campo.quota[y * L + x] = Math.hypot(x - 150, y - 60) < 30 ? 12 : 0.2;
    }
  }
  const senza = modulo.cellaPiuViva(campo, []);
  const d = modulo.DOMINIO;
  assert.ok(senza.lon > d.ovest && senza.lon < d.est, 'punto fuori dominio');
  // Con un lampo lontano dalle cime alte, vince comunque il lampo piu' grosso.
  const conLampo = modulo.cellaPiuViva(campo, [
    { lon: -5, lat: 35, area: 500 }, { lon: 20, lat: 55, area: 30 }
  ]);
  assert.ok(Math.abs(conLampo.lon + 5) < 1e-6 && Math.abs(conLampo.lat - 35) < 1e-6,
    'il lampo piu\' esteso non vince: ' + JSON.stringify(conLampo));
});

// --- il contratto con chi guarda -----------------------------------------
prova('niente terreno spostato: le nubi sono un volume', () => {
  // Disegnare le nubi come rilievo di un terreno le fa venire a punta.
  // Questa pagina non lo fa piu', e non deve tornare a farlo di nascosto.
  assert.ok(!/setTerrain|raster-dem|Terrain-?RGB|terrain-rgb/i.test(corpo[1]),
    'e\' ricomparso il terreno spostato al posto del volume');
  assert.ok(/this\.type = "custom"/.test(corpo[1]),
    'manca lo strato personalizzato del volume');
  assert.ok(/renderingMode = "3d"/.test(corpo[1]), 'lo strato non e\' in tre dimensioni');
});

prova('l\'HUD dichiara quello che non e\' misurato', () => {
  for (const parola of ['Dichiarati', 'relativa', 'non un CTTH',
                        'spessore della nube', 'esagerazione verticale']) {
    assert.ok(html.includes(parola),
      'l\'HUD non dichiara piu\': ' + parola);
  }
  // E dichiara i limiti veri della sorgente.
  assert.ok(/immagini gia' colorate/.test(html),
    'non e\' piu\' scritto che EUMETView serve immagini gia\' colorate');
  assert.ok(/di sbieco/.test(html), 'manca la parallasse del punto di vista di MTG');
});

prova('quando un campo manca, quella parte resta vuota', () => {
  assert.ok(/non disponibile/.test(corpo[1]),
    'manca la via d\'uscita per un campo assente');
  assert.ok(/non ne e' stata inventata una|non disegna nubi che non ha misurato/.test(corpo[1]),
    'manca la promessa di non inventare');
  // Il caso senza infrarosso esce senza pubblicare niente.
  const ramo = corpo[1].slice(corpo[1].indexOf('if (!forma) {'));
  assert.ok(ramo.indexOf('return;') > 0 && ramo.indexOf('return;') < 900,
    'senza infrarosso la scena prova a costruirsi lo stesso');
});

// --- lo shader ------------------------------------------------------------
prova('lo shader misura il cammino vero, non quello allungato', () => {
  const f = modulo.FRAMMENTO;
  // Il volume e' disegnato stirato in verticale, ma la luce lo attraversa
  // come se non lo fosse: se l'estinzione usasse il cammino allungato, la
  // stessa nube diventerebbe piu' opaca solo alzando il cursore.
  assert.ok(f.includes('direzione.z / uEsagerazione'),
    'il passo dell\'estinzione non viene riportato alla scala vera');
  assert.ok(/altKm = p\.z \* uCircKm \* cosLat \/ uEsagerazione/.test(f),
    'la quota nel volume non viene riportata alla scala vera');
});

prova('lo shader non inventa nubi fuori dal riquadro del dato', () => {
  const f = modulo.FRAMMENTO;
  assert.ok(/uv\.x < 0\.0 \|\| uv\.y < 0\.0 \|\| uv\.x > 1\.0 \|\| uv\.y > 1\.0/.test(f),
    'fuori dal dominio il volume non viene chiuso');
});

prova('la grana orizzontale cresce con l\'esagerazione', () => {
  // Stirare in altezza senza ingrossare in larghezza trasforma ogni colonna
  // in un ago: e' successo, ed e' il motivo per cui questa prova esiste.
  assert.ok(modulo.FRAMMENTO.includes('textureLod(uCampo, uv, uGrana)'),
    'il campo non viene piu\' preso a grana variabile');
  assert.ok(/Math\.log2\(this\.esagerazione \/ ESAGERAZIONE_GRANA\)/.test(corpo[1]),
    'la grana non e\' piu\' legata all\'esagerazione');
  assert.ok(/orizzontaleKm \/ this\.esagerazione/.test(corpo[1]),
    'la cella di rumore non e\' piu\' cubica in quello che si vede');
});

prova('il passo del raggio non supera mezzo mammellone', () => {
  // Con passi piu' lunghi delle strutture, il raggio le scavalca e lascia
  // punti neri sparsi sulle nubi.
  assert.ok(/uPassoMax/.test(modulo.FRAMMENTO), 'manca il tetto al passo del raggio');
  assert.ok(/passoFine = min\(attraversata \/ passi, uPassoMax\)/.test(modulo.FRAMMENTO),
    'il passo non e\' piu\' limitato dalla scala delle strutture');
});

prova('i posti dei lampi nello shader sono quelli dichiarati', () => {
  assert.ok(modulo.FRAMMENTO.includes('uniform vec4 uLampi[' + modulo.LAMPI_MAX + '];'),
    'lo shader e il codice non concordano su quanti lampi stanno in scena');
  assert.ok(modulo.FRAMMENTO.includes('for (int j = 0; j < ' + modulo.LAMPI_MAX + '; j++)'),
    'il giro sui lampi non copre tutti i posti');
});

prova('il lampo nasce nella regione di carica, a quota quasi fissa', () => {
  // E' la modifica che fa comparire da sola la faccia giusta del lampo.
  // Con una profondita' FISSA sotto la cima, sotto una torre alta quattordici
  // chilometri e sotto un'incudine alta nove c'e' lo stesso ghiaccio sopra la
  // sorgente, e la luce esce uguale dappertutto: niente nucleo scuro, niente
  // bordo acceso. Ancorata a una quota assoluta -- la regione fra -10 e -25
  // gradi -- la differenza c'e' e viene dalla fisica.
  const torre = modulo.quotaDiCarica(14);
  const incudine = modulo.quotaDiCarica(9);
  assert.ok(Math.abs(torre - incudine) < 1e-9,
    'la sorgente segue ancora la cima invece di stare alla sua quota');
  assert.ok(14 - torre > (9 - incudine) + 4,
    'sotto la torre non c\'e\' piu\' ghiaccio che sotto l\'incudine: '
    + 'il nucleo scuro non puo\' formarsi');
  assert.ok(modulo.LAMPO_QUOTA_CARICA_KM >= 4 && modulo.LAMPO_QUOTA_CARICA_KM <= 8,
    'la regione di carica e\' fuori dall\'intervallo osservato');
  // Sotto una nube bassa il fulmine nasce piu' in basso, non fuori dalla nube.
  const bassa = modulo.quotaDiCarica(3);
  assert.ok(bassa > 0 && bassa <= 3 - modulo.LAMPO_SOTTO_LA_CIMA_MIN_KM + 1e-9,
    'sotto una nube bassa la sorgente esce dalla nube: ' + bassa.toFixed(2));
  assert.ok(/exp\(-ottico \/ uLampoDiffusione\)/.test(modulo.FRAMMENTO),
    'la luce del lampo non si spegne piu\' nel ghiaccio');
});

prova('il cammino della luce e\' pesato dalla nube misurata', () => {
  // Dentro il cuore fitto la luce si ferma, dentro l'incudine sottile corre:
  // e' l'unico modo in cui il dato entra nella FORMA del bagliore, e senza
  // questo il bordo dell'incudine non si accende.
  assert.ok(modulo.LAMPO_PESO_COPERTURA > 0 && modulo.LAMPO_PESO_COPERTURA < 1,
    'il cammino ottico non dipende piu\' dalla copertura');
  assert.ok(/distanza \* \(1\.0 - uLampoPesoCopertura/.test(modulo.FRAMMENTO),
    'lo shader non pesa piu\' il cammino con la copertura');
});

prova('un flash e\' una sequenza di colpi, non un rigonfiamento solo', () => {
  for (let giro = 0; giro < 40; giro++) {
    const colpi = modulo.sequenzaDiColpi();
    assert.ok(colpi.length >= modulo.LAMPO_COLPI_MIN
      && colpi.length <= modulo.LAMPO_COLPI_MAX,
      'colpi fuori dall\'intervallo dichiarato: ' + colpi.length);
    assert.equal(colpi[0], 0, 'il primo colpo non e\' all\'istante zero');
    for (let i = 1; i < colpi.length; i++) {
      const salto = colpi[i] - colpi[i - 1];
      assert.ok(salto >= modulo.LAMPO_FRA_COLPI_MS[0] - 1e-9
        && salto <= modulo.LAMPO_FRA_COLPI_MS[1] + 1e-9,
        'intervallo fra colpi fuori dal dichiarato: ' + salto.toFixed(0));
    }
  }
  // La somma si satura: due colpi sovrapposti non fanno il doppio di luce.
  const doppio = modulo.luceDellaSequenza([0, 0], 36);
  assert.ok(doppio <= 1 + 1e-9, 'la luce dei colpi sovrapposti sfonda l\'uno');
  assert.ok(doppio > modulo.luceDellaSequenza([0], 36),
    'due colpi non danno piu\' luce di uno');
  // E lo sfarfallio c'e' davvero: fra un colpo e il successivo la luce cala.
  const sequenza = [0, 90];
  const fra = modulo.luceDellaSequenza(sequenza, 70);
  assert.ok(fra < modulo.luceDellaSequenza(sequenza, 36),
    'fra un colpo e l\'altro la luce non cala: non sfarfalla');
  assert.ok(modulo.luceDellaSequenza(sequenza, 126) > fra,
    'il secondo colpo non riaccende');
});

prova('il bagliore resta una palla, non diventa una colonna', () => {
  // Il bagliore e' una palla nello spazio vero. Disegnato in uno spazio
  // stirato quattordici volte in verticale diventava una colonna di luce che
  // sbucava dalla nube come un faro -- si vedeva nei fotogrammi. La verticale
  // viene schiacciata della stessa proporzione con cui viene stirata, come
  // gia' si fa per la grana del rumore.
  assert.ok(/uLampoSchiaccia/.test(modulo.FRAMMENTO),
    'il bagliore non viene piu\' schiacciato: torna a essere una colonna');
  assert.ok(/\(altKm - L\.z\) \* uLampoSchiaccia/.test(modulo.FRAMMENTO),
    'lo schiacciamento non si applica alla distanza verticale');
  assert.ok(/uniform1f\(u\("uLampoSchiaccia"\), ingrosso\)/.test(corpo[1]),
    'lo schiacciamento non segue piu\' l\'esagerazione');
  // E il bagliore non si allunga all\'infinito: oltre tre lunghezze di
  // diffusione non resta niente da sommare.
  assert.ok(/distanza > 3\.0 \* uLampoDiffusione/.test(modulo.FRAMMENTO),
    'il contributo del lampo non viene piu\' tagliato a distanza');
});

prova('il flash attraversa la cella invece di accendersi tutto insieme', () => {
  assert.ok(modulo.LAMPO_VELOCITA_KM_S > 20 && modulo.LAMPO_VELOCITA_KM_S < 3e5,
    'la velocita\' del leader non e\' un ordine di grandezza plausibile');
  assert.ok(/ritardoMs: Math\.sqrt/.test(corpo[1]),
    'le sorgenti di uno stesso gruppo non hanno piu\' un ritardo di propagazione');
  assert.ok(/ora - gruppo\.acceso - \(l\.ritardoMs \|\| 0\)/.test(corpo[1]),
    'il ritardo di propagazione non viene piu\' usato');
});

prova('la matrice si inverte davvero', () => {
  // Senza questa, il raggio che parte da un pixel punta da un\'altra parte.
  const m = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1.5, 0, 4, -2, 7, 1];
  const inv = modulo.inverti4(m);
  const moltiplica = (a, b) => {
    const c = new Array(16).fill(0);
    for (let r = 0; r < 4; r++) {
      for (let col = 0; col < 4; col++) {
        for (let k = 0; k < 4; k++) c[col * 4 + r] += a[k * 4 + r] * b[col * 4 + k];
      }
    }
    return c;
  };
  const identita = moltiplica(m, inv);
  for (let i = 0; i < 16; i++) {
    const atteso = (i % 5 === 0) ? 1 : 0;
    assert.ok(Math.abs(identita[i] - atteso) < 1e-5,
      'M * M^-1 non e\' l\'identita\' in posizione ' + i);
  }
});

process.exitCode = ok ? 0 : 1;
if (!ok) console.log('\nALCUNE PROVE SONO FALLITE');
else console.log('\nSUPERATO');
