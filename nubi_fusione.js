// ==================================================================
// ---- NUBI 3D DA FUSIONE SATELLITE + ICON-2I ------------------------
// ==================================================================
//
// Il ray marcher del volume costruito lato server. La texture RGBA arriva
// gia' cotta da scripts/bake_cloud_volume.py:
//
//   R  cima       (IR 10,5 um EUMETSAT sul profilo termico ICON-2I)
//   G  densita'   (albedo VIS 0,6 um; di notte contrasto IR)
//   B  base       (LCL ICON-2I sopra l'orografia)
//   A  convezione (0,45*sqrt(2*CAPE) ICON-2I, normalizzata a 40 m/s)
//
// Il satellite e' la maschera: dove G vale zero la nube non esiste. Qui il
// raggio di vista attraversa soltanto la fascia fra base e cima di ogni
// colonna; il rumore procedurale 3D scolpisce la forma sotto la risoluzione
// del satellite -- Perlin per i vuoti macroscopici, Worley sottratto per i
// bordi -- e il canale A decide QUANTO il Worley morde: cavolfiori per la
// convezione profonda, lamine per gli strati. La luce e' Beer-Lambert verso
// il sole, con l'effetto polvere e la funzione di fase di Henyey-Greenstein.
(function (radice) {
  "use strict";

  const SORGENTE = "data_weather/live/clouds/volume.json";
  const RAGGIO_TERRA = 6378137;
  const CIRCONFERENZA_KM = 2 * Math.PI * RAGGIO_TERRA / 1000;

  const mobile = typeof navigator !== "undefined"
    && /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
  // La qualita' e' fissa per dispositivo, come nel volume esistente: il
  // raggio si disegna in una tela ridotta e la si stende sulla mappa.
  const QUALITA = mobile
    ? { pixel: 480000, passi: 128, passiLuce: 4, esagerazione: 3.2 }
    : { pixel: 1600000, passi: 224, passiLuce: 6, esagerazione: 3.5 };

  // L'esagerazione verticale: piena alla vista d'insieme, poi cala con lo
  // zoom -- la stessa regola del volume dal solo satellite.
  const ESAGERAZIONE_ZOOM_PIENO = 5.0;
  const ESAGERAZIONE_CALO_PER_ZOOM = 0.55;
  const ESAGERAZIONE_MINIMA = 1.6;

  // Ottica. SIGMA e' l'estinzione per km di nube a densita' 1: un cumulo di
  // un chilometro e' gia' opaco. G e' l'anisotropia del lobo in avanti delle
  // goccioline: 0,6 da' il bordo d'argento controluce senza accecare.
  const SIGMA_PER_KM = 3.6;
  const FASE_G = 0.6;
  // Il rumore: i vuoti di Perlin hanno un periodo di 48 km (buchi larghi
  // quanto i varchi fra le celle), le bolle di Worley di 6 km.
  const PERLIN_LATO = 64;
  const PERLIN_KM = 48;
  const WORLEY_LATO = 32;
  const WORLEY_KM = 6;
  const SOLE_MINIMO_GRADI = -6;

  const clamp = (v, a, b) => (v < a ? a : (v > b ? b : v));
  const unitaX = (lon) => (lon + 180) / 360;
  function unitaY(lat) {
    const f = clamp(lat, -85.0511, 85.0511) * Math.PI / 180;
    return 0.5 - Math.log(Math.tan(Math.PI / 4 + f / 2)) / (2 * Math.PI);
  }

  // Il dominio della texture in coordinate unitarie di MapLibre (x verso
  // est, y verso sud, entrambe in [0,1]).
  function dominioUnitario(bounds) {
    return {
      x0: unitaX(bounds.west), x1: unitaX(bounds.east),
      yNord: unitaY(bounds.north), ySud: unitaY(bounds.south)
    };
  }

  // --- il file ------------------------------------------------------------
  // Il PNG si decodifica qui e non con <img>: il canale alfa e' un DATO (la
  // convezione) e un browser che premoltiplica l'alfa azzererebbe cima,
  // densita' e base di ogni nube stratiforme, dove A vale zero.
  function paeth(a, b, c) {
    const p = a + b - c;
    const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
    return pa <= pb && pa <= pc ? a : (pb <= pc ? b : c);
  }

  async function decodificaPng(buffer) {
    const byte = new Uint8Array(buffer);
    const firma = [137, 80, 78, 71, 13, 10, 26, 10];
    for (let i = 0; i < 8; i++) {
      if (byte[i] !== firma[i]) throw new Error("volume: non e' un PNG");
    }
    const vista = new DataView(buffer instanceof ArrayBuffer ? buffer : byte.buffer,
      byte.byteOffset, byte.byteLength);
    let pos = 8, larghezza = 0, altezza = 0;
    const pezzi = [];
    while (pos < byte.length) {
      const lunghezza = vista.getUint32(pos);
      const tipo = String.fromCharCode(byte[pos + 4], byte[pos + 5], byte[pos + 6], byte[pos + 7]);
      const dati = byte.subarray(pos + 8, pos + 8 + lunghezza);
      if (tipo === "IHDR") {
        larghezza = vista.getUint32(pos + 8);
        altezza = vista.getUint32(pos + 12);
        if (dati[8] !== 8 || dati[9] !== 6 || dati[12] !== 0) {
          throw new Error("volume: atteso PNG RGBA a 8 bit non interlacciato");
        }
      } else if (tipo === "IDAT") {
        pezzi.push(dati);
      } else if (tipo === "IEND") {
        break;
      }
      pos += 12 + lunghezza;
    }
    const compresso = new Blob(pezzi);
    const flusso = compresso.stream().pipeThrough(new DecompressionStream("deflate"));
    const grezzo = new Uint8Array(await new Response(flusso).arrayBuffer());
    const riga = larghezza * 4;
    if (grezzo.length < altezza * (riga + 1)) throw new Error("volume: PNG troncato");
    const fuori = new Uint8Array(altezza * riga);
    for (let y = 0; y < altezza; y++) {
      const filtro = grezzo[y * (riga + 1)];
      const da = y * (riga + 1) + 1, a = y * riga, sopra = a - riga;
      for (let x = 0; x < riga; x++) {
        const r = grezzo[da + x];
        const sinistra = x >= 4 ? fuori[a + x - 4] : 0;
        const su = y > 0 ? fuori[sopra + x] : 0;
        const diagonale = y > 0 && x >= 4 ? fuori[sopra + x - 4] : 0;
        let v;
        if (filtro === 0) v = r;
        else if (filtro === 1) v = r + sinistra;
        else if (filtro === 2) v = r + su;
        else if (filtro === 3) v = r + ((sinistra + su) >> 1);
        else if (filtro === 4) v = r + paeth(sinistra, su, diagonale);
        else throw new Error("volume: filtro PNG " + filtro);
        fuori[a + x] = v & 255;
      }
    }
    return { larghezza: larghezza, altezza: altezza, dati: fuori };
  }

  async function caricaVolume(url) {
    const indirizzo = url || SORGENTE;
    const risposta = await fetch(indirizzo + "?t=" + Date.now(), { cache: "no-store" });
    if (!risposta.ok) throw new Error("volume: metadati assenti (" + risposta.status + ")");
    const meta = await risposta.json();
    const base = indirizzo.slice(0, indirizzo.lastIndexOf("/") + 1);
    const immagine = await fetch(base + (meta.image || "volume.png"));
    if (!immagine.ok) throw new Error("volume: texture assente (" + immagine.status + ")");
    const png = await decodificaPng(await immagine.arrayBuffer());
    if (png.larghezza !== meta.width || png.altezza !== meta.height) {
      throw new Error("volume: dimensioni della texture diverse dai metadati");
    }
    let cimaMassima = 0;
    for (let k = 0; k < png.dati.length; k += 4) {
      if (png.dati[k + 1] > 2 && png.dati[k] > cimaMassima) cimaMassima = png.dati[k];
    }
    return {
      meta: meta,
      texture: png,
      cimaMassimaKm: cimaMassima / 255 * (meta.scaleKm || 16)
    };
  }

  // --- il rumore procedurale -------------------------------------------
  // Tutto periodico: il rumore si ripete senza giunture sul lato del cubo.
  function hash3(x, y, z, seme) {
    let n = (x * 374761393 + y * 668265263 + z * 1274126177 + seme * 1442695040) | 0;
    n = Math.imul(n ^ (n >>> 13), 1103515245);
    n = Math.imul(n ^ (n >>> 16), 2654435769);
    return ((n ^ (n >>> 15)) >>> 0) / 4294967295;
  }

  // Perlin a gradiente (non a valori): i 12 gradienti di Perlin 2002 su un
  // reticolo periodico, piu' ottave, riportato a [0,1] sui suoi estremi.
  const GRADIENTI = [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1],
    [1, 0, -1], [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]];
  function perlin3D(lato, periodo, ottave, seme) {
    const fuori = new Float32Array(lato * lato * lato);
    const dissolvi = (t) => t * t * t * (t * (t * 6 - 15) + 10);
    let ampiezza = 1;
    for (let o = 0; o < ottave; o++) {
      const celle = periodo << o;
      const g = (x, y, z) => GRADIENTI[Math.floor(hash3(((x % celle) + celle) % celle,
        ((y % celle) + celle) % celle, ((z % celle) + celle) % celle, seme + o) * 12) % 12];
      for (let z = 0; z < lato; z++) {
        const fz = z / lato * celle, z0 = Math.floor(fz), tz = fz - z0, wz = dissolvi(tz);
        for (let y = 0; y < lato; y++) {
          const fy = y / lato * celle, y0 = Math.floor(fy), ty = fy - y0, wy = dissolvi(ty);
          for (let x = 0; x < lato; x++) {
            const fx = x / lato * celle, x0 = Math.floor(fx), tx = fx - x0, wx = dissolvi(tx);
            let somma = 0;
            for (let c = 0; c < 8; c++) {
              const dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
              const gr = g(x0 + dx, y0 + dy, z0 + dz);
              const prodotto = gr[0] * (tx - dx) + gr[1] * (ty - dy) + gr[2] * (tz - dz);
              somma += prodotto * (dx ? wx : 1 - wx) * (dy ? wy : 1 - wy) * (dz ? wz : 1 - wz);
            }
            fuori[(z * lato + y) * lato + x] += somma * ampiezza;
          }
        }
      }
      ampiezza *= 0.5;
    }
    return normalizza(fuori);
  }

  // Worley rovesciato e periodico: 1 al centro della bolla, 0 fra le bolle.
  function worley3D(lato, celle, seme) {
    const punti = new Float32Array(celle * celle * celle * 3);
    for (let c = 0; c < celle * celle * celle; c++) {
      punti[c * 3] = hash3(c, 1, 7, seme);
      punti[c * 3 + 1] = hash3(c, 2, 11, seme);
      punti[c * 3 + 2] = hash3(c, 3, 13, seme);
    }
    const fuori = new Float32Array(lato * lato * lato);
    const scala = celle / lato;
    for (let z = 0; z < lato; z++) {
      const fz = (z + 0.5) * scala, cz = Math.floor(fz);
      for (let y = 0; y < lato; y++) {
        const fy = (y + 0.5) * scala, cy = Math.floor(fy);
        for (let x = 0; x < lato; x++) {
          const fx = (x + 0.5) * scala, cx = Math.floor(fx);
          let minimo = 9;
          for (let dz = -1; dz <= 1; dz++) {
            const nz = cz + dz, wz = (nz + celle) % celle;
            for (let dy = -1; dy <= 1; dy++) {
              const ny = cy + dy, wy = (ny + celle) % celle;
              for (let dx = -1; dx <= 1; dx++) {
                const nx = cx + dx, wx = (nx + celle) % celle;
                const k = ((wz * celle + wy) * celle + wx) * 3;
                const ex = nx + punti[k] - fx, ey = ny + punti[k + 1] - fy, ez = nz + punti[k + 2] - fz;
                const d = ex * ex + ey * ey + ez * ez;
                if (d < minimo) minimo = d;
              }
            }
          }
          fuori[(z * lato + y) * lato + x] = 1 - Math.min(1, Math.sqrt(minimo));
        }
      }
    }
    return fuori;
  }

  function normalizza(campo) {
    let min = Infinity, max = -Infinity;
    for (let k = 0; k < campo.length; k++) {
      if (campo[k] < min) min = campo[k];
      if (campo[k] > max) max = campo[k];
    }
    const ampiezza = max - min || 1;
    for (let k = 0; k < campo.length; k++) campo[k] = (campo[k] - min) / ampiezza;
    return campo;
  }

  // R: Perlin fBm a tre ottave. Un canale solo.
  function texturePerlin(lato) {
    const p = perlin3D(lato, 4, 3, 17);
    const dati = new Uint8Array(p.length);
    for (let k = 0; k < p.length; k++) dati[k] = Math.round(p[k] * 255);
    return dati;
  }

  // RGB: Worley a 4, 8 e 16 celle per lato; lo shader ne fa l'fBm.
  function textureWorley(lato) {
    const w1 = worley3D(lato, 4, 31), w2 = worley3D(lato, 8, 32), w3 = worley3D(lato, 16, 33);
    const dati = new Uint8Array(w1.length * 4);
    for (let k = 0; k < w1.length; k++) {
      dati[k * 4] = Math.round(w1[k] * 255);
      dati[k * 4 + 1] = Math.round(w2[k] * 255);
      dati[k * 4 + 2] = Math.round(w3[k] * 255);
      dati[k * 4 + 3] = 255;
    }
    return dati;
  }

  // --- gli shader ---------------------------------------------------------
  const VERTICE = [
    "#version 300 es",
    "in vec2 aPos;",
    "out vec2 vNdc;",
    "void main() { vNdc = aPos; gl_Position = vec4(aPos, 0.0, 1.0); }"
  ].join("\n");

  const FRAMMENTO = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vNdc;
out vec4 colorePixel;

uniform mat4 uInversa;
uniform vec4 uDominio;        // x0, yNord, x1, ySud (unita' Mercatore)
uniform sampler2D uCampo;     // R cima, G densita', B base, A convezione
uniform sampler3D uPerlin;    // fBm di Perlin: i vuoti macroscopici
uniform sampler3D uWorley;    // Worley a 4/8/16 celle: l'erosione dei bordi
uniform float uScalaKm;
uniform float uCircKm;
uniform float uEsagerazione;
uniform float uZMax;
uniform float uTexelCampo;    // lato di un pixel della texture (Mercatore)
uniform float uPixelAngolo;   // ampiezza angolare di un pixel della tela
uniform float uPassoKm;       // salita verticale del raggio fra due campioni
uniform int uPassi;
uniform int uPassiLuce;
uniform float uSigma;
uniform float uFaseG;
uniform vec3 uSole;           // verso il sole, spazio vero (x est, y sud, z su)
uniform vec3 uColoreSole;
uniform float uForzaSole;
uniform vec3 uCielo;          // luce diffusa del cielo, dall'alto
uniform vec3 uSuolo;          // luce riflessa dal suolo, dal basso
uniform vec3 uFoschia;
uniform float uPerlinKm;
uniform float uWorleyKm;
uniform float uLatoPerlin;
uniform float uLatoWorley;
uniform vec3 uSemenza;

const float PI = 3.14159265359;

float cosLatDa(float y) { return 1.0 / cosh(PI * (1.0 - 2.0 * y)); }

float rimappa(float v, float a, float b, float c, float d) {
  return c + (v - a) / max(1e-4, b - a) * (d - c);
}

// HENYEY-GREENSTEIN. La goccia d'acqua diffonde soprattutto in avanti: con
// il sole dietro la cella, il raggio che sfiora il bordo raccoglie il lobo
// in avanti ed e' li' che nasce il bordo d'argento. Un piccolo lobo
// all'indietro (g negativo) tiene viva la nube vista con il sole alle spalle.
float henyeyGreenstein(float coseno, float g) {
  float g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1e-4, 1.0 + g2 - 2.0 * g * coseno), 1.5));
}
float fase(float coseno) {
  return mix(henyeyGreenstein(coseno, uFaseG), henyeyGreenstein(coseno, -0.25), 0.22) * 4.0 * PI;
}

// La colonna letta dalla texture: la nube esiste solo fra base e cima.
// Ritorna la densita' 0-1; in uscita l'altezza relativa e la convezione.
float densita(vec3 p, float lodCampo, float lodRumore, bool leggera,
              out float hRel, out float conv) {
  hRel = 0.0; conv = 0.0;
  vec2 uv = vec2((p.x - uDominio.x) / (uDominio.z - uDominio.x),
                 (p.y - uDominio.y) / (uDominio.w - uDominio.y));
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return 0.0;
  vec4 campo = textureLod(uCampo, uv, lodCampo);
  // LA MASCHERA E' DEL SATELLITE: senza densita' osservata non c'e' nube.
  if (campo.g < 0.004) return 0.0;
  float kmPerUnita = uCircKm * cosLatDa(p.y);
  float altKm = p.z * kmPerUnita / uEsagerazione;
  // Due scale: la sagoma (G) alla risoluzione piena, le QUOTE un paio di
  // livelli piu' in su. Stirata dall'esagerazione, ogni colonna di un pixel
  // con la sua cima diventerebbe un ago; il server estende R e B fuori
  // dalla maschera, cosi' la media non trascina a terra le quote di bordo.
  vec4 liscio = textureLod(uCampo, uv, lodCampo + 1.5);
  campo.rba = mix(liscio.rba, campo.rba, 0.3);
  float cimaKm = campo.r * uScalaKm;
  float baseKm = campo.b * uScalaKm;
  // BOUNDING VOLUME: fuori dalla fascia base-cima il raggio non trova nulla.
  if (altKm < baseKm || altKm > cimaKm) return 0.0;
  // Il CAPE scolpisce a cavolfiore solo una nube che il satellite vede
  // spessa: un velo sopra aria instabile resta un velo.
  conv = campo.a * smoothstep(0.3, 0.8, campo.g);
  float h = (altKm - baseKm) / max(0.05, cimaKm - baseKm);
  hRel = h;
  float convettiva = smoothstep(0.05, 0.65, conv);
  // Il profilo della colonna: uno strato e' una lastra con base e cima
  // morbide; una nube convettiva ha base piatta e si stringe verso la cupola.
  float fondo = smoothstep(0.0, mix(0.18, 0.05, convettiva), h);
  float tetto = 1.0 - smoothstep(mix(0.72, 0.45, convettiva), 1.0, h);
  float copertura = campo.g * fondo * tetto;
  if (copertura <= 0.0) return 0.0;

  // Coordinate del rumore in km VISTI: la verticale e' moltiplicata per
  // l'esagerazione, altrimenti ogni bolla stirata diventerebbe un ago e la
  // convezione una foresta di colonne. Negli strati la verticale e'
  // compressa ancora: le strutture si allungano in orizzontale e restano
  // lamine. Nella convezione la bolla resta tonda sullo schermo.
  float lamina = mix(4.0, 1.0, convettiva);
  vec3 q = vec3(p.xy * kmPerUnita, altKm * uEsagerazione * lamina);

  // PERLIN: i vuoti macroscopici. Dove la densita' osservata e' bassa solo
  // i massimi del Perlin sopravvivono e la nube si apre in banchi e varchi;
  // dove e' alta il Perlin modula la materia senza bucarla.
  // La soglia usa la radice della copertura e non scende mai sotto il 35%
  // del rumore: un velo sottile osservato dal satellite si apre in banchi
  // ma non sparisce -- la maschera resta del satellite.
  // Il periodo verticale e' un sesto di quello orizzontale: con lo stesso
  // periodo il Perlin resterebbe costante lungo tutta la colonna e i vuoti
  // diventerebbero pozzi verticali -- il cilindro estruso da evitare.
  vec3 qp = vec3(q.xy / uPerlinKm, q.z / (uPerlinKm / 6.0)) + uSemenza;
  float perlin = textureLod(uPerlin, qp, lodRumore).r;
  float soglia = (1.0 - sqrt(copertura)) * 0.65;
  float forma = clamp(rimappa(perlin, soglia, 1.0, 0.0, 1.0), 0.0, 1.0) * copertura;
  if (forma <= 0.0 || leggera) return forma;

  // WORLEY SOTTRATTO: il bordo cumuliforme. Il Worley rovesciato vale 1 al
  // centro delle bolle e 0 fra le bolle: sottraendolo si scavano i solchi
  // fra i lobi. Il CAPE (canale A) decide quanto morde: con la convezione
  // profonda i solchi arrivano in profondita' e restano i cavolfiori;
  // negli strati sfrangia appena il bordo.
  // Due letture a scale non commensurabili e ruotate: una sola ripeteva il
  // cubo di rumore in file regolari di bolle.
  vec3 qw = q / uWorleyKm + uSemenza.zxy;
  vec3 w1 = textureLod(uWorley, qw, lodRumore).rgb;
  vec3 w2 = textureLod(uWorley, vec3(mat2(0.8, -0.6, 0.6, 0.8) * qw.xy * 1.37, qw.z * 1.37 + 0.41),
    lodRumore + 0.45).rgb;
  vec3 w = mix(w1, w2, 0.4);
  float worley = w.r * 0.625 + w.g * 0.25 + w.b * 0.125;
  // In basso ciuffi (Worley diritto), in alto cupole (rovesciato).
  float dettaglio = mix(worley, 1.0 - worley, smoothstep(0.15, 0.6, h));
  float morso = mix(0.14, 0.9, convettiva);
  return clamp(rimappa(forma, dettaglio * morso, 1.0, 0.0, 1.0), 0.0, 1.0);
}

// BEER-LAMBERT verso il sole: da ogni campione si cammina verso il sole a
// passi che raddoppiano e si somma lo spessore ottico attraversato.
float spessoreVersoSole(vec3 p, float lodCampo, float lodRumore) {
  float kmPerUnita = uCircKm * cosLatDa(p.y);
  vec3 verso = vec3(uSole.xy, uSole.z * uEsagerazione) / kmPerUnita;
  float tau = 0.0, fatto = 0.0, lung = 0.3;
  for (int i = 0; i < 8; i++) {
    if (i >= uPassiLuce) break;
    vec3 q = p + verso * (fatto + 0.5 * lung);
    float h, c;
    tau += densita(q, lodCampo + 0.5 * float(i), lodRumore + 0.5 * float(i), true, h, c)
      * uSigma * lung;
    fatto += lung;
    lung *= 2.0;
  }
  return tau;
}

float intreccio(vec2 f) {
  return fract(52.9829189 * fract(dot(f, vec2(0.06711056, 0.00583715))));
}

void main() {
  vec4 a = uInversa * vec4(vNdc, -1.0, 1.0);
  vec4 b = uInversa * vec4(vNdc, 1.0, 1.0);
  vec3 origine = a.xyz / a.w;
  vec3 direzione = normalize(b.xyz / b.w - origine);
  vec3 minimo = vec3(min(uDominio.x, uDominio.z), min(uDominio.y, uDominio.w), 0.0);
  vec3 massimo = vec3(max(uDominio.x, uDominio.z), max(uDominio.y, uDominio.w), uZMax);
  vec3 sicura = direzione;
  if (abs(sicura.x) < 1e-9) sicura.x = 1e-9;
  if (abs(sicura.y) < 1e-9) sicura.y = 1e-9;
  if (abs(sicura.z) < 1e-9) sicura.z = 1e-9;
  vec3 t1 = (minimo - origine) / sicura;
  vec3 t2 = (massimo - origine) / sicura;
  vec3 tMin = min(t1, t2), tMax = max(t1, t2);
  float tVicino = max(max(tMin.x, tMin.y), max(tMin.z, 0.0));
  float tLontano = min(min(tMax.x, tMax.y), tMax.z);
  if (tLontano <= tVicino) { colorePixel = vec4(0.0); return; }

  float cosLat0 = cosLatDa(origine.y + direzione.y * tVicino);
  // Il passo: il raggio sale di uPassoKm veri fra due campioni, cosi' uno
  // strato sottile non viene scavalcato; mai piu' di un pixel della texture
  // in orizzontale, e mai piu' passi di uPassi.
  float passoZ = uPassoKm * uEsagerazione / (uCircKm * cosLat0);
  float passoMin = min(passoZ / max(abs(direzione.z), 1e-3), uTexelCampo * 0.9);
  // Il passo cresce con la distanza (0,4% del cammino gia' fatto): fitto
  // vicino alla camera, dove una bolla occupa molti pixel, largo lontano,
  // dove ne occupa uno. Un passo uniforme sull'intero raggio radente
  // lasciava striature radiali davanti alla camera.
  float crescita = 0.004;
  float scarto = intreccio(gl_FragCoord.xy);
  float t = tVicino + max(passoMin, tVicino * crescita) * scarto;

  vec3 direzioneVera = normalize(vec3(direzione.xy, direzione.z / uEsagerazione));
  float coseno = dot(direzioneVera, uSole);
  float fasePasso = fase(coseno);
  // L'effetto polvere scurisce i bordi rivolti al sole e accende le pieghe
  // interne; guardando verso il sole non deve spegnere il bordo d'argento.
  float pesoPolvere = 0.5 - 0.5 * coseno;

  vec3 luce = vec3(0.0);
  float trasmissione = 1.0;
  float tPrimo = -1.0;
  for (int i = 0; i < 400; i++) {
    if (i >= uPassi || t > tLontano || trasmissione < 0.01) break;
    // ...ma mai cosi' corto da non arrivare in fondo con i passi rimasti.
    float passo = max(max(passoMin, t * crescita), (tLontano - t) / float(max(1, uPassi - i)));
    vec3 p = origine + direzione * t;
    float lodCampo = max(0.0, log2(max(1e-12, t * uPixelAngolo) / uTexelCampo));
    float kmPerUnita = uCircKm * cosLatDa(p.y);
    float lodRumore = max(0.0, log2(max(1e-12, t * uPixelAngolo) * kmPerUnita
      * uLatoWorley / uWorleyKm));
    float h, conv;
    float d = densita(p, lodCampo, lodRumore, false, h, conv);
    if (d > 0.001) {
      if (tPrimo < 0.0) tPrimo = t;
      float passoKm = length(vec3(direzione.xy, direzione.z / uEsagerazione)) * passo * kmPerUnita;
      float estinzione = uSigma * d;                     // per km
      float trasmPasso = exp(-estinzione * passoKm);     // Beer-Lambert sul raggio di vista

      float tauSole = uForzaSole > 0.01 ? spessoreVersoSole(p, lodCampo, lodRumore) : 0.0;
      float beer = exp(-tauSole);
      // POWDER: 1 - e^(-densita' x 2). E' la probabilita' che la luce abbia
      // gia' incontrato abbastanza goccioline da essere rimbalzata verso di
      // noi: bassa sul velo esterno, piena nelle pieghe dentro la nube.
      float powder = 1.0 - exp(-estinzione * 2.0);
      float direttaSole = beer * mix(1.0, 2.0 * powder, pesoPolvere);
      // La diffusione multipla, in due ottave piu' morbide: e' cio' che
      // impedisce al nucleo di diventare nero opaco.
      float multipla = 0.45 * exp(-tauSole * 0.25) + 0.2 * exp(-tauSole * 0.08);
      vec3 sole = uColoreSole * uForzaSole * (direttaSole * fasePasso + multipla);
      vec3 ambiente = mix(uSuolo, uCielo, smoothstep(0.0, 1.0, h))
        * (0.55 + 0.45 * powder);
      // Integrazione a energia conservata (Hillaire 2016): la luce diffusa
      // dal tratto e' quella che il tratto toglie al raggio.
      luce += trasmissione * (sole + ambiente) * (1.0 - trasmPasso);
      trasmissione *= trasmPasso;
    }
    t += passo;
  }

  float alfa = 1.0 - trasmissione;
  if (alfa < 0.003) { colorePixel = vec4(0.0); return; }
  // Foschia: una nube a centinaia di chilometri si vela d'azzurro.
  float distanzaKm = (tPrimo > 0.0 ? tPrimo : tVicino) * uCircKm * cosLat0;
  float velo = 1.0 - exp(-distanzaKm / 900.0);
  vec3 colore = mix(luce, uFoschia * alfa, velo * 0.55);
  // Esposizione morbida: il bordo d'argento satura senza tagliare.
  colore = alfa * (1.0 - exp(-colore / max(alfa, 1e-3) * 1.15));
  colorePixel = vec4(colore, alfa);
}
`;

  // La stesura: la tela ridotta si distende con un filtro a tenda 3x3,
  // quattro letture bilineari che assorbono anche la trama dello scarto.
  const STESURA = [
    "#version 300 es",
    "precision highp float;",
    "in vec2 vNdc;",
    "out vec4 colorePixel;",
    "uniform sampler2D uVolume;",
    "uniform vec2 uTexel;",
    "void main() {",
    "  vec2 uv = vNdc * 0.5 + 0.5;",
    "  colorePixel = 0.25 * (",
    "      texture(uVolume, uv + uTexel * vec2(-0.5, -0.5))",
    "    + texture(uVolume, uv + uTexel * vec2(0.5, -0.5))",
    "    + texture(uVolume, uv + uTexel * vec2(-0.5, 0.5))",
    "    + texture(uVolume, uv + uTexel * vec2(0.5, 0.5)));",
    "}"
  ].join("\n");

  // --- geometria e sole ---------------------------------------------------
  function inverti4(m) {
    const f = new Float64Array(16);
    f[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    f[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    f[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    f[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    f[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    f[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    f[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    f[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    f[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    f[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    f[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    f[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    f[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
    f[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
    f[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
    f[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];
    const det = m[0]*f[0] + m[1]*f[4] + m[2]*f[8] + m[3]*f[12];
    if (!det) return null;
    const fuori = new Float32Array(16);
    for (let i = 0; i < 16; i++) fuori[i] = f[i] / det;
    return fuori;
  }

  // Posizione del sole (serie brevi di Meeus): elevazione e azimut da nord.
  function posizioneSole(quando, lat, lon) {
    const giorni = (quando - Date.UTC(2000, 0, 1, 12)) / 86400000;
    const L = (280.46 + 0.9856474 * giorni) * Math.PI / 180;
    const g = (357.528 + 0.9856003 * giorni) * Math.PI / 180;
    const lambda = L + (1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * Math.PI / 180;
    const eps = (23.439 - 0.0000004 * giorni) * Math.PI / 180;
    const decl = Math.asin(Math.sin(eps) * Math.sin(lambda));
    const gmst = (18.697374558 + 24.06570982441908 * giorni) % 24;
    const ar = Math.atan2(Math.cos(eps) * Math.sin(lambda), Math.cos(lambda));
    const oraAngolo = (gmst * 15 + lon) * Math.PI / 180 - ar;
    const la = lat * Math.PI / 180;
    const elevazione = Math.asin(Math.sin(la) * Math.sin(decl)
      + Math.cos(la) * Math.cos(decl) * Math.cos(oraAngolo));
    const azimut = Math.atan2(-Math.sin(oraAngolo),
      Math.tan(decl) * Math.cos(la) - Math.sin(la) * Math.cos(oraAngolo));
    return { elevazione: elevazione, azimut: azimut };
  }

  // Il sole nello spazio della mappa: x est, y SUD (Mercatore), z su.
  function vettoreSole(sole) {
    const v = [
      Math.cos(sole.elevazione) * Math.sin(sole.azimut),
      -Math.cos(sole.elevazione) * Math.cos(sole.azimut),
      Math.max(0.02, Math.sin(sole.elevazione))
    ];
    const n = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / n, v[1] / n, v[2] / n];
  }

  // La luce della scena: il sole si arrossa quando e' basso e sparisce sotto
  // l'orizzonte; di notte resta solo un cielo scuro.
  function luceDiScena(sole) {
    const gradi = sole.elevazione * 180 / Math.PI;
    const forza = clamp((gradi - SOLE_MINIMO_GRADI) / 10, 0, 1);
    const basso = 1 - clamp(gradi / 25, 0, 1);
    const colore = [1.0, 0.96 - 0.22 * basso, 0.9 - 0.42 * basso].map((v) => v * 2.6);
    const mescola = (notte, giorno) => notte.map((v, i) => v + (giorno[i] - v) * forza);
    return {
      forza: forza,
      colore: colore,
      cielo: mescola([0.10, 0.12, 0.18], [0.46, 0.58, 0.78]),
      suolo: mescola([0.03, 0.035, 0.05], [0.26, 0.25, 0.23]),
      foschia: mescola([0.06, 0.08, 0.12], [0.72, 0.81, 0.92])
    };
  }

  // --- lo strato personalizzato -------------------------------------------
  function compila(gl, tipo, sorgente) {
    const s = gl.createShader(tipo);
    gl.shaderSource(s, sorgente);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error("shader nubi fuse: " + gl.getShaderInfoLog(s));
    }
    return s;
  }

  function collega(gl, frammento) {
    const programma = gl.createProgram();
    gl.attachShader(programma, compila(gl, gl.VERTEX_SHADER, VERTICE));
    gl.attachShader(programma, compila(gl, gl.FRAGMENT_SHADER, frammento));
    gl.bindAttribLocation(programma, 0, "aPos");
    gl.linkProgram(programma);
    if (!gl.getProgramParameter(programma, gl.LINK_STATUS)) {
      throw new Error("programma nubi fuse: " + gl.getProgramInfoLog(programma));
    }
    return programma;
  }

  function StratoNubiFuse(id) {
    this.id = id || "nubi-fuse";
    this.type = "custom";
    this.renderingMode = "3d";
    this.volume = null;
    this.acceso = true;
    this.fbo = null;
    this.tela = null;
    this.misuraTela = { larghezza: 0, altezza: 0 };
    this.errore = null;
  }

  StratoNubiFuse.prototype.onAdd = function (mappa, gl) {
    this.mappa = mappa;
    this.gl = gl;
    this.programma = collega(gl, FRAMMENTO);
    this.programmaStesura = collega(gl, STESURA);
    this.buffer = gl.createBuffer();
    // Il layer possiede il suo VAO: sui driver Android il VAO lasciato attivo
    // da MapLibre non porta gli attributi del triangolo a pieno schermo.
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    const carica3D = function (lato, formato, formatoDati, dati) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, formato, lato, lato, lato, 0, formatoDati,
        gl.UNSIGNED_BYTE, dati);
      gl.generateMipmap(gl.TEXTURE_3D);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.REPEAT);
      return tex;
    };
    this.texPerlin = carica3D(PERLIN_LATO, gl.R8, gl.RED, texturePerlin(PERLIN_LATO));
    this.texWorley = carica3D(WORLEY_LATO, gl.RGBA8, gl.RGBA, textureWorley(WORLEY_LATO));
    this.texCampo = gl.createTexture();
    if (this.volume) this.pubblica(this.volume);
  };

  StratoNubiFuse.prototype.onRemove = function () {
    const gl = this.gl;
    if (!gl) return;
    [this.texPerlin, this.texWorley, this.texCampo, this.tela].forEach(function (t) {
      if (t) gl.deleteTexture(t);
    });
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this.buffer) gl.deleteBuffer(this.buffer);
    gl.deleteProgram(this.programma);
    gl.deleteProgram(this.programmaStesura);
    this.gl = null;
    this.fbo = null;
    this.tela = null;
  };

  // La texture di fusione, con la sua piramide: da lontano si legge un
  // livello piu' grosso invece di campionare colonne piu' piccole del pixel.
  StratoNubiFuse.prototype.pubblica = function (volume) {
    this.volume = volume;
    const gl = this.gl;
    if (!gl) return;
    gl.bindTexture(gl.TEXTURE_2D, this.texCampo);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, volume.texture.larghezza, volume.texture.altezza,
      0, gl.RGBA, gl.UNSIGNED_BYTE, volume.texture.dati);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (this.mappa && this.mappa.triggerRepaint) this.mappa.triggerRepaint();
  };

  StratoNubiFuse.prototype.esagerazione = function () {
    const zoom = this.mappa && this.mappa.getZoom ? this.mappa.getZoom() : ESAGERAZIONE_ZOOM_PIENO;
    const calo = Math.pow(2, -Math.max(0, zoom - ESAGERAZIONE_ZOOM_PIENO) * ESAGERAZIONE_CALO_PER_ZOOM);
    return Math.max(ESAGERAZIONE_MINIMA, QUALITA.esagerazione * calo);
  };

  function matriceDi(argomenti) {
    if (!argomenti) return null;
    if (argomenti.length === 16) return argomenti;
    if (argomenti.defaultProjectionData && argomenti.defaultProjectionData.mainMatrix) {
      return argomenti.defaultProjectionData.mainMatrix;
    }
    return null;
  }

  StratoNubiFuse.prototype.preparaTela = function (gl) {
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    let scala = Math.min(1, Math.sqrt(QUALITA.pixel / Math.max(1, w * h)));
    if (this.mappa && this.mappa.isMoving && this.mappa.isMoving()) scala *= 0.66;
    const lw = Math.max(1, Math.round(w * scala)), lh = Math.max(1, Math.round(h * scala));
    if (this.fbo && this.misuraTela.larghezza === lw && this.misuraTela.altezza === lh) return true;
    if (!this.tela) this.tela = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tela);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, lw, lh, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (!this.fbo) this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tela, 0);
    const completo = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (!completo) {
      gl.deleteFramebuffer(this.fbo);
      this.fbo = null;
      this.senzaTela = true;
      return false;
    }
    this.misuraTela = { larghezza: lw, altezza: lh };
    return true;
  };

  StratoNubiFuse.prototype.prerender = function (gl, argomenti) {
    this.disegnatoInTela = false;
    if (!this.volume || !this.acceso || this.senzaTela) return;
    if (!this.preparaTela(gl)) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.misuraTela.larghezza, this.misuraTela.altezza);
    gl.disable(gl.SCISSOR_TEST);
    gl.colorMask(true, true, true, true);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.BLEND);
    this.disegnatoInTela = this.disegna(gl, argomenti, this.misuraTela.altezza);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  };

  StratoNubiFuse.prototype.render = function (gl, argomenti) {
    if (!this.volume || !this.acceso) return;
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.SCISSOR_TEST);
    gl.colorMask(true, true, true, true);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    // Colore gia' moltiplicato per l'opacita': composizione premoltiplicata.
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    if (this.disegnatoInTela) {
      gl.useProgram(this.programmaStesura);
      gl.bindVertexArray(this.vao);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.tela);
      gl.uniform1i(gl.getUniformLocation(this.programmaStesura, "uVolume"), 0);
      gl.uniform2f(gl.getUniformLocation(this.programmaStesura, "uTexel"),
        1 / this.misuraTela.larghezza, 1 / this.misuraTela.altezza);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
      return;
    }
    this.disegna(gl, argomenti, gl.drawingBufferHeight);
  };

  StratoNubiFuse.prototype.disegna = function (gl, argomenti, altezzaPixel) {
    const matrice = matriceDi(argomenti);
    if (!matrice) return false;
    const inversa = inverti4(matrice);
    if (!inversa) return false;
    const meta = this.volume.meta;
    const dom = dominioUnitario(meta.bounds);
    const esagerazione = this.esagerazione();
    const cosNord = Math.cos(Math.max(Math.abs(meta.bounds.north), Math.abs(meta.bounds.south))
      * Math.PI / 180);
    const zMax = (this.volume.cimaMassimaKm + 0.5) * esagerazione / (CIRCONFERENZA_KM * cosNord);

    // Il sole nello stesso istante del satellite, al centro della vista.
    const centro = this.mappa && this.mappa.getCenter ? this.mappa.getCenter() : null;
    const lat = centro ? clamp(centro.lat, meta.bounds.south, meta.bounds.north)
      : (meta.bounds.south + meta.bounds.north) / 2;
    const lon = centro ? clamp(centro.lng, meta.bounds.west, meta.bounds.east)
      : (meta.bounds.west + meta.bounds.east) / 2;
    const istante = Date.parse(meta.satelliteTime) || Date.now();
    const sole = posizioneSole(istante, lat, lon);
    const luce = luceDiScena(sole);

    const p = this.programma;
    gl.useProgram(p);
    gl.bindVertexArray(this.vao);
    const u = (nome) => gl.getUniformLocation(p, nome);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texCampo);
    gl.uniform1i(u("uCampo"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, this.texPerlin);
    gl.uniform1i(u("uPerlin"), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_3D, this.texWorley);
    gl.uniform1i(u("uWorley"), 2);

    gl.uniformMatrix4fv(u("uInversa"), false, inversa);
    gl.uniform4f(u("uDominio"), dom.x0, dom.yNord, dom.x1, dom.ySud);
    gl.uniform1f(u("uScalaKm"), meta.scaleKm || 16);
    gl.uniform1f(u("uCircKm"), CIRCONFERENZA_KM);
    gl.uniform1f(u("uEsagerazione"), esagerazione);
    gl.uniform1f(u("uZMax"), zMax);
    gl.uniform1f(u("uTexelCampo"), (dom.x1 - dom.x0) / this.volume.texture.larghezza);
    const tr = this.mappa && this.mappa.transform;
    const fov = (tr && (tr.fovInRadians || (tr.fov && tr.fov * Math.PI / 180))) || 0.6435;
    gl.uniform1f(u("uPixelAngolo"), 2 * Math.tan(fov / 2) / Math.max(1, altezzaPixel));
    gl.uniform1f(u("uPassoKm"), 0.3);
    gl.uniform1i(u("uPassi"), QUALITA.passi);
    gl.uniform1i(u("uPassiLuce"), QUALITA.passiLuce);
    gl.uniform1f(u("uSigma"), SIGMA_PER_KM);
    gl.uniform1f(u("uFaseG"), FASE_G);
    gl.uniform3fv(u("uSole"), new Float32Array(vettoreSole(sole)));
    gl.uniform3fv(u("uColoreSole"), new Float32Array(luce.colore));
    gl.uniform1f(u("uForzaSole"), luce.forza);
    gl.uniform3fv(u("uCielo"), new Float32Array(luce.cielo));
    gl.uniform3fv(u("uSuolo"), new Float32Array(luce.suolo));
    gl.uniform3fv(u("uFoschia"), new Float32Array(luce.foschia));
    gl.uniform1f(u("uPerlinKm"), PERLIN_KM);
    gl.uniform1f(u("uWorleyKm"), WORLEY_KM);
    gl.uniform1f(u("uLatoPerlin"), PERLIN_LATO);
    gl.uniform1f(u("uLatoWorley"), WORLEY_LATO);
    // Il seme appartiene al fotogramma: nubi osservate ferme non bollono.
    const seme = (istante / 600000) % 997;
    gl.uniform3f(u("uSemenza"), (seme * 0.137) % 1, (seme * 0.291) % 1, (seme * 0.419) % 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    return true;
  };

  // --- il comando ---------------------------------------------------------
  // Un solo strato per pagina. ``opzioni.avviso`` riceve i messaggi per
  // l'utente, ``opzioni.inclina`` accende o spegne la vista inclinata.
  function Controllo() {
    this.strato = null;
    this.attivo = false;
    this.seriale = 0;
    this.volume = null;
  }

  Controllo.prototype.attiva = function (mappa, opzioni) {
    const o = opzioni || {};
    const avviso = o.avviso || function () {};
    const gl = mappa && mappa.painter && mappa.painter.context && mappa.painter.context.gl;
    if (typeof WebGL2RenderingContext === "undefined" || !(gl instanceof WebGL2RenderingContext)) {
      avviso("Nubi 3D satellite + ICON-2I: serve WebGL2");
      return Promise.resolve(false);
    }
    if (typeof DecompressionStream === "undefined") {
      avviso("Nubi 3D satellite + ICON-2I: browser troppo vecchio per la texture");
      return Promise.resolve(false);
    }
    this.attivo = true;
    this.mappa = mappa;
    const seriale = ++this.seriale;
    if (!this.strato) this.strato = new StratoNubiFuse("nubi-fuse");
    this.strato.acceso = true;
    if (!mappa.getLayer(this.strato.id)) mappa.addLayer(this.strato);
    if (o.inclina) o.inclina(true);
    const self = this;
    return caricaVolume(o.sorgente).then(function (volume) {
      if (!self.attivo || seriale !== self.seriale) return false;
      self.volume = volume;
      self.strato.pubblica(volume);
      const m = volume.meta;
      const run = m.model && m.model.runTime ? m.model.runTime.slice(0, 13).replace("T", " ") + " UTC" : "?";
      const ora = m.satelliteTime.slice(11, 16);
      avviso("Nubi 3D · satellite " + ora + " UTC (maschera " + (m.mask === "clm" ? "CLM" : "IR")
        + ") · ambiente ICON-2I run " + run
        + (m.model && m.model.blend && m.model.blend.mode === "interpolated" ? " interpolato" : ""));
      return true;
    }).catch(function (errore) {
      if (!self.attivo || seriale !== self.seriale) return false;
      avviso("Nubi 3D satellite + ICON-2I non disponibili: " + errore.message);
      self.disattiva(o);
      return false;
    });
  };

  Controllo.prototype.disattiva = function (opzioni) {
    const o = opzioni || {};
    this.attivo = false;
    this.seriale++;
    if (this.strato) this.strato.acceso = false;
    if (this.mappa && this.strato && this.mappa.getLayer(this.strato.id)) {
      this.mappa.removeLayer(this.strato.id);
    }
    if (o.inclina) o.inclina(false);
    if (this.mappa) this.mappa.triggerRepaint();
  };

  Controllo.prototype.descrizione = function () {
    return this.volume ? this.volume.meta : null;
  };

  const api = {
    Controllo: Controllo,
    StratoNubiFuse: StratoNubiFuse,
    caricaVolume: caricaVolume,
    decodificaPng: decodificaPng,
    dominioUnitario: dominioUnitario,
    perlin3D: perlin3D,
    worley3D: worley3D,
    posizioneSole: posizioneSole,
    vettoreSole: vettoreSole,
    luceDiScena: luceDiScena,
    FRAMMENTO: FRAMMENTO,
    VERTICE: VERTICE,
    STESURA: STESURA
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  radice.NubiFusione = api;
})(typeof window !== "undefined" ? window : globalThis);
