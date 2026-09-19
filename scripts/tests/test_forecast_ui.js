'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {ResourceCache, FrameCache} = require('../../forecast-cache.js');
const html = fs.readFileSync(path.join(__dirname,'../../index.html'),'utf8');
function implementation(name) {
  const re = new RegExp('      (?:async )?function '+name+'\\(');
  const start = html.search(re); assert(start >= 0, name);
  return html.slice(start,html.indexOf('\n      }',start)+8);
}
function deferred() { let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject}; }
async function test(name,fn) {await fn();console.log('PASS '+name);}
(async()=>{
 await test('deduplicates prefetch and foreground requests',async()=>{
  const cache=new ResourceCache(), gate=deferred();let requests=0;
  const loader=()=>{requests++;return gate.promise;};
  const a=cache.load('h1',loader),b=cache.load('h1',loader);assert.equal(a,b);
  gate.resolve({temperature:22});await a;assert.equal(requests,1);
  assert.equal((await cache.load('h1',loader)).temperature,22);assert.equal(requests,1);
 });
 await test('failed request can retry without a poisoned promise',async()=>{
  const cache=new ResourceCache();await assert.rejects(cache.load('a',()=>Promise.reject(Error('offline'))));
  assert.equal(await cache.load('a',()=>23),23);
 });
 await test('late response from previous run cannot repopulate the new cache',async()=>{
  const cache=new ResourceCache(),gate=deferred();const a=cache.load('0',()=>gate.promise);
  await Promise.resolve();cache.clear();await cache.load('0',()=>99);gate.resolve(12);await a;
  assert.equal(cache.get('0'),99);
 });
 await test('bounded LRU keeps recent frames and accounts for replacements',()=>{
  const cache=new FrameCache(12);const frame=()=>({pixels:new Uint8ClampedArray(4)});
  cache.set('a',frame());cache.set('b',frame());cache.set('c',frame());cache.get('a');cache.set('d',frame());
  assert.equal(cache.has('b'),false);assert.equal(cache.bytes,12);cache.set('d',frame());assert.equal(cache.bytes,12);
  cache.clear();assert.equal(cache.bytes,0);
 });
 await test('fast scrubbing commits only the newest response and matching hour',async()=>{
  const first=deferred(),second=deferred(),commits=[];
  const context={catalog:[{},{}],loadingToken:0,currentData:{},currentIndex:0,weatherView:'forecast',
    explicitLoadInFlight:false,rasterRenderToken:0,pendingRasterCallback:null,lastRasterSignature:"",console,document:{getElementById:()=>({})},
    activeExtraKind:()=>null,fetchStep:i=>i===0?first.promise:second.promise,
    ui:{slider:{}},showParticles:false,renderWeather:(_,done)=>done(),
    updateTimeUi:()=>commits.push(context.currentIndex),toast:()=>{},showLoading:()=>{},hideLoading:()=>{}};
  for(const name of ['updateLayerUi','updateBulletinUi','updateLegend','updateIsobars','updateMapPresentation','requestVectorRender','refreshSelectedPoint','prefetchWholeRun','updateBufferUi','scheduleFrameWarmup'])context[name]=()=>{};
  vm.createContext(context);vm.runInContext(implementation('loadStep'),context);
  const a=context.loadStep(0),b=context.loadStep(1);second.resolve({temp:2});await b;first.resolve({temp:1});await a;
  assert.deepEqual(commits,[1]);assert.equal(context.currentData.temp,2);
 });
 await test('changing field while an hour downloads discards the obsolete request',async()=>{
  const gate=deferred();let kind=null,commits=0;
  const ctx={catalog:[{}],loadingToken:0,currentData:{temp:1},currentIndex:0,weatherView:'forecast',console,
    rasterRenderToken:0,pendingRasterCallback:null,lastRasterSignature:'',document:{getElementById:()=>({})},activeExtraKind:()=>kind,fetchStep:()=>gate.promise,
    hideLoading:()=>{},toast:()=>{},renderWeather:()=>commits++};
  vm.createContext(ctx);vm.runInContext(implementation('loadStep'),ctx);
  const a=ctx.loadStep(0);kind='upper';gate.resolve({temp:99});await a;
  assert.equal(commits,0);assert.equal(ctx.currentData.temp,1);
 });
 // Il commit del passo -- etichette del run, legenda, isobare, avvio del
 // prefetch -- appartiene al PASSO, non al raster che lo ha chiesto. Era
 // invece appeso a pendingRasterCallback, che ogni nuova richiesta di disegno
 // sovrascrive: bastava un secondo renderWeather prima che il primo tornasse
 // dal worker perche' il commit sparisse per sempre. Con le tessere del
 // terreno che ora arrivano subito e chiedono un ridisegno, quel secondo
 // renderWeather e' la norma, e il sito restava fermo a "In attesa dei dati"
 // con la legenda vuota. Misurato sulla build pubblicata: run "In attesa dei
 // dati", validita' "--", legenda vuota; con la correzione, "gio 17 set ·
 // 08:00", "+6 h" e la legenda completa.
 await test('il commit del passo sopravvive a un secondo disegno che lo supera',()=>{
  const ctx={pendingRenderReady:null};
  vm.createContext(ctx);
  vm.runInContext(implementation('commitRenderReady'),ctx);
  let commits=0;
  // Primo disegno: deposita il commit.
  ctx.pendingRenderReady=()=>commits++;
  // Secondo disegno che lo supera: con il difetto la callback veniva
  // sostituita, e il commit non veniva mai eseguito. Qui nessuno la tocca.
  ctx.commitRenderReady();
  assert.equal(commits,1,'il commit del passo non e\' stato eseguito');
  // Una volta sola: un ridisegno successivo non deve rifare il commit.
  ctx.commitRenderReady();
  assert.equal(commits,1,'il commit del passo e\' stato eseguito due volte');
 });

 // La stessa invariante sul sorgente: renderWeather deve depositare il commit
 // PRIMA di qualunque uscita anticipata, altrimenti un disegno saltato per una
 // guardia lo perde comunque.
 await test('renderWeather deposita il commit prima delle uscite anticipate',()=>{
  const body=implementation('renderWeather');
  const store=body.indexOf('pendingRenderReady = onReady');
  assert.ok(store>=0,'renderWeather non deposita piu\' il commit del passo');
  assert.ok(store<body.indexOf('return;'),
   'il commit viene depositato dopo la prima uscita anticipata');
  assert.ok(!/if\s*\(\s*onReady\s*\)\s*onReady\(\)/.test(body),
   'renderWeather chiama ancora direttamente la callback di un singolo raster');
  assert.ok(/commitRenderReady\(\)/.test(body),
   'renderWeather non usa piu\' il commit condiviso');
  // I disegni anticipati non commettono niente: scaldano la cache mentre
  // currentData e' temporaneamente quello di un'altra scadenza.
  assert.ok(/if \(!prewarm && onReady\) pendingRenderReady = onReady;/.test(body),
   'anche un disegno anticipato deposita il commit di un altro passo');
 });

 await test('raster worker coalesces obsolete jobs instead of queuing every hour',()=>{
  const sent=[],context={rasterRenderToken:0,rasterWorkerSupported:true,rasterWorkerBusy:false,
    queuedRasterParams:null,rasterWorker:{postMessage:p=>sent.push(p)},pendingRasterCallback:null,
    rasterWorkerState:{terrainKey:''},rasterMessage:(p)=>p};
  vm.createContext(context);vm.runInContext(implementation('dispatchRasterFill'),context);
  context.dispatchRasterFill({hour:1},()=>{});context.dispatchRasterFill({hour:2},()=>{});context.dispatchRasterFill({hour:3},()=>{});
  assert.equal(sent.length,1);assert.equal(context.queuedRasterParams.hour,3);
 });
 // La sorgente canvas eviterebbe la codifica PNG, ma in MapLibre 5.24 non
 // carica la texture da un canvas fuori documento: misurato in Chromium,
 // "InvalidStateError: The source image could not be decoded" a ogni
 // aggiornamento e mappa nera. Il raster si pubblica come immagine, e ogni
 // pubblicazione deve annullare quella precedente: senza il contatore, due
 // codifiche in volo possono consegnare i fotogrammi in ordine invertito e
 // lasciare sulla mappa l'ora sbagliata.
 await test('il raster si pubblica come immagine e la pubblicazione vecchia viene annullata',async()=>{
  const aggiornamenti=[],revocati=[];
  const source={updateImage:o=>aggiornamenti.push(o.url)};
  let daChiamare=null;
  const ctx={map:{getSource:()=>source},console,setTimeout:fn=>fn(),
   URL:{createObjectURL:b=>'blob:'+b.tag,revokeObjectURL:u=>revocati.push(u)},
   rasterCanvas:{toBlob:(fn)=>{daChiamare=fn;},toDataURL:()=>'data:,x'},
   rasterPublishToken:0,rasterObjectUrl:'',lastRasterSignature:'firma'};
  vm.createContext(ctx);vm.runInContext(implementation('publishWeatherRaster'),ctx);
  ctx.publishWeatherRaster([[0,1],[1,1],[1,0],[0,0]]);
  const primaCodifica=daChiamare;
  ctx.publishWeatherRaster([[0,1],[1,1],[1,0],[0,0]]);
  const secondaCodifica=daChiamare;
  // La prima codifica termina per ultima: non deve arrivare sulla mappa.
  secondaCodifica({tag:'nuovo'});primaCodifica({tag:'vecchio'});
  assert.deepEqual(aggiornamenti,['blob:nuovo']);
  assert.deepEqual(revocati,['blob:vecchio']);
 });
 await test('generated decoder worker is self-contained, including derived fields',async()=>{
  const blobs=[];const Worker=function(){this.postMessage=()=>{};};
  const ctx={console,Worker,Blob:class {constructor(parts){blobs.push(parts.join(''));}},URL:{createObjectURL:()=>'',revokeObjectURL:()=>{}},
    dataWorker:null,dataWorkerFailed:false,dataRequestId:0,dataRequests:new Map(),location:{href:'https://test.example/'},
    FEELS_LIKE_METHOD:'heat-index-wind-chill-v1',BINARY_STEP_MAGIC:'MSB1',BINARY_STEP_NODATA:-32768};
  // URL must support both URL construction and blob helpers.
  ctx.URL=class extends URL{static createObjectURL(){return 'blob:test';}static revokeObjectURL(){}};
  vm.createContext(ctx);
  for(const fn of ['clamp','getGrid','fetchDecompressedResponse','decodeBinaryStep','saturationVapourHpa','wetBulbCelsius','deriveWetBulb','heatIndexCelsius','windChillCelsius','calculateFeelsLike','prepareData','requestPreparedStep'])vm.runInContext(implementation(fn),ctx);
  ctx.requestPreparedStep('step.bin.gz');assert.equal(blobs.length,1);
  const worker={self:{},console,Set,ArrayBuffer,Float32Array,Int16Array,DataView,TextDecoder};
  vm.createContext(worker);vm.runInContext(blobs[0],worker);
  const value=vm.runInContext('wetBulbCelsius(20,100,1013.25)',worker);assert(Math.abs(value-20)<.001);
 });
 await test('warm frames reuse the exact raster pixels without computing or publishing during prewarm',()=>{
  let computes=0,publishes=0,paints=0;
  const ctx={console,currentIndex:0,catalog:Array(3).fill({}),activeLayer:'temp',selectedLevel:'surface',activeModel:'icon2i',
    weatherView:'forecast',mapLoaded:true,showFusion:false,show3D:false,useNearestCell:false,
    currentData:{meta:{nx:2,ny:2,lo1:10,la1:40,dx:1,dy:1,validTime:'test'},temp:new Float32Array([10,20,30,40])},
    window:{devicePixelRatio:1},weatherFrameEpoch:0,lastRasterSignature:'',terrainCoverageStamp:0,rasterRenderToken:0,pendingRasterCallback:null,pendingRenderReady:null,
    weatherFrames:new FrameCache(1000000),rasterCanvas:{},rasterContext:{putImageData:()=>paints++},
    ImageData:class {constructor(p,w,h){this.data=p;}},isMobile:()=>false,
    RASTER_CONFIG:{padding:.2,maxDensityDesktop:1,pixelLimitDesktop:100},
    activeLayerInfo:()=>({stops:[]}),usesUpperData:()=>false,usesStormData:()=>false,usesProbData:()=>false,usesProfileData:()=>false,
    elevationDownscalingActive:()=>false,getGrid:x=>x,applyWeatherPaint:()=>{},scheduleFrameWarmup:()=>{},
    publishWeatherRaster:()=>publishes++,map:{getSource:()=>({}),getBounds:()=>({getWest:()=>9,getEast:()=>12,getSouth:()=>38,getNorth:()=>41}),getPitch:()=>0,getContainer:()=>({clientWidth:10,clientHeight:10})}};
  ctx.dispatchRasterFill=ctx.dispatchWarmRaster=(p,done)=>{computes++;done(new Uint8ClampedArray(p.width*p.height*4).fill(17));};
  vm.createContext(ctx);
  vm.runInContext(implementation('commitRenderReady'),ctx);
  vm.runInContext(implementation('renderWeather'),ctx);
  ctx.renderWeather(true);assert.equal(computes,1);assert.equal(publishes,0);assert.equal(paints,0);
  let commits=0;
  ctx.renderWeather(false,()=>commits++);assert.equal(computes,1);assert.equal(publishes,1);assert.equal(paints,1);
  assert.equal(commits,1,'il disegno dalla cache non commette il passo');
 });
 await test('satellite view clears overlays and restores the chosen field and forecast hour',()=>{
  const classes=new Set();const element={hidden:true,setAttribute:()=>{}};const visibilita={};
  const ctx={console,weatherView:'forecast',mapLoaded:true,currentIndex:13,activeLayer:'wind',synopticChart:false,
    loadingToken:0,rasterRenderToken:0,pendingRasterCallback:null,forecastRestore:null,
    document:{getElementById:()=>element,body:{classList:{toggle:(c,v)=>{if(v)classes.add(c);else classes.delete(c);}}}},
    map:{setLayoutProperty:(id,k,value)=>{visibilita[id]=value;},getLayer:()=>({})},
    showParticles:false,showVectors:true,showIsobars:true,showIsotherms:false,showIsohypses:false,
    showFronts:true,showFusion:false,showStations:false,showTerrain:false,showSatellite:false,show3D:false,showGraticule:false,
    showLightning:true,showRadar:true,showLiveLightning:true,liveStrikes:[{},{}],cloudTimeSelected:123456};
  for(const name of ['setPlaying','updateTerrain3D','updateSatelliteBase','updateSatelliteClouds','updateLightningLayer','updateSatelliteControlsVisibility','updateLayerUi','updateLegend','setDrawer','updateMapPresentation','updateIsobars','updateStationMarkers','renderWeather','requestVectorRender','updateTimeUi','updateBufferUi','scheduleFrameWarmup','stopStrikeAnimation'])ctx[name]=()=>{};
  let chiusure=0;ctx.blitzDisconnect=()=>chiusure++;
  ctx.clearMeteorologicalLayers=()=>{ctx.showVectors=false;ctx.showFronts=false;ctx.showIsobars=false;};
  vm.createContext(ctx);
  const source=fs.readFileSync(path.join(__dirname,'../../modern-ui.js'),'utf8');
  vm.runInContext(source.slice(source.indexOf('function setWeatherView'),source.indexOf('(function modernControls')),ctx);
  ctx.setWeatherView('satellite');assert.equal(ctx.showVectors,false);assert.equal(ctx.showFronts,false);assert.equal(ctx.showSatelliteClouds,true);
  ctx.setWeatherView('forecast');assert.equal(ctx.activeLayer,'wind');assert.equal(ctx.currentIndex,13);assert.equal(ctx.showVectors,true);assert.equal(ctx.showFronts,true);
  // I due livelli osservati vivono solo nella vista satellite, e l'ora scelta
  // con loro: rientrando nella previsione devono spegnersi e tornare in
  // diretta, altrimenti al giro dopo si riaprirebbe il satellite su
  // un'immagine vecchia che nessuno ha piu' chiesto.
  assert.equal(ctx.showSatelliteClouds,false,'le nubi restano accese in previsione');
  assert.equal(ctx.showLightning,false,'i fulmini restano accesi in previsione');
  assert.equal(ctx.cloudTimeSelected,0,"l'ora dell'osservato non torna in diretta");
  // Radar e fulmini in diretta hanno il comando SOLO nel pannello del
  // satellite: se restassero accesi tornando alla previsione, resterebbero
  // sulla mappa senza piu' alcun modo di spegnerli.
  assert.equal(ctx.showRadar,false,'il radar resta acceso in previsione');
  assert.equal(visibilita['radar-layer'],'none','il livello radar resta visibile in previsione');
  assert.equal(ctx.showLiveLightning,false,'la diretta resta accesa in previsione');
  assert.equal(ctx.liveStrikes.length,0,'le scariche restano in memoria');
  assert.equal(chiusure,1,'la connessione al flusso resta aperta senza nessuno che guardi');
 });

 await test('i comandi osservati stanno nel pannello del satellite e sono sovrapponibili',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../../index.html'),'utf8');
  const markup=html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,'');
  const pannello=markup.slice(markup.indexOf('id="satellite-status"'),
    markup.indexOf('</section>',markup.indexOf('id="satellite-status"')));
  // Erano sparsi nel cassetto dei campi previsti, dove accenderne uno faceva
  // uscire dalla vista satellite: il comando c'era ma non si poteva usare
  // insieme all'immagine.
  for(const nome of ['satimage','radar','livelightning','lightning','strikesound']) {
    assert(pannello.includes('data-toggle="'+nome+'"'),
      'il comando '+nome+' non e\' nel pannello del satellite');
    assert.equal(markup.split('data-toggle="'+nome+'"').length-1,1,
      'il comando '+nome+' compare piu\' volte: due interruttori per lo stesso livello');
  }
  // E nessuno dei tre deve piu' far uscire dalla vista quando lo si accende.
  const uscita=html.slice(html.indexOf('const SATELLITE_TOGGLES'),
    html.indexOf('function setToggle'));
  for(const nome of ['satimage','radar','lightning','livelightning','strikesound'])
    assert(uscita.includes('"'+nome+'"'),nome+' fa ancora uscire dalla vista satellite');
  // L'ordine dei livelli sulla mappa: il radar sopra le nubi e sopra i
  // fulmini da satellite, altrimenti la pioggia sparisce sotto la coltre.
  const nubi=html.indexOf('id: "satellite-clouds-layer"');
  const fulmini=html.indexOf('id: "satellite-lightning-layer"');
  const inchiostro=html.indexOf('id: "base-rivers-layer"');
  assert(nubi<fulmini && fulmini<inchiostro,
    'i fulmini da satellite non stanno piu\' sopra le nubi');
  assert(html.includes('firstLabelLayerId()'),
    'il radar non viene piu\' inserito prima del primo livello di inchiostro');
 });
 await test('il pannello dell\'osservato si richiude e si ricorda la scelta',()=>{
  // Aperto misura 327 px: su un telefono da 844 ne prende il quaranta per
  // cento, e della mappa resta una striscia. Chiuso scende sotto i settanta.
  // Qui si prova il comportamento, non l'aspetto: chi decide lo stato
  // iniziale, chi lo ricorda, e che cosa racconta la barra quando e' chiusa.
  function ambiente(larghezza, memoria) {
    const nodi = {};
    const fai = (id) => (nodi[id] = nodi[id] || {
      id, textContent: '', classi: new Set(), attributi: {},
      classList: {
        toggle(c, v) { if (v) nodi[id].classi.add(c); else nodi[id].classi.delete(c); },
        contains: (c) => nodi[id].classi.has(c)
      },
      setAttribute(k, v) { nodi[id].attributi[k] = v; }
    });
    const ctx = {
      console, MOBILE_BREAKPOINT: 960,
      window: { innerWidth: larghezza },
      document: { getElementById: (id) => fai(id) },
      localStorage: {
        dati: Object.assign({}, memoria),
        getItem(k) { return k in this.dati ? this.dati[k] : null; },
        setItem(k, v) { this.dati[k] = String(v); }
      },
      showSatelliteClouds: true, showRadar: false,
      showLiveLightning: false, showLightning: false,
      cloudProduct: 'geocolour',
      CLOUD_PRODUCTS: { geocolour: { name: 'GeoColour · reale di giorno, IR di notte' } }
    };
    vm.createContext(ctx);
    vm.runInContext(implementation('isMobile'), ctx);
    // Il blocco di stato sta fuori da una funzione: si estrae a parte.
    const stato = html.slice(html.indexOf('const SAT_PANNELLO_KEY'),
                             html.indexOf('      function setSatellitePannello'));
    vm.runInContext(stato, ctx);
    vm.runInContext(implementation('setSatellitePannello'), ctx);
    vm.runInContext(implementation('aggiornaRiassuntoSatellite'), ctx);
    // Le dichiarazioni con let non diventano proprieta' dell'oggetto globale
    // del contesto: il valore si legge valutando l'espressione, non
    // guardando ctx.
    return { ctx, nodi: fai, leggi: (e) => vm.runInContext(e, ctx) };
  }

  // Senza scelta memorizzata: chiuso sul telefono, aperto sul desktop.
  const tel = ambiente(390, {});
  assert.equal(tel.leggi('satellitePannelloAperto'), false,
    'sul telefono il pannello si apre da solo e ricopre la mappa');
  const desk = ambiente(1440, {});
  assert.equal(desk.leggi('satellitePannelloAperto'), true,
    'sul desktop il pannello parte chiuso, dove non dava fastidio a nessuno');

  // La scelta di chi guarda vince sul valore predefinito, su entrambi.
  assert.equal(ambiente(390, {'meteo.pannelloSatellite': '1'}).leggi('satellitePannelloAperto'), true,
    'chi ha aperto i comandi sul telefono se li ritrova chiusi');
  assert.equal(ambiente(1440, {'meteo.pannelloSatellite': '0'}).leggi('satellitePannelloAperto'), false,
    'chi ha chiuso i comandi sul desktop se li ritrova aperti');

  // Chiudere marca la sezione e lo dice alle tecnologie assistive: la
  // freccia e' l'unico invito ad aprire, e per chi non la vede aria-expanded
  // e' l'unico modo di sapere che c'e' qualcosa sotto.
  const a = ambiente(1440, {});
  a.ctx.setSatellitePannello(false, true);
  assert.equal(a.nodi('satellite-status').classList.contains('sat-chiuso'), true);
  assert.equal(a.nodi('satellite-fold').attributi['aria-expanded'], 'false');
  assert.equal(a.ctx.localStorage.dati['meteo.pannelloSatellite'], '0',
    'la scelta non viene memorizzata');
  a.ctx.setSatellitePannello(true, true);
  assert.equal(a.nodi('satellite-status').classList.contains('sat-chiuso'), false);
  assert.equal(a.nodi('satellite-fold').attributi['aria-expanded'], 'true');

  // Chi apre il pannello senza chiederlo -- il rientro nella vista -- non
  // deve sovrascrivere la scelta memorizzata.
  const b = ambiente(1440, {'meteo.pannelloSatellite': '0'});
  b.ctx.setSatellitePannello(b.leggi('satellitePannelloAperto'), false);
  assert.equal(b.ctx.localStorage.dati['meteo.pannelloSatellite'], '0',
    'rientrare nella vista riscrive la scelta di chi guarda');

  // Chiuso, la barra deve dire cosa resta acceso sotto, o chiuderla
  // vorrebbe dire perdere di vista quello che si sta guardando.
  const c = ambiente(390, {});
  c.ctx.showRadar = true; c.ctx.showLiveLightning = true;
  c.ctx.aggiornaRiassuntoSatellite();
  assert.equal(c.nodi('satellite-riassunto').textContent,
    'GeoColour · radar · fulmini in diretta',
    'il riassunto non elenca i livelli accesi');
  // Solo il nome del canale: la descrizione che lo segue non ci sta.
  assert.ok(!c.nodi('satellite-riassunto').textContent.includes('reale di giorno'));
  c.ctx.showSatelliteClouds = false; c.ctx.showRadar = false;
  c.ctx.showLiveLightning = false;
  c.ctx.aggiornaRiassuntoSatellite();
  assert.match(c.nodi('satellite-riassunto').textContent, /tocca per aprire/,
    'a pannello chiuso e livelli spenti la barra non dice piu\' nulla, e '
    + 'nessuno capisce che si puo\' riaprire');
 });

 await test('toccare un interruttore non richiude il pannello sotto le dita',()=>{
  // Il modo silenzioso in cui questa cosa si rompe: tutta l'intestazione e'
  // il bersaglio del tocco, e senza l'esclusione dei comandi ogni volta che
  // si accende il radar il pannello si chiuderebbe da solo.
  const gancio = html.slice(html.indexOf('(function pannelloOsservato()'),
                            html.indexOf('})();', html.indexOf('(function pannelloOsservato()')));
  assert.ok(gancio.length, 'manca il gancio del pannello');
  assert.match(gancio, /if \(event\.target\.closest\("#satellite-controls"\)\) return;/,
    'un tocco su un interruttore richiude il pannello');
  assert.match(gancio, /event\.stopPropagation\(\);/,
    'il tocco sulla freccia arriva anche alla riga e il pannello si riapre subito');
  // Selezionare l'ora non e' chiedere di chiudere.
  assert.match(gancio, /getSelection/,
    'selezionare del testo nella barra la richiude');
  // E il CSS deve davvero nascondere i comandi, non solo marcare la classe.
  const css = fs.readFileSync(path.join(__dirname,'../../modern-ui.css'),'utf8');
  assert.match(css, /\.sat-chiuso #satellite-controls \{ display: none !important; \}/,
    'la classe c\'e\' ma i comandi restano visibili');
  assert.match(css, /\.sat-chiuso #satellite-riassunto \{/,
    'il riassunto non compare a pannello chiuso');
 });

 await test('page IDs are unique and new runtime assets are deployed',()=>{
  for(const file of ['index.html','meteograms.html']) {
    const source=fs.readFileSync(path.join(__dirname,'../..',file),'utf8');
    const markup=source.replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,'');
    const ids=[...markup.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);assert.equal(ids.length,new Set(ids).size,file);
  }
  const workflow=fs.readFileSync(path.join(__dirname,'../../.github/workflows/update_meteo.yml'),'utf8');
  for(const asset of ['forecast-cache.js','modern-ui.js','modern-ui.css','meteograms-ui.css'])assert(workflow.includes(asset),asset);
 });
})().catch(error=>{console.error(error);process.exitCode=1;});
