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
 await test('raster worker coalesces obsolete jobs instead of queuing every hour',()=>{
  const sent=[],context={rasterRenderToken:0,rasterWorkerSupported:true,rasterWorkerBusy:false,
    queuedRasterParams:null,rasterWorker:{postMessage:p=>sent.push(p)},pendingRasterCallback:null};
  vm.createContext(context);vm.runInContext(implementation('dispatchRasterFill'),context);
  context.dispatchRasterFill({hour:1},()=>{});context.dispatchRasterFill({hour:2},()=>{});context.dispatchRasterFill({hour:3},()=>{});
  assert.equal(sent.length,1);assert.equal(context.queuedRasterParams.hour,3);
 });
 await test('canvas publishing skips PNG encoding and stops continuous GPU uploads',()=>{
  const calls=[];const source={setCoordinates:()=>calls.push('coords'),play:()=>calls.push('play'),pause:()=>calls.push('pause')};
  const ctx={map:{getSource:()=>source,once:(event,fn)=>fn(),triggerRepaint:()=>calls.push('repaint')}};
  vm.createContext(ctx);vm.runInContext(implementation('publishWeatherRaster'),ctx);ctx.publishWeatherRaster([]);
  assert.deepEqual(calls,['coords','play','pause','repaint']);
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
    window:{devicePixelRatio:1},weatherFrameEpoch:0,lastRasterSignature:'',terrainSamplerKey:'',rasterRenderToken:0,pendingRasterCallback:null,
    weatherFrames:new FrameCache(1000000),rasterCanvas:{},rasterContext:{putImageData:()=>paints++},
    ImageData:class {constructor(p,w,h){this.data=p;}},isMobile:()=>false,
    RASTER_CONFIG:{padding:.2,maxDensityDesktop:1,pixelLimitDesktop:100},
    activeLayerInfo:()=>({stops:[]}),usesUpperData:()=>false,usesStormData:()=>false,usesProbData:()=>false,usesProfileData:()=>false,
    elevationDownscalingActive:()=>false,getGrid:x=>x,applyWeatherPaint:()=>{},scheduleFrameWarmup:()=>{},
    publishWeatherRaster:()=>publishes++,map:{getSource:()=>({}),getBounds:()=>({getWest:()=>9,getEast:()=>12,getSouth:()=>38,getNorth:()=>41}),getPitch:()=>0,getContainer:()=>({clientWidth:10,clientHeight:10})}};
  ctx.dispatchRasterFill=ctx.dispatchWarmRaster=(p,done)=>{computes++;done(new Uint8ClampedArray(p.width*p.height*4).fill(17));};
  vm.createContext(ctx);vm.runInContext(implementation('renderWeather'),ctx);
  ctx.renderWeather(true);assert.equal(computes,1);assert.equal(publishes,0);assert.equal(paints,0);
  ctx.renderWeather();assert.equal(computes,1);assert.equal(publishes,1);assert.equal(paints,1);
 });
 await test('satellite view clears overlays and restores the chosen field and forecast hour',()=>{
  const classes=new Set();const element={hidden:true,setAttribute:()=>{}};
  const ctx={console,weatherView:'forecast',mapLoaded:true,currentIndex:13,activeLayer:'wind',synopticChart:false,
    loadingToken:0,rasterRenderToken:0,pendingRasterCallback:null,forecastRestore:null,
    document:{getElementById:()=>element,body:{classList:{toggle:(c,v)=>{if(v)classes.add(c);else classes.delete(c);}}}},
    map:{setLayoutProperty:()=>{}},showParticles:false,showVectors:true,showIsobars:true,showIsotherms:false,showIsohypses:false,
    showFronts:true,showFusion:false,showStations:false,showTerrain:false,showSatellite:false,show3D:false,showGraticule:false};
  for(const name of ['setPlaying','updateTerrain3D','updateSatelliteBase','updateSatelliteClouds','updateLayerUi','updateLegend','setDrawer','updateMapPresentation','updateIsobars','updateStationMarkers','renderWeather','requestVectorRender','updateTimeUi','updateBufferUi','scheduleFrameWarmup'])ctx[name]=()=>{};
  ctx.clearMeteorologicalLayers=()=>{ctx.showVectors=false;ctx.showFronts=false;ctx.showIsobars=false;};
  vm.createContext(ctx);
  const source=fs.readFileSync(path.join(__dirname,'../../modern-ui.js'),'utf8');
  vm.runInContext(source.slice(source.indexOf('function setWeatherView'),source.indexOf('(function modernControls')),ctx);
  ctx.setWeatherView('satellite');assert.equal(ctx.showVectors,false);assert.equal(ctx.showFronts,false);assert.equal(ctx.showSatelliteClouds,true);
  ctx.setWeatherView('forecast');assert.equal(ctx.activeLayer,'wind');assert.equal(ctx.currentIndex,13);assert.equal(ctx.showVectors,true);assert.equal(ctx.showFronts,true);
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
