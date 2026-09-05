const assert=require('assert');
const {evaluate}=require('../../local_downscaling.js');
const now=Date.parse('2026-09-05T01:00:00Z');
const policy={radiusKm:40,elevationToleranceM:250,maxAltitudeDeltaM:300,maxObservationAgeHours:2,maxHorizonHours:6,decayHours:2,minimumStations:3,maximumSpreadC:2,maximumCorrectionC:3,lapseRateKPerM:-.0065};
const product={schemaVersion:1,runTime:'2026-09-05T00:00:00Z',generatedAt:new Date(now).toISOString(),status:'active-experimental',policy,
 stations:Array.from({length:6},(_,i)=>({id:'S'+i,lat:38+i*.015,lon:13+i*.01,elevationM:100,obsTime:now/1000,residualC:2,modelC:22,observedC:24}))};
const target={lat:38,lon:13,elevationM:100,terrainM:100,temperatureC:22,runTime:product.runTime,validTime:new Date(now).toISOString()};
let r=evaluate(product,target,now);assert.equal(r.status,'experimental');assert(r.correctedC>22&&r.correctedC<24);
const immediate=r.residualCorrectionC;
r=evaluate(product,{...target,validTime:new Date(now+3*3600000).toISOString()},now);assert(r.residualCorrectionC<immediate);
for(const delta of [-1,7*3600000])assert.equal(evaluate(product,{...target,validTime:new Date(now+delta).toISOString()},now).status,'physical-only');
assert.equal(evaluate(product,target,now+7200001).status,'physical-only');
assert.equal(evaluate(product,{...target,runTime:'old'},now).status,'unavailable');
assert.equal(evaluate(product,{...target,elevationM:null},now).status,'unavailable');
assert.equal(evaluate(product,{...target,elevationM:900},now).status,'unavailable');
assert.equal(evaluate(product,{...target,lat:42},now).status,'physical-only');
assert.equal(evaluate({...product,status:'withheld'},target,now).status,'physical-only');
assert.equal(evaluate({...product,stations:product.stations.slice(0,2)},target,now).status,'physical-only');
assert.equal(evaluate({...product,stations:product.stations.map((s,i)=>({...s,residualC:i%2?5:-5}))},target,now).status,'physical-only');
console.log('PASS: numerical output, decay, past/future, stale, mismatched run, missing altitude, sparse coverage and disagreement');

assert.equal(evaluate({...product,status:'withheld'},{...target,elevationM:200},now).correctedC,21.35);
