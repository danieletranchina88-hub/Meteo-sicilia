/* Deterministic meteorological tool used by the local-downscaling agent.
 * No LLM-generated numbers. Experimental policy is supplied in the product. */
(function (root) {
  'use strict';
  function finite(x) { return typeof x === 'number' && Number.isFinite(x); }
  function distance(a,b) {
    const rad=Math.PI/180, dlat=(b.lat-a.lat)*rad, dlon=(b.lon-a.lon)*rad;
    const h=Math.sin(dlat/2)**2+Math.cos(a.lat*rad)*Math.cos(b.lat*rad)*Math.sin(dlon/2)**2;
    return 6371.0088*2*Math.asin(Math.sqrt(Math.min(1,Math.max(0,h))));
  }
  function evaluate(product, target, now=Date.now()) {
    const result={status:'unavailable',baseC:target.temperatureC,correctedC:null,stations:[]};
    function stop(reason) { result.reason=reason;return result; }
    if (!product || product.schemaVersion!==1 || !product.policy || product.runTime!==target.runTime)
      return stop('Prodotto locale assente o appartenente a un altro run.');
    if (!finite(target.temperatureC)||!finite(target.lat)||!finite(target.lon)||!finite(target.terrainM)||!finite(target.elevationM))
      return stop('Servono temperatura ICON, quota del modello e quota locale verificabile. Inserisci la quota o usa il rilievo 3D.');
    const p=product.policy;
    if (![p.maxAltitudeDeltaM,p.radiusKm,p.elevationToleranceM,p.minimumStations,p.decayHours,p.maxHorizonHours,p.maxObservationAgeHours,p.maximumSpreadC,p.maximumCorrectionC,p.lapseRateKPerM].every(finite)
        || p.radiusKm<=0 || p.decayHours<=0 || p.minimumStations<1) return stop('Parametri del prodotto non validi.');
    const dz=target.elevationM-target.terrainM;
    if (Math.abs(dz)>p.maxAltitudeDeltaM) return stop('Dislivello superiore al limite: la correzione richiede un profilo atmosferico locale.');
    // Physical estimate remains explicitly separate from observation-supported
    // correction. A standard lapse rate is an assumption, not a local profile.
    result.status='physical-only';
    result.altitudeCorrectionC=p.lapseRateKPerM*dz;
    result.residualCorrectionC=0;
    result.correctedC=target.temperatureC+result.altitudeCorrectionC;
    // Additional gates apply only to the observation residual.
    const valid=Date.parse(target.validTime), generated=Date.parse(product.generatedAt);
    if (!finite(valid)||!finite(generated)||generated>now||now-generated>p.maxObservationAgeHours*3600000)
      return stop('Analisi scaduta o orario non valido.');
    if (product.status!=='active-experimental') return stop(product.reason||'Controllo spaziale non superato.');
    const weighted=[], seen=new Set();
    for (const s of product.stations||[]) {
      if (!s.id||seen.has(s.id)) continue;
      seen.add(s.id);
      if (![s.lat,s.lon,s.elevationM,s.obsTime,s.residualC].every(finite)) continue;
      const age=(now-s.obsTime*1000)/3600000, lead=(valid-s.obsTime*1000)/3600000;
      if (age<0||age>p.maxObservationAgeHours||lead<0||lead>p.maxHorizonHours) continue;
      const d=distance(target,s), zd=Math.abs(target.elevationM-s.elevationM);
      if (d>=p.radiusKm||zd>p.elevationToleranceM) continue;
      const w=(1-(d/p.radiusKm)**2)**2*Math.exp(-((zd/150)**2));
      const decay=Math.exp(-lead/p.decayHours)*(1-lead/p.maxHorizonHours);
      weighted.push({s,w,decay,d});
    }
    if (weighted.length<p.minimumStations) return stop('Stazioni recenti e rappresentative insufficienti entro '+p.radiusKm+' km.');
    const total=weighted.reduce((a,x)=>a+x.w,0);
    if (!(total>0)) return stop('Supporto locale insufficiente.');
    const mean=weighted.reduce((a,x)=>a+x.w*x.s.residualC,0)/total;
    const spread=Math.sqrt(weighted.reduce((a,x)=>a+x.w*(x.s.residualC-mean)**2,0)/total);
    if (spread>p.maximumSpreadC) return stop('Stazioni discordanti: possibile discontinuità meteorologica, correzione sospesa.');
    const residual=weighted.reduce((a,x)=>a+x.w*x.s.residualC*x.decay,0)/(1+total);
    const adjustment=p.lapseRateKPerM*dz+residual;
    if (Math.abs(adjustment)>p.maximumCorrectionC) return stop('Correzione oltre il limite prudenziale.');
    return Object.assign(result,{status:'experimental', correctedC:target.temperatureC+adjustment,
      altitudeCorrectionC:p.lapseRateKPerM*dz,residualCorrectionC:residual,spreadC:spread,
      reason:'Gradiente standard assunto; controllo spaziale, senza validazione temporale indipendente.',
      stations:weighted.map(x=>({id:x.s.id,distanceKm:x.d,observedC:x.s.observedC,
        modelC:x.s.modelC,obsTime:x.s.obsTime,weight:x.w,decay:x.decay}))});
  }
  const api={evaluate,distance};
  if (typeof module!=='undefined'&&module.exports) module.exports=api;
  else root.MeteoLocalDownscaling=api;
})(typeof globalThis!=='undefined'?globalThis:this);
