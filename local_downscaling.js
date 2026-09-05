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
  // Reduce model MSLP into observation space before comparing it with true
  // station pressure. The residual is then applied back to the MSLP grid.
  // This is deliberately an approximation: the grid correction remains
  // capped and requires three mutually consistent stations.
  function pressureAtElevation(mslpHpa,tempC,elevationM) {
    if (![mslpHpa,tempC,elevationM].every(finite) || mslpHpa<800 || mslpHpa>1100
        || tempC<-80 || tempC>60 || elevationM<0 || elevationM>2500) return NaN;
    const meanLayerK=tempC+273.15+0.00325*elevationM;
    return mslpHpa*Math.exp(-9.80665*elevationM/(287.05*meanLayerK));
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
  // Experimental objective analysis on the existing model grid. This does
  // not claim a finer effective resolution or a dynamically rerun forecast.
  function analyzeGrid(base, meta, points, policy, terrain) {
    const out = new Float32Array(base), support = new Uint16Array(base.length);
    let correctedCells = 0;
    const radius = policy.radiusKm, radius2 = radius * radius;
    const weights = new Float64Array(base.length), sums = new Float64Array(base.length);
    const squares = new Float64Array(base.length);
    // Visit station neighborhoods instead of grid × all stations.
    for (const point of points) {
      if (![point.lat,point.lon,point.increment].every(finite) || Math.abs(point.increment)>policy.maxResidual) continue;
      const halfY=radius/111.195/meta.dy;
      const halfX=radius/(111.195*Math.cos(point.lat*Math.PI/180))/meta.dx;
      const gx=(point.lon-meta.lo1)/meta.dx, gy=(meta.la1-point.lat)/meta.dy;
      for(let j=Math.max(0,Math.floor(gy-halfY));j<=Math.min(meta.ny-1,Math.ceil(gy+halfY));j++) {
        for(let i=Math.max(0,Math.floor(gx-halfX));i<=Math.min(meta.nx-1,Math.ceil(gx+halfX));i++) {
          const k=j*meta.nx+i;
          if(!finite(base[k])) continue;
          const target={lat:meta.la1-j*meta.dy,lon:meta.lo1+i*meta.dx};
          const d=distance(point,target); if(d>=radius) continue;
          if(policy.elevationToleranceM && (!terrain || !finite(terrain[k]) || !finite(point.elevationM) || Math.abs(terrain[k]-point.elevationM)>policy.elevationToleranceM)) continue;
          const w=(radius2-d*d)/(radius2+d*d);
          weights[k]+=w; sums[k]+=w*point.increment;squares[k]+=w*point.increment**2;support[k]++;
        }
      }
    }
    for(let k=0;k<out.length;k++) {
      if(support[k]<3 || weights[k]<=0) continue;
      const mean=sums[k]/weights[k];
      const spread=Math.sqrt(Math.max(0,squares[k]/weights[k]-mean*mean));
      if(spread>policy.maxSpread) continue;
      // A model prior and compact support avoid distant full-strength shifts.
      const correction=Math.max(-policy.maxCorrection,Math.min(policy.maxCorrection,sums[k]/(1+weights[k])));
      out[k]=Math.max(policy.min,Math.min(policy.max,base[k]+correction));
      correctedCells++;
    }
    return {grid:out,correctedCells,totalCells:base.length,stationCount:points.length};
  }
  const api={evaluate,distance,analyzeGrid,pressureAtElevation};
  if (typeof module!=='undefined'&&module.exports) module.exports=api;
  else root.MeteoLocalDownscaling=api;
})(typeof globalThis!=='undefined'?globalThis:this);
