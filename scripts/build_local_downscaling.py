#!/usr/bin/env python3
"""Observation-constrained temperature tool; no trained weights or LLM arithmetic.

Parameters are conservative, provisional engineering limits, not calibrated skill.
Forecast samples come from the native ICON grid, not the coarse meteogram tiles.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path

LAPSE = -0.0065  # K/m: explicit standard-atmosphere assumption, not measured
POLICY = dict(radiusKm=40, elevationToleranceM=250, maxAltitudeDeltaM=300,
              maxObservationAgeHours=2, maxHorizonHours=6, decayHours=2,
              minimumStations=3, maximumResidualC=5, maximumSpreadC=2,
              maximumCorrectionC=3, lapseRateKPerM=LAPSE)


def number(x):
    if x is None or isinstance(x, bool):
        return None
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def epoch(x):
    try:
        dt = datetime.fromisoformat(str(x).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            return None
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace('+00:00', 'Z')


def sample(forecast, field, station_index, when):
    times = [epoch(t.get('validTime')) for t in forecast.get('times', [])]
    rows = forecast.get('fields', {}).get(field, {}).get('valuesByStation', [])
    if station_index >= len(rows) or any(t is None for t in times):
        return None
    if any(b <= a for a, b in zip(times, times[1:])):
        return None
    row = rows[station_index]
    if len(row) != len(times):
        return None
    for i, t in enumerate(times):
        if when == t:
            return number(row[i])
        if i and times[i-1] < when < t and t-times[i-1] <= 5400:
            a, b = number(row[i-1]), number(row[i])
            if a is not None and b is not None:
                return a+(b-a)*(when-times[i-1])/(t-times[i-1])
    return None


def distance(a, b):
    lat1, lat2 = math.radians(a['lat']), math.radians(b['lat'])
    dlat = lat2-lat1
    dlon = math.radians(b['lon']-a['lon'])
    h = math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0088*2*math.asin(math.sqrt(min(1, max(0,h))))


def estimate(target, stations, exclude=None):
    weighted=[]
    for s in stations:
        if s['id'] == exclude:
            continue
        d=distance(target,s)
        dz=abs(target['elevationM']-s['elevationM'])
        if d >= POLICY['radiusKm'] or dz > POLICY['elevationToleranceM']:
            continue
        w=(1-(d/POLICY['radiusKm'])**2)**2 * math.exp(-(dz/150)**2)
        weighted.append((w,s['residualC']))
    if len(weighted) < POLICY['minimumStations']:
        return None
    total=sum(w for w,r in weighted)
    # Background prior avoids extending a single distant residual unchanged.
    mean=sum(w*r for w,r in weighted)/total
    spread=math.sqrt(sum(w*(r-mean)**2 for w,r in weighted)/total)
    if spread > POLICY['maximumSpreadC']:
        return None
    return max(-3,min(3,sum(w*r for w,r in weighted)/(1+total)))


def build(forecast, observations, now=None):
    now = now if now is not None else datetime.now(timezone.utc).timestamp()
    product=dict(schemaVersion=1, method='observation-constrained-temperature-v1',
                 agent='Downscaling locale · strumenti fisici', execution='deterministic-tools',
                 runTime=forecast.get('runTime'), generatedAt=iso(now), policy=POLICY,
                 status='insufficient-data', source=observations.get('source'),
                 sourceUrl=observations.get('sourceUrl'), stations=[], rejected={},
                 limitations=['Solo temperatura; nessuna modifica a vento, pioggia o temporali.',
                              'Gradiente standard -6.5 K/km assunto, non misurato; inversioni non risolte.',
                              'Limiti spaziali e temporali provvisori, non calibrati.',
                              'Verifica spaziale sullo snapshot, non validazione indipendente nel tempo.'])
    def reject(reason):
        product['rejected'][reason]=product['rejected'].get(reason,0)+1
    if epoch(forecast.get('runTime')) is None:
        reject('invalid-run'); return product
    indices={s.get('id'):(i,s) for i,s in enumerate(forecast.get('stations',[]))}
    reports={}
    for obs in observations.get('stations',[]):
        sid=obs.get('id')
        if not sid: continue
        previous=reports.get(sid)
        if previous is None or (number(obs.get('obsTime')) or 0) > (number(previous.get('obsTime')) or 0):
            reports[sid]=obs
    for sid,obs in reports.items():
        when=number(obs.get('obsTime')); temp=number(obs.get('tempC'))
        lat=number(obs.get('lat')); lon=number(obs.get('lon')); z=number(obs.get('elevationM'))
        if any(x is None for x in [when,temp,lat,lon,z]) or not -50 <= temp <= 55 or not -90 <= lat <= 90 or not -180 <= lon <= 180 or not -500 <= z <= 5000:
            reject('missing-or-invalid'); continue
        if when > now or now-when > POLICY['maxObservationAgeHours']*3600:
            reject('stale-or-future'); continue
        if sid not in indices:
            reject('no-native-sample'); continue
        i,site=indices[sid]
        if number(site.get('lat')) is None or number(site.get('lon')) is None or distance(obs,site)>1:
            reject('station-location-mismatch'); continue
        model=sample(forecast,'temperature2m',i,when)
        terrain=sample(forecast,'terrainHeight',i,when)
        if model is None or terrain is None:
            reject('missing-time-pair-or-terrain'); continue
        dz=z-terrain
        if abs(dz)>POLICY['maxAltitudeDeltaM']:
            reject('altitude-gap'); continue
        residual=temp-(model+LAPSE*dz)
        if abs(residual)>POLICY['maximumResidualC']:
            reject('large-residual'); continue
        product['stations'].append(dict(id=sid,lat=lat,lon=lon,elevationM=z,
            obsTime=when,observedC=temp,modelC=round(model,3),terrainM=round(terrain,1),
            altitudeCorrectionC=round(LAPSE*dz,3),residualC=round(residual,3)))
    stations=product['stations']
    # Same support and same baseline for both spatial scores; held-out station
    # never contributes to its own prediction.
    pairs=[]
    for s in stations:
        correction=estimate(s,stations,exclude=s['id'])
        if correction is not None:
            pairs.append((abs(s['residualC']),abs(s['residualC']-correction)))
    baseline=sum(a for a,b in pairs)/len(pairs) if pairs else None
    corrected=sum(b for a,b in pairs)/len(pairs) if pairs else None
    product['verification']=dict(method='leave-one-station-out-current-snapshot',
        count=len(pairs), baselineMaeC=baseline, correctedMaeC=corrected,
        independentTemporalValidation=False)
    product['status']='active-experimental' if len(pairs)>=5 and corrected < baseline else 'withheld'
    product['reason']=('Controllo spaziale superato; correzioni sperimentali solo con supporto locale.'
        if product['status']=='active-experimental' else
        'Correzione osservata sospesa: supporto o miglioramento spaziale insufficienti.')
    return product


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',default='data_weather')
    args=parser.parse_args(); directory=Path(args.data_dir)
    try:
        with gzip.open(directory/'verification/forecast_samples.json.gz','rt') as f:
            forecast=json.load(f)
        observations=json.loads((directory/'observations.json').read_text())
        product=build(forecast,observations)
    except (OSError,ValueError,KeyError,TypeError) as error:
        product=dict(schemaVersion=1,status='unavailable',method='observation-constrained-temperature-v1',
                     generatedAt=iso(datetime.now(timezone.utc).timestamp()),stations=[],policy=POLICY,
                     reason='Dati di ingresso non disponibili: '+type(error).__name__)
    output=directory/'downscaling.json'
    partial=output.with_suffix('.json.part')
    partial.write_text(json.dumps(product,ensure_ascii=False,allow_nan=False,separators=(',',':')))
    partial.replace(output)
    print(json.dumps({k:product.get(k) for k in ['status','reason','verification']},ensure_ascii=False))

if __name__=='__main__': main()
