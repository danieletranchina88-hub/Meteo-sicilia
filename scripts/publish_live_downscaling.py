#!/usr/bin/env python3
"""Publish a run-bound live product without altering the immutable run snapshot."""
import argparse
import gzip
import json
from pathlib import Path
from build_local_downscaling import build

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--site-dir',required=True)
    p.add_argument('--observations-dir',required=True)
    args=p.parse_args()
    data=Path(args.site_dir)/'data_weather'
    snapshots=sorted(Path(args.observations_dir).glob('metar_*.json'))
    if not snapshots: raise RuntimeError('Nessuna osservazione raccolta')
    observations=json.loads(snapshots[-1].read_text())
    with gzip.open(data/'verification/forecast_samples.json.gz','rt') as f:
        forecast=json.load(f)
    product=build(forecast,observations)
    output=data/'live';output.mkdir(parents=True,exist_ok=True)
    for name,value in [('observations.json',observations),('downscaling.json',product)]:
        dest=output/name;partial=dest.with_suffix('.json.part')
        partial.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')))
        partial.replace(dest)
    print(product['status'],len(product['stations']))

if __name__=='__main__': main()
