#!/usr/bin/env python3
"""Conserva nel nuovo ambiente nubi le ore passate dei run precedenti.

Uso: merge_cloud_environment.py --new data_weather/cloud_env --old <gh-pages>/data_weather/cloud_env
Non fallisce mai il deploy: senza ambiente vecchio non c'e' niente da fare.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meteo_analysis.clouds.environment import merge_previous  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new", required=True)
    parser.add_argument("--old", required=True)
    args = parser.parse_args(argv)
    if not os.path.exists(os.path.join(args.new, "index.json")):
        print("Ambiente nubi del nuovo run assente: niente da unire.")
        return 0
    index = merge_previous(args.new, args.old)
    print(f"Ambiente nubi: {len(index['hours'])} ore disponibili "
          f"({index['hours'][0]['valid']} - {index['hours'][-1]['valid']}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
