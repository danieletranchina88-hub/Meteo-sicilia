"""Contract and isolation tests using a reduced real MeteoHub export record."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch, Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from meteo_analysis.verification.meteohub import normalize_jsonl, fetch_site_observations, fetch_meteohub_stations

NOW = datetime(2026, 9, 5, 1, tzinfo=timezone.utc)
ROW = {"version":"0.1","network":"dpcn-sicilia","ident":None,
       "lon":1450112,"lat":3762172,"date":"2026-09-05T00:00:00Z",
       "data":[{"vars":{"B01019":{"v":"Agira"},"B05001":{"v":37.62172},
                         "B06001":{"v":14.50112},"B07030":{"v":0.0}}},
               {"timerange":[254,0,0],"level":[103,2000,None,None],
                "vars":{"B12101":{"v":296.09}}}]}


def test_real_contract():
    session = Mock()
    session.post.return_value.iter_lines.return_value = [json.dumps(ROW).encode()]
    assert len(fetch_meteohub_stations(session, NOW)) == 1
    assert session.post.call_args.kwargs["params"]["output_format"] == "JSON"
    session.post.return_value.close.assert_called_once()
    result = normalize_jsonl([json.dumps(ROW)], NOW)
    assert len(result) == 1 and result[0]["tempC"] == 22.94
    assert result[0]["elevationM"] is None
    assert result[0]["lat"] == 37.62172
    for bad_value in [None, True, 9999, "NaN"]:
        bad = copy.deepcopy(ROW)
        bad["data"][1]["vars"]["B12101"]["v"] = bad_value
        assert not normalize_jsonl([json.dumps(bad)], NOW)
    for date in ["2026-09-05T02:00:00Z", "2026-09-04T00:00:00Z", "2026-09-05T00:00:00"]:
        bad = {**ROW, "date": date}
        assert not normalize_jsonl([json.dumps(bad)], NOW)
    newer = {**ROW, "date":"2026-09-05T00:30:00Z"}
    assert len(normalize_jsonl([json.dumps(newer), json.dumps(ROW)], NOW)) == 1
    bad = copy.deepcopy(ROW); bad["data"][1]["timerange"]=[0,0,3600]
    assert not normalize_jsonl([json.dumps(bad)], NOW)


def test_sources_fail_independently():
    station = normalize_jsonl([json.dumps(ROW)], NOW)[0]
    with patch("meteo_analysis.verification.observations.fetch_italy_metar_observations", side_effect=RuntimeError), patch("meteo_analysis.verification.meteohub.fetch_meteohub_stations", return_value=[station]):
        result = fetch_site_observations()
        assert result["count"] == 1
        assert result["sourceStatus"][0]["status"] == "unavailable"
        assert result["sourceStatus"][1]["status"] == "ok"
    metar={"count":1,"stations":[{"id":"LICJ","obsTime":1}],"stationNetwork":{"stations":[{"id":"LICJ"}]}}
    with patch("meteo_analysis.verification.observations.fetch_italy_metar_observations", return_value=metar), patch("meteo_analysis.verification.meteohub.fetch_meteohub_stations", side_effect=RuntimeError):
        result = fetch_site_observations()
        assert result["stations"][0]["id"] == "LICJ"
        assert result["sourceStatus"][1]["status"] == "unavailable"


if __name__ == "__main__":
    test_real_contract()
    test_sources_fail_independently()
    print("MeteoHub tests passed")
