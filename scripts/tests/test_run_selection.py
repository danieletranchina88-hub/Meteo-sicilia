"""Il catalogo opendata di MeteoHub puo' restare fermo mentre la directory
NWP pubblica gia' run piu' nuovi: il sito ha continuato a rieseguire con
successo un run vecchio di 43 ore, perche' get_latest_run_files() guardava
solo il catalogo. Queste prove fissano il confronto fra le due fonti.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import process_data


def _risposta(status_code=200, json_body=None, text=""):
    risposta = Mock()
    risposta.status_code = status_code
    risposta.text = text
    if json_body is not None:
        risposta.json.return_value = json_body
    if status_code >= 400:
        risposta.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        risposta.raise_for_status.return_value = None
    return risposta


CATALOGO_FERMO = [
    {"date": "2026-09-18", "run": "12:00", "filename": "aggregato-18-12.grib"},
    {"date": "2026-09-18", "run": "00:00", "filename": "aggregato-18-00.grib"},
]

DIRECTORY_HTML = """<pre>
<a href="../">../</a>
<a href="2026091812/">2026091812/</a>
<a href="2026091900/">2026091900/</a>
<a href="2026091912/">2026091912/</a>
<a href="2026092000/">2026092000/</a>
</pre>"""


def _instrada(catalogo_json, marker_ok_tags):
    """Router comune: GET sul catalogo o sulla directory, HEAD sul marker."""

    def get(url, timeout=None):
        if url == process_data.API_LIST_URL:
            return _risposta(json_body=catalogo_json)
        if url == f"{process_data.NWP_DIRECT_BASE}/{process_data.NWP_DIRECTORY_ID}/":
            return _risposta(text=DIRECTORY_HTML)
        raise AssertionError(f"GET inatteso: {url}")

    def head(url, timeout=None, allow_redirects=True):
        tag_atteso_ok = any(tag in url for tag in marker_ok_tags)
        return _risposta(status_code=200 if tag_atteso_ok else 404)

    return get, head


def test_directory_piu_recente_del_catalogo_vince():
    # Catalogo fermo al run delle 12 del 18; la directory ha gia' il 20 alle 00.
    get, head = _instrada(CATALOGO_FERMO, marker_ok_tags=["2026092000"])
    with patch("process_data.requests.get", side_effect=get), \
         patch("process_data.requests.head", side_effect=head):
        run_dt, file_list, source = process_data.get_latest_run_files()

    assert source == "nwp-direct"
    assert run_dt == datetime(2026, 9, 20, 0, tzinfo=timezone.utc)
    assert file_list == ["__nwp_direct__"]


def test_catalogo_vince_quando_e_gia_allineato():
    # Il catalogo ha gia' il run piu' recente della directory: nessun motivo
    # di pagare il costo di sette download separati invece di uno solo.
    catalogo_aggiornato = [
        {"date": "2026-09-20", "run": "00:00", "filename": "aggregato-20-00.grib"},
    ]
    get, head = _instrada(catalogo_aggiornato, marker_ok_tags=["2026092000"])
    with patch("process_data.requests.get", side_effect=get), \
         patch("process_data.requests.head", side_effect=head):
        run_dt, file_list, source = process_data.get_latest_run_files()

    assert source == "catalog"
    assert run_dt == datetime(2026, 9, 20, 0, tzinfo=timezone.utc)
    assert file_list == ["aggregato-20-00.grib"]


def test_catalogo_vince_se_piu_recente_della_directory():
    # Caso limite ma non impossibile: il catalogo puo' anche essere avanti.
    catalogo_avanti = [
        {"date": "2026-09-20", "run": "12:00", "filename": "aggregato-20-12.grib"},
    ]
    get, head = _instrada(catalogo_avanti, marker_ok_tags=["2026092000"])
    with patch("process_data.requests.get", side_effect=get), \
         patch("process_data.requests.head", side_effect=head):
        run_dt, file_list, source = process_data.get_latest_run_files()

    assert source == "catalog"
    assert run_dt == datetime(2026, 9, 20, 12, tzinfo=timezone.utc)


def test_cartella_senza_t2m_ancora_in_scrittura_viene_scartata():
    # La cartella 2026092000 compare nell'elenco ma MeteoHub non ha ancora
    # finito di scriverci dentro: niente marker T_2M, si scende al run prima.
    get, head = _instrada(CATALOGO_FERMO, marker_ok_tags=["2026091912"])
    with patch("process_data.requests.get", side_effect=get), \
         patch("process_data.requests.head", side_effect=head):
        run_dt, file_list, source = process_data.get_latest_run_files()

    assert source == "nwp-direct"
    assert run_dt == datetime(2026, 9, 19, 12, tzinfo=timezone.utc)


def test_catalogo_irraggiungibile_usa_comunque_la_directory():
    def get(url, timeout=None):
        if url == process_data.API_LIST_URL:
            raise ConnectionError("MeteoHub non risponde")
        if url == f"{process_data.NWP_DIRECT_BASE}/{process_data.NWP_DIRECTORY_ID}/":
            return _risposta(text=DIRECTORY_HTML)
        raise AssertionError(f"GET inatteso: {url}")

    def head(url, timeout=None, allow_redirects=True):
        return _risposta(status_code=200 if "2026092000" in url else 404)

    with patch("process_data.requests.get", side_effect=get), \
         patch("process_data.requests.head", side_effect=head):
        run_dt, file_list, source = process_data.get_latest_run_files()

    assert source == "nwp-direct"
    assert run_dt == datetime(2026, 9, 20, 0, tzinfo=timezone.utc)


def test_nessuna_fonte_disponibile_non_pubblica_nulla():
    def get(url, timeout=None):
        raise ConnectionError("MeteoHub irraggiungibile")

    with patch("process_data.requests.get", side_effect=get):
        run_dt, file_list, source = process_data.get_latest_run_files()

    assert run_dt is None
    assert file_list == []


if __name__ == "__main__":
    test_directory_piu_recente_del_catalogo_vince()
    test_catalogo_vince_quando_e_gia_allineato()
    test_catalogo_vince_se_piu_recente_della_directory()
    test_cartella_senza_t2m_ancora_in_scrittura_viene_scartata()
    test_catalogo_irraggiungibile_usa_comunque_la_directory()
    test_nessuna_fonte_disponibile_non_pubblica_nulla()
    print("Run-selection tests passed")
