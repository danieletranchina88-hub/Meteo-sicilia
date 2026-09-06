#!/usr/bin/env python3
"""Prove della probabilita' di vicinato.

Sono quasi tutte verificabili a mano: una frazione di punti in un cerchio si
conta, e il risultato non dipende da tarature. Dove serve un riferimento, il
conto e' rifatto per forza bruta sulla definizione, cosi' la prova controlla
l'implementazione veloce contro la formula e non contro se stessa.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meteo_analysis.core.neighbourhood import (  # noqa: E402
    cell_sizes_km,
    event_probabilities,
    event_probability,
    neighbourhood_probability,
)


def brute_force(values, threshold, radius_km, cell_x_km, cell_y_km):
    """La definizione, applicata punto per punto senza scorciatoie.

    Le distanze usano il passo in longitudine del punto CENTRALE, che e' la
    convenzione dichiarata dal modulo: il vicinato e' un insieme attorno a x.
    """
    height, width = values.shape
    cells_x = np.broadcast_to(np.asarray(cell_x_km, dtype=float), (height,))
    out = np.full((height, width), np.nan)
    for j in range(height):
        for i in range(width):
            hits = total = 0
            for jj in range(height):
                dy = (jj - j) * cell_y_km
                if abs(dy) > radius_km:
                    continue
                for ii in range(width):
                    dx = (ii - i) * cells_x[j]
                    if dx * dx + dy * dy > radius_km * radius_km:
                        continue
                    if not np.isfinite(values[jj, ii]):
                        continue
                    total += 1
                    if values[jj, ii] >= threshold:
                        hits += 1
            if total:
                out[j, i] = hits / total
    return out


class TestVicinato(unittest.TestCase):
    def test_campo_tutto_sopra_soglia(self):
        campo = np.full((21, 21), 30.0)
        p = neighbourhood_probability(campo, 20.0, 10.0, cell_x_km=2.0, cell_y_km=2.0)
        np.testing.assert_allclose(p, 1.0)

    def test_campo_tutto_sotto_soglia(self):
        campo = np.full((21, 21), 5.0)
        p = neighbourhood_probability(campo, 20.0, 10.0, cell_x_km=2.0, cell_y_km=2.0)
        np.testing.assert_allclose(p, 0.0)

    def test_soglia_inclusiva(self):
        # "almeno 20 mm" deve contare anche il punto che fa esattamente 20.
        campo = np.full((11, 11), 20.0)
        p = neighbourhood_probability(campo, 20.0, 4.0, cell_x_km=2.0, cell_y_km=2.0)
        np.testing.assert_allclose(p, 1.0)

    def test_probabilita_sempre_fra_zero_e_uno(self):
        rng = np.random.default_rng(20260906)
        campo = rng.normal(10.0, 6.0, size=(40, 45))
        p = neighbourhood_probability(campo, 12.0, 9.0, cell_x_km=2.3, cell_y_km=2.2)
        self.assertTrue(np.all(p >= 0.0) and np.all(p <= 1.0))

    def test_monotona_nella_soglia(self):
        # Alzando l'asticella la probabilita' non puo' crescere.
        rng = np.random.default_rng(7)
        campo = rng.gamma(2.0, 4.0, size=(35, 35))
        bassa = neighbourhood_probability(campo, 4.0, 8.0, cell_x_km=2.2, cell_y_km=2.2)
        alta = neighbourhood_probability(campo, 12.0, 8.0, cell_x_km=2.2, cell_y_km=2.2)
        self.assertTrue(np.all(alta <= bassa + 1e-12))

    def test_coincide_con_la_definizione(self):
        rng = np.random.default_rng(99)
        campo = rng.gamma(1.5, 5.0, size=(23, 27))
        for raggio, cx, cy in ((6.0, 2.2, 2.2), (9.0, 1.9, 2.4), (5.0, 2.6, 1.8)):
            atteso = brute_force(campo, 8.0, raggio, cx, cy)
            ottenuto = neighbourhood_probability(
                campo, 8.0, raggio, cell_x_km=cx, cell_y_km=cy
            )
            np.testing.assert_allclose(ottenuto, atteso, atol=1e-12)

    def test_coincide_con_la_definizione_su_righe_variabili(self):
        # Il caso che conta davvero: passo in longitudine diverso per riga.
        rng = np.random.default_rng(4242)
        campo = rng.gamma(1.5, 5.0, size=(19, 23))
        cx = np.linspace(2.32, 1.83, 19)
        atteso = brute_force(campo, 6.0, 8.0, cx, 2.226)
        ottenuto = neighbourhood_probability(
            campo, 6.0, 8.0, cell_x_km=cx, cell_y_km=2.226
        )
        np.testing.assert_allclose(ottenuto, atteso, atol=1e-12)

    def test_punto_isolato_illumina_un_cerchio_in_chilometri(self):
        # Con celle piu' strette in x che in y il vicinato deve restare un
        # cerchio sul terreno, quindi in CELLE deve essere piu' largo che alto.
        campo = np.zeros((41, 41))
        campo[20, 20] = 100.0
        p = neighbourhood_probability(
            campo, 50.0, 20.0, cell_x_km=1.0, cell_y_km=2.0
        )
        acceso = p > 0.0
        larghezza = np.count_nonzero(acceso[20, :])
        altezza = np.count_nonzero(acceso[:, 20])
        self.assertEqual(larghezza, 41)  # 20 km / 1.0 km = 20 celle per lato
        self.assertEqual(altezza, 21)  # 20 km / 2.0 km = 10 celle per lato
        # E il rapporto e' esattamente quello dei passi, non un valore vicino.
        self.assertEqual((larghezza - 1) // (altezza - 1), 2)

    def test_area_del_disco_segue_pi_erre_quadro(self):
        # Il denominatore e' il numero di punti del cerchio: deve tendere a
        # pi*R^2 diviso l'area di una cella.
        campo = np.zeros((81, 81))
        raggio, cella = 20.0, 1.0
        p = neighbourhood_probability(
            np.where(campo == 0, 1.0, 1.0), 1.0, raggio,
            cell_x_km=cella, cell_y_km=cella,
        )
        np.testing.assert_allclose(p, 1.0)  # tutti dentro: frazione unitaria
        singolo = np.zeros((81, 81))
        singolo[40, 40] = 1.0
        q = neighbourhood_probability(
            singolo, 1.0, raggio, cell_x_km=cella, cell_y_km=cella
        )
        # Al centro la frazione e' 1/(punti del disco).
        punti = 1.0 / q[40, 40]
        atteso = np.pi * raggio * raggio / (cella * cella)
        self.assertLess(abs(punti - atteso) / atteso, 0.03)

    def test_i_mancanti_non_contano_da_nessuna_parte(self):
        campo = np.full((21, 21), 30.0)
        campo[:, :10] = np.nan
        p = neighbourhood_probability(campo, 20.0, 6.0, cell_x_km=2.0, cell_y_km=2.0)
        # Dove restano punti validi la frazione e' piena: i NaN non diluiscono.
        np.testing.assert_allclose(p[:, 12:], 1.0)

    def test_tutto_mancante_resta_mancante(self):
        campo = np.full((11, 11), np.nan)
        p = neighbourhood_probability(campo, 1.0, 5.0, cell_x_km=2.0, cell_y_km=2.0)
        self.assertTrue(np.all(np.isnan(p)))

    def test_raggio_sotto_la_cella_resta_puntuale(self):
        campo = np.array([[0.0, 30.0], [0.0, 0.0]])
        p = neighbourhood_probability(campo, 20.0, 0.4, cell_x_km=2.0, cell_y_km=2.0)
        np.testing.assert_allclose(p, np.array([[0.0, 1.0], [0.0, 0.0]]))

    def test_parametri_impossibili(self):
        campo = np.zeros((5, 5))
        with self.assertRaises(ValueError):
            neighbourhood_probability(campo, 1.0, -1.0, cell_x_km=2.0, cell_y_km=2.0)
        with self.assertRaises(ValueError):
            neighbourhood_probability(campo, 1.0, 5.0, cell_x_km=0.0, cell_y_km=2.0)
        with self.assertRaises(ValueError):
            neighbourhood_probability(
                np.zeros(5), 1.0, 5.0, cell_x_km=2.0, cell_y_km=2.0
            )
        with self.assertRaises(ValueError):
            neighbourhood_probability(
                campo, 1.0, 5.0, cell_x_km=np.ones(4), cell_y_km=2.0
            )


class TestProbabilitaEvento(unittest.TestCase):
    """Il doppio stadio: l'evento e' areale, poi la posizione e' incerta."""

    def test_una_cella_isolata_diventa_probabilita_alta_vicino(self):
        # Il punto della trappola: una cella sola bagna una frazione d'area
        # minuscola, ma la probabilita' che piova VICINO a te e' alta.
        campo = np.zeros((81, 81))
        campo[40, 40] = 30.0
        frazione = neighbourhood_probability(
            campo, 20.0, 25.0, cell_x_km=2.2, cell_y_km=2.2
        )
        evento = event_probability(
            campo, 20.0, cell_x_km=2.2, cell_y_km=2.2,
            event_radius_km=10.0, spread_radius_km=25.0,
        )
        self.assertLess(frazione[40, 40], 0.01)
        self.assertGreater(evento[40, 40], 0.10)
        self.assertGreater(evento[40, 40], frazione[40, 40] * 10)

    def test_nel_cuore_di_un_campo_esteso_vale_uno(self):
        # Se piove ovunque, la probabilita' che piova vicino e' certezza.
        campo = np.full((81, 81), 30.0)
        p = event_probability(campo, 20.0, cell_x_km=2.2, cell_y_km=2.2)
        np.testing.assert_allclose(p[30:50, 30:50], 1.0)

    def test_campo_asciutto_resta_a_zero(self):
        campo = np.zeros((41, 41))
        p = event_probability(campo, 20.0, cell_x_km=2.2, cell_y_km=2.2)
        np.testing.assert_allclose(p, 0.0)

    def test_non_supera_mai_uno(self):
        rng = np.random.default_rng(11)
        campo = rng.gamma(1.2, 6.0, size=(60, 60))
        p = event_probability(campo, 10.0, cell_x_km=2.1, cell_y_km=2.2)
        self.assertTrue(np.all(p >= 0.0) and np.all(p <= 1.0))

    def test_monotona_nella_soglia(self):
        rng = np.random.default_rng(3)
        campo = rng.gamma(1.2, 6.0, size=(50, 50))
        bassa = event_probability(campo, 5.0, cell_x_km=2.2, cell_y_km=2.2)
        alta = event_probability(campo, 15.0, cell_x_km=2.2, cell_y_km=2.2)
        self.assertTrue(np.all(alta <= bassa + 1e-12))

    def test_evento_piu_largo_non_puo_ridurre_la_probabilita(self):
        rng = np.random.default_rng(5)
        campo = rng.gamma(1.2, 6.0, size=(50, 50))
        stretto = event_probability(
            campo, 12.0, cell_x_km=2.2, cell_y_km=2.2, event_radius_km=5.0
        )
        largo = event_probability(
            campo, 12.0, cell_x_km=2.2, cell_y_km=2.2, event_radius_km=15.0
        )
        self.assertTrue(np.all(largo >= stretto - 1e-12))

    def test_multi_soglia_uguale_alle_singole(self):
        rng = np.random.default_rng(2026)
        campo = rng.gamma(1.4, 5.0, size=(45, 55))
        cx = np.linspace(2.31, 1.84, 45)
        soglie = [1.0, 5.0, 10.0, 20.0]
        insieme = event_probabilities(
            campo, soglie, cell_x_km=cx, cell_y_km=2.226
        )
        for soglia, ottenuto in zip(soglie, insieme):
            atteso = event_probability(
                campo, soglia, cell_x_km=cx, cell_y_km=2.226
            )
            np.testing.assert_allclose(ottenuto, atteso, atol=1e-12)

    def test_i_mancanti_non_diventano_evento(self):
        campo = np.full((41, 41), np.nan)
        campo[20:, :] = 0.0
        p = event_probability(campo, 1.0, cell_x_km=2.2, cell_y_km=2.2)
        # Dove non si sa nulla la probabilita' resta ignota, non zero.
        self.assertTrue(np.all(np.isnan(p[:5, :])))
        np.testing.assert_allclose(p[35:, :], 0.0)

    def test_raggi_impossibili(self):
        campo = np.zeros((5, 5))
        with self.assertRaises(ValueError):
            event_probability(
                campo, 1.0, cell_x_km=2.0, cell_y_km=2.0, event_radius_km=0.0
            )
        with self.assertRaises(ValueError):
            event_probability(
                campo, 1.0, cell_x_km=2.0, cell_y_km=2.0, spread_radius_km=-3.0
            )


class TestPassoDellaGriglia(unittest.TestCase):
    def test_passo_reale_del_dominio_icon(self):
        meta = {"la1": 48.9, "dy": 0.02, "dx": 0.025, "ny": 761}
        cx, cy = cell_sizes_km(meta)
        self.assertAlmostEqual(cy, 2.2264, places=3)
        self.assertEqual(cx.shape, (761,))
        # Il bordo nord ha celle piu' strette del bordo sud: e' il motivo per
        # cui il raggio in celle non puo' essere costante.
        self.assertAlmostEqual(cx[0], 1.829, places=2)
        self.assertAlmostEqual(cx[-1], 2.315, places=2)
        self.assertLess(cx[0] / cx[-1], 0.80 + 0.03)

    def test_il_disco_isotropo_sbaglierebbe_del_diciotto_per_cento(self):
        # La misura che giustifica tutto il modulo: con un passo unico medio
        # il vicinato al bordo nord sarebbe un'ellisse, non un cerchio.
        meta = {"la1": 48.9, "dy": 0.02, "dx": 0.025, "ny": 761}
        cx, cy = cell_sizes_km(meta)
        medio = (cx[0] + cy) / 2.0
        celle = 25.0 / medio
        # Il raggio nominale e' 25 km: con un passo unico diventa un'ellisse.
        self.assertAlmostEqual(celle * cx[0], 22.6, delta=0.1)
        self.assertAlmostEqual(celle * cy, 27.5, delta=0.1)
        schiacciamento = 1.0 - (celle * cx[0]) / (celle * cy)
        self.assertAlmostEqual(schiacciamento, 0.178, delta=0.005)


class TestPubblicazione(unittest.TestCase):
    """Il blocco che finisce nel file del passo."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import process_data

        self.pipeline = process_data
        self.header = {
            "nx": 60, "ny": 50, "lo1": 3.0, "la1": 48.9,
            "dx": 0.025, "dy": 0.02,
        }

    def test_pubblica_le_soglie_dichiarate(self):
        rng = np.random.default_rng(1)
        pioggia = rng.gamma(0.4, 6.0, size=50 * 60)
        raffica = rng.gamma(2.0, 6.0, size=50 * 60)  # m/s
        blocco = self.pipeline.build_exceedance_probabilities(
            self.header, pioggia, raffica
        )
        self.assertEqual(
            sorted(blocco["fields"]),
            sorted(["rain_1", "rain_5", "rain_10", "rain_20",
                    "gust_50", "gust_75"]),
        )
        # La griglia e' diradata: passo doppio, meta' dei punti per lato.
        self.assertEqual(blocco["nx"], 30)
        self.assertEqual(blocco["ny"], 25)
        self.assertAlmostEqual(blocco["dx"], 0.05)
        self.assertAlmostEqual(blocco["dy"], 0.04)
        for nome, valori in blocco["fields"].items():
            self.assertEqual(len(valori), 30 * 25, nome)
            noti = [v for v in valori if v is not None]
            self.assertTrue(all(0 <= v <= 100 for v in noti), nome)

    def test_semantica_dichiarata_nel_file(self):
        # Chi legge il JSON deve trovare scritto che non e' calibrata, senza
        # dover risalire al codice che l'ha prodotta.
        blocco = self.pipeline.build_exceedance_probabilities(
            self.header, np.ones(50 * 60), np.ones(50 * 60) * 20.0
        )
        self.assertIn("not-calibrated", blocco["semantics"])
        self.assertEqual(blocco["eventRadiusKm"], self.pipeline.PROB_EVENT_RADIUS_KM)
        self.assertEqual(blocco["spreadRadiusKm"], self.pipeline.PROB_SPREAD_RADIUS_KM)

    def test_senza_campi_non_pubblica_nulla(self):
        # Meglio assente che pieno di zeri: uno zero si legge come "non
        # succedera'", che e' un'affermazione, mentre qui non si sa.
        self.assertIsNone(
            self.pipeline.build_exceedance_probabilities(self.header, None, None)
        )
        vuoto = np.full(50 * 60, np.nan)
        self.assertIsNone(
            self.pipeline.build_exceedance_probabilities(self.header, vuoto, vuoto)
        )

    def test_la_raffica_viene_convertita_in_kmh(self):
        # 20 m/s sono 72 km/h: sopra 50, sotto 75. Se la conversione sparisse,
        # la soglia da 50 km/h verrebbe letta su un campo in m/s e non
        # scatterebbe quasi mai.
        raffica = np.full(50 * 60, 20.0)
        blocco = self.pipeline.build_exceedance_probabilities(
            self.header, None, raffica
        )
        self.assertEqual(max(blocco["fields"]["gust_50"]), 100)
        self.assertEqual(max(blocco["fields"]["gust_75"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
