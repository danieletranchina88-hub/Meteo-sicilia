#!/usr/bin/env python3
"""Prove del profilo termico verticale.

Quasi tutte hanno un riferimento indipendente: l'atmosfera standard e' tabulata
da decenni, e l'equazione ipsometrica si verifica a mano. Dove il riferimento
non c'e', la prova controlla una proprieta' fisica che deve valere comunque
(l'aria calda e' piu' spessa, un profilo isotermo non corregge nulla,
un'inversione corregge in su).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meteo_analysis.core.vertical_profile import (  # noqa: E402
    HYPSOMETRIC_M_PER_K,
    elevation_correction_c,
    layer_thickness_m,
    saturation_vapour_pressure_pa,
    snow_line_m,
    temperature_profile,
    vapour_pressure_pa,
    wet_bulb_c,
)


def atmosfera_standard(quota_m):
    """Temperatura dell'atmosfera standard ICAO, in gradi."""
    return 15.0 - 0.0065 * np.asarray(quota_m, dtype=float)


class TestEquazioneIpsometrica(unittest.TestCase):
    def test_costante(self):
        # R_d/g = 287,05 / 9,80665. Se cambia, cambia ogni quota calcolata.
        self.assertAlmostEqual(HYPSOMETRIC_M_PER_K, 29.271, places=3)

    def test_aria_calda_piu_spessa(self):
        freddo = layer_thickness_m(925.0, 850.0, -10.0)
        caldo = layer_thickness_m(925.0, 850.0, 20.0)
        self.assertGreater(caldo, freddo)
        # Lo spessore e' proporzionale alla temperatura assoluta: 30 gradi su
        # circa 273 sono circa l'11% in piu'.
        self.assertAlmostEqual(caldo / freddo, 293.15 / 263.15, places=6)

    def test_spessore_925_850_atmosfera_standard(self):
        # Nell'atmosfera standard 925 hPa sta a 762 m e 850 a 1457: lo spessore
        # e' circa 695 m. Il conto qui deve ritrovarlo.
        media = atmosfera_standard((762.0 + 1457.0) / 2.0)
        spessore = layer_thickness_m(925.0, 850.0, media)
        self.assertAlmostEqual(spessore, 1457.0 - 762.0, delta=8.0)

    def test_livelli_invertiti_rifiutati(self):
        with self.assertRaises(ValueError):
            layer_thickness_m(850.0, 925.0, 0.0)
        with self.assertRaises(ValueError):
            layer_thickness_m(925.0, 0.0, 0.0)


class TestProfilo(unittest.TestCase):
    def profilo_standard(self, terreno=0.0):
        """Colonna in atmosfera standard: quote note, per confronto."""
        quote_vere = np.array([762.0, 1457.0, 3012.0])
        t = [np.array([atmosfera_standard(z)]) for z in quote_vere]
        return temperature_profile(
            np.array([1013.25]),
            np.array([atmosfera_standard(terreno)]),
            np.array([terreno]),
            t[0], t[1], t[2],
        ), quote_vere

    def test_quote_dei_livelli_in_atmosfera_standard(self):
        (quote, _), attese = self.profilo_standard()
        for calcolata, attesa, nome in zip(quote, attese, ("925", "850", "700")):
            self.assertAlmostEqual(
                float(calcolata[0]), attesa, delta=25.0,
                msg=f"quota di {nome} hPa lontana dall'atmosfera standard",
            )

    def test_quote_crescenti(self):
        (quote, _), _ = self.profilo_standard()
        self.assertLess(float(quote[0][0]), float(quote[1][0]))
        self.assertLess(float(quote[1][0]), float(quote[2][0]))

    def test_accetta_kelvin_e_gradi(self):
        (in_gradi, _), _ = self.profilo_standard()
        t = [np.array([atmosfera_standard(z) + 273.15])
             for z in (762.0, 1457.0, 3012.0)]
        in_kelvin, _ = temperature_profile(
            np.array([1013.25]), np.array([288.15]), np.array([0.0]),
            t[0], t[1], t[2],
        )
        for a, b in zip(in_gradi, in_kelvin):
            np.testing.assert_allclose(a, b, atol=1e-9)


class TestCorrezione(unittest.TestCase):
    def profilo(self, temperature, quote=(762.0, 1457.0, 3012.0)):
        t = [np.array([v], dtype=float) for v in temperature]
        return [np.array([z]) for z in quote], t

    def test_profilo_isotermo_non_corregge(self):
        # Se la temperatura non cambia con la quota, spostare il punto non
        # cambia niente. Un gradiente assunto direbbe di si', e sbaglierebbe.
        quote, t = self.profilo((5.0, 5.0, 5.0))
        correzione = elevation_correction_c(quote, t, 2078.0, 2912.0)
        np.testing.assert_allclose(correzione, 0.0, atol=1e-9)

    def test_gradiente_ordinario_raffredda_le_cime(self):
        quote, t = self.profilo((10.0, 5.5, -4.6))  # circa -6,5 K/km
        correzione = elevation_correction_c(quote, t, 2078.0, 2912.0)
        # 834 m di dislivello a -6,5 K/km sono circa -5,4 gradi.
        self.assertAlmostEqual(float(correzione[0]), -5.4, delta=0.4)

    def test_inversione_riscalda_le_cime(self):
        # Il caso che un lapse fisso sbaglia di SEGNO: sotto un'inversione la
        # cima e' piu' calda del fondovalle.
        quote, t = self.profilo((-4.0, 3.0, -6.0))
        correzione = elevation_correction_c(quote, t, 300.0, 1200.0)
        self.assertGreater(float(correzione[0]), 0.0)

    def test_nessuno_spostamento_nessuna_correzione(self):
        quote, t = self.profilo((10.0, 5.5, -4.6))
        correzione = elevation_correction_c(quote, t, 1500.0, 1500.0)
        np.testing.assert_allclose(correzione, 0.0, atol=1e-12)

    def test_antisimmetrica(self):
        # Scendere di 800 m deve dare l'opposto di salire di 800.
        quote, t = self.profilo((10.0, 5.5, -4.6))
        su = elevation_correction_c(quote, t, 1500.0, 2300.0)
        giu = elevation_correction_c(quote, t, 2300.0, 1500.0)
        np.testing.assert_allclose(su, -giu, atol=1e-12)

    def test_usa_il_livello_giusto(self):
        # Sopra 850 hPa deve valere il gradiente del segmento alto, non quello
        # del basso: qui i due sono deliberatamente opposti.
        quote, t = self.profilo((0.0, 10.0, 0.0))  # inversione sotto, calo sopra
        bassa = elevation_correction_c(quote, t, 800.0, 1400.0)
        alta = elevation_correction_c(quote, t, 1600.0, 2600.0)
        self.assertGreater(float(bassa[0]), 0.0)
        self.assertLess(float(alta[0]), 0.0)

    def test_funziona_su_griglie(self):
        quote = [np.full((4, 5), z) for z in (762.0, 1457.0, 3012.0)]
        t = [np.full((4, 5), v) for v in (10.0, 5.5, -4.6)]
        modello = np.full((4, 5), 2078.0)
        vero = np.full((4, 5), 2912.0)
        correzione = elevation_correction_c(quote, t, modello, vero)
        self.assertEqual(correzione.shape, (4, 5))
        self.assertTrue(np.all(correzione < 0.0))


class TestCasoReale(unittest.TestCase):
    def test_le_cime_misurate_sul_run(self):
        """L'errore che il modulo esiste per correggere, con i numeri veri.

        Orografia ICON-2I contro quota reale, misurata sul run del 6 settembre
        2026. In atmosfera ordinaria la correzione deve avere questi ordini di
        grandezza; se un giorno cambiassero di segno o di scala, qualcosa nella
        catena si e' rotto.
        """
        cime = [
            ("Gran Sasso", 2078.0, 2912.0, -5.4),
            ("Monte Bianco", 4000.0, 4808.0, -5.3),
            ("Marmolada", 2616.0, 3343.0, -4.7),
            ("Etna", 2866.0, 3357.0, -3.2),
            ("Vesuvio", 886.0, 1281.0, -2.6),
        ]
        quote = [np.array([762.0]), np.array([1457.0]), np.array([3012.0])]
        t = [np.array([10.0]), np.array([5.5]), np.array([-4.6])]
        for nome, modello, reale, atteso in cime:
            correzione = float(
                elevation_correction_c(quote, t, modello, reale)[0]
            )
            self.assertAlmostEqual(correzione, atteso, delta=0.6, msg=nome)


class TestPubblicazione(unittest.TestCase):
    """Il blocco che finisce nel file del passo."""

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import process_data

        self.pipeline = process_data
        self.header = {
            "nx": 48, "ny": 40, "lo1": 3.0, "la1": 48.9,
            "dx": 0.025, "dy": 0.02,
        }
        n = 48 * 40
        self.terreno = np.full(n, 500.0)
        self.t2m = np.full(n, 12.0)
        self.press = np.full(n, 1013.0)

    def profilo(self):
        return self.pipeline.build_vertical_profile(
            self.header, self.press, self.t2m, self.terreno,
            np.full(48 * 40, 10.0), np.full(48 * 40, 5.5),
            np.full(48 * 40, -4.6),
        )

    def test_pubblica_tre_livelli(self):
        blocco = self.profilo()
        self.assertEqual(len(blocco["z"]), 3)
        self.assertEqual(len(blocco["t"]), 3)
        self.assertEqual(blocco["levelsHpa"], [925.0, 850.0, 700.0])
        # Griglia diradata di 8: il profilo e' sinottico, non serve fitto.
        self.assertEqual(blocco["nx"], 6)
        self.assertEqual(blocco["ny"], 5)
        for livello in blocco["z"] + blocco["t"]:
            self.assertEqual(len(livello), 30)

    def test_quote_crescenti_anche_dopo_il_diradamento(self):
        blocco = self.profilo()
        z = [np.array([v for v in livello if v is not None]) for livello in blocco["z"]]
        self.assertTrue(np.all(z[0] < z[1]))
        self.assertTrue(np.all(z[1] < z[2]))

    def test_dichiara_di_non_usare_un_gradiente_di_manuale(self):
        # Chi legge il JSON deve capire da dove viene il profilo senza
        # risalire al codice.
        self.assertIn("not-standard-lapse-rate", self.profilo()["semantics"])

    def test_senza_un_livello_non_pubblica_nulla(self):
        # Un profilo con un buco produrrebbe correzioni peggiori dell'assenza
        # di correzione: meglio niente.
        for mancante in range(3):
            livelli = [np.full(48 * 40, v) for v in (10.0, 5.5, -4.6)]
            livelli[mancante] = None
            self.assertIsNone(
                self.pipeline.build_vertical_profile(
                    self.header, self.press, self.t2m, self.terreno, *livelli
                ),
                f"livello {mancante} assente ma il blocco viene pubblicato",
            )
        self.assertIsNone(
            self.pipeline.build_vertical_profile(
                self.header, self.press, self.t2m, None,
                *[np.full(48 * 40, v) for v in (10.0, 5.5, -4.6)]
            )
        )


class TestBulboBagnato(unittest.TestCase):
    def test_saturazione_a_zero_gradi(self):
        # Valore tabulato: 611,2 Pa a 0 C. E' la costante di Bolton.
        self.assertAlmostEqual(float(saturation_vapour_pressure_pa(0.0)), 611.2, places=1)

    def test_saturazione_a_venti_gradi(self):
        # Tabelle psicrometriche: circa 2339 Pa a 20 C.
        self.assertAlmostEqual(
            float(saturation_vapour_pressure_pa(20.0)), 2339.0, delta=15.0
        )

    def test_aria_satura_ha_bulbo_uguale_alla_temperatura(self):
        # Con umidita' relativa al 100% non c'e' evaporazione, quindi non c'e'
        # raffreddamento: il bulbo bagnato coincide con quello secco.
        t = 12.0
        e = saturation_vapour_pressure_pa(t)
        p = 1000.0
        q = 0.622 * e / (p * 100.0 - 0.378 * e)
        self.assertAlmostEqual(float(wet_bulb_c(t, q, p)), t, delta=0.05)

    def test_aria_secca_abbassa_il_bulbo(self):
        secca = float(wet_bulb_c(20.0, 0.002, 1000.0))
        umida = float(wet_bulb_c(20.0, 0.012, 1000.0))
        self.assertLess(secca, umida)
        self.assertLess(secca, 20.0)

    def test_bulbo_fra_rugiada_e_temperatura(self):
        # Proprieta' che deve valere sempre, ed e' il motivo per cui la
        # bisezione ha un intervallo sicuro.
        rng = np.random.default_rng(3)
        t = rng.uniform(-15.0, 35.0, 400)
        q = rng.uniform(0.0005, 0.018, 400)
        p = rng.uniform(700.0, 1020.0, 400)
        tw = wet_bulb_c(t, q, p)
        e = vapour_pressure_pa(q, p)
        ratio = np.log(e / 611.2)
        rugiada = 243.5 * ratio / (17.67 - ratio)
        self.assertTrue(np.all(tw <= t + 1e-6))
        self.assertTrue(np.all(tw >= np.minimum(rugiada, t) - 1e-6))

    def test_caso_psicrometrico_noto(self):
        # 25 C con rugiada 10 C a 1000 hPa: le tavole danno un bulbo bagnato
        # intorno a 15,5 C.
        e = saturation_vapour_pressure_pa(10.0)
        q = 0.622 * e / (1000.0 * 100.0 - 0.378 * e)
        self.assertAlmostEqual(float(wet_bulb_c(25.0, q, 1000.0)), 15.5, delta=0.6)


class TestQuotaNeve(unittest.TestCase):
    def quota(self, bulbi, quote=(762.0, 1457.0, 3012.0), suolo=0.0, bulbo_suolo=None):
        w = [np.array([v], dtype=float) for v in bulbi]
        z = [np.array([v]) for v in quote]
        return float(snow_line_m(z, w, np.array([suolo]),
                                 np.array([bulbo_suolo]))[0])

    def test_attraversamento_semplice(self):
        # Bulbo bagnato che passa da +4 al suolo a -2 a 1457 m: lo zero cade
        # a due terzi del tratto, cioe' circa 1225 m.
        q = self.quota((2.0, -2.0, -10.0), bulbo_suolo=4.0)
        self.assertAlmostEqual(q, 762.0 + (0.0 - 2.0) / (-2.0 - 2.0) * (1457.0 - 762.0), delta=1.0)

    def test_neve_fino_a_terra(self):
        # Sotto zero gia' al suolo: la quota neve e' la quota del terreno.
        self.assertEqual(self.quota((-1.0, -5.0, -12.0), suolo=300.0, bulbo_suolo=-0.5), 300.0)

    def test_sopra_soglia_ovunque_resta_ignota(self):
        # Affermare una quota fuori dai dati sarebbe inventarla.
        self.assertTrue(np.isnan(self.quota((8.0, 5.0, 2.0), bulbo_suolo=12.0)))

    def test_prende_l_attraversamento_piu_alto(self):
        # Con due attraversamenti conta quello che si incontra scendendo.
        q = self.quota((-1.0, 2.0, -3.0), bulbo_suolo=3.0)
        self.assertGreater(q, 1457.0)

    def test_la_quota_neve_sta_sotto_lo_zero_termico(self):
        """La proprieta' fisica per cui il modulo usa il bulbo bagnato.

        Stessa colonna, stessa temperatura: in aria secca la quota neve deve
        risultare piu' bassa dello zero termico, perche' la fusione raffredda.
        """
        quote = [np.array([v]) for v in (762.0, 1457.0, 3012.0)]
        t = np.array([6.0]), np.array([2.0]), np.array([-8.0])
        pressioni = (925.0, 850.0, 700.0)
        secco = [wet_bulb_c(t[i], 0.0012, pressioni[i]) for i in range(3)]
        umido = [wet_bulb_c(t[i], 0.0075, pressioni[i]) for i in range(3)]
        suolo, t_suolo = np.array([0.0]), np.array([9.0])
        neve_secca = snow_line_m(quote, secco, suolo,
                                 wet_bulb_c(t_suolo, 0.0012, 1013.0))
        neve_umida = snow_line_m(quote, umido, suolo,
                                 wet_bulb_c(t_suolo, 0.0075, 1013.0))
        # Zero termico del bulbo secco, per confronto.
        zero_secco = 1457.0 + (0.0 - 2.0) / (-8.0 - 2.0) * (3012.0 - 1457.0)
        self.assertLess(float(neve_secca[0]), zero_secco)
        self.assertLess(float(neve_secca[0]), float(neve_umida[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
