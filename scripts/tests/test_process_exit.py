"""L'uscita del processore, che decide se il sito si aggiorna.

Il 18/09 due run consecutivi hanno scritto tutti i dati, stampato
"ELABORAZIONE COMPLETATA CON SUCCESSO" e poi sono morti con "double free or
corruption (!prev)" dentro i distruttori di una libreria nativa dello stack
GRIB. Exit 134, passo fallito, deploy saltato: il sito e' rimasto fermo
all'ultimo run buono pur avendo i dati nuovi pronti.

Queste prove fissano le due meta' dell'invariante: un run riuscito deve
uscire con zero senza passare dallo smontaggio dell'interprete, e un run
senza dati validi deve continuare a fallire.
"""

import os
import subprocess
import sys
import textwrap
import unittest


def esegui(programma):
    """Sottoprocesso con il buffering predefinito di Python.

    PYTHONUNBUFFERED cambia proprio la proprieta' che qui si vuole misurare --
    se e' impostata nell'ambiente, stdout non e' piu' bufferizzato e la
    contro-prova sullo svuotamento non dimostrerebbe nulla. Va tolta, non
    subita.
    """
    ambiente = dict(os.environ)
    ambiente.pop("PYTHONUNBUFFERED", None)
    return subprocess.run([sys.executable, "-c", programma], capture_output=True,
                          text=True, timeout=30, env=ambiente)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SORGENTE = os.path.join(ROOT, "scripts", "process_data.py")


class TestUscitaDelProcessore(unittest.TestCase):
    def setUp(self):
        with open(SORGENTE, encoding="utf-8") as handle:
            self.testo = handle.read()
        coda = self.testo.split('if __name__ == "__main__":')
        self.assertEqual(len(coda), 2, "manca il blocco __main__ di process_data")
        self.coda = coda[1]

    def test_il_successo_non_passa_dallo_smontaggio(self):
        self.assertIn("os._exit(0)", self.coda,
                      "il run riuscito torna a smontare l'interprete, e un "
                      "distruttore nativo difettoso puo' di nuovo far fallire "
                      "un passo gia' completato")

    def test_i_buffer_vengono_svuotati_prima(self):
        # os._exit non svuota nulla: senza questo il log del run riuscito
        # arriverebbe troncato proprio dove serve leggerlo.
        posizione_flush = self.coda.find("sys.stdout.flush()")
        posizione_uscita = self.coda.find("os._exit(0)")
        self.assertNotEqual(posizione_flush, -1, "manca lo svuotamento di stdout")
        self.assertIn("sys.stderr.flush()", self.coda, "manca lo svuotamento di stderr")
        self.assertLess(posizione_flush, posizione_uscita,
                        "i buffer vengono svuotati dopo l'uscita, cioe' mai")

    def test_il_fallimento_resta_un_fallimento(self):
        # La guardia che conta: l'uscita netta non deve trasformare un run
        # senza dati in un successo silenzioso che pubblica il vuoto.
        self.assertIn('print("\\nNESSUN DATO VALIDO ESTRATTO.")', self.testo)
        self.assertIn("sys.exit(1)", self.testo,
                      "il ramo senza dati validi non fallisce piu'")

    def test_l_uscita_netta_salta_davvero_lo_smontaggio(self):
        # La proprieta' su cui poggia il rimedio, verificata sul serio: con
        # os._exit gli handler di chiusura non girano, quindi non gira
        # nemmeno il distruttore nativo che abortiva il processo.
        programma = textwrap.dedent(
            """
            import atexit, os, sys
            atexit.register(lambda: os.write(1, b"SMONTAGGIO\\n"))
            sys.stdout.write("FATTO\\n")
            sys.stdout.flush()
            os._exit(0)
            """
        )
        esito = esegui(programma)
        self.assertEqual(esito.returncode, 0)
        self.assertIn("FATTO", esito.stdout)
        self.assertNotIn("SMONTAGGIO", esito.stdout,
                         "lo smontaggio viene eseguito lo stesso: il rimedio "
                         "non protegge da un distruttore nativo difettoso")

    def test_senza_svuotamento_il_log_si_perderebbe(self):
        # Contro-prova del test precedente: dimostra che lo svuotamento
        # esplicito non e' decorazione ma l'unica cosa che salva il log.
        programma = textwrap.dedent(
            """
            import os, sys
            sys.stdout.write("QUESTO SI PERDE\\n")
            os._exit(0)
            """
        )
        esito = esegui(programma)
        self.assertEqual(esito.returncode, 0)
        self.assertNotIn("QUESTO SI PERDE", esito.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
