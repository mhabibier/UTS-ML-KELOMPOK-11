"""Pemeriksaan transformasi Task 3 dari modul yang dipakai notebook dan CLI."""
import unittest

import numpy as np
import pandas as pd

from task3_clustering import bobot_blok, clr, numerik


class PreprocessChecks(unittest.TestCase):
    def test_clr_nol_dan_penutupan(self):
        x = pd.DataFrame([[0., 1., 0.], [2., 2., 2.]])
        z = clr(x)
        np.testing.assert_allclose(z.sum(axis=1), 0, atol=1e-12)
        np.testing.assert_allclose(z[1], [0., 0., 0.], atol=1e-12)
        self.assertTrue(np.isfinite(z).all())

    def test_clr_negatif_ditolak(self):
        with self.assertRaises(ValueError):
            clr(pd.DataFrame([[-1., 2.]]))

    def test_numerik_tidak_mengubah_sumber(self):
        x = pd.DataFrame({"nilai": [0., 1., np.nan, 1e10], "konstan": [3.] * 4})
        saved = x.copy(deep=True)
        z, rows = numerik(x)
        self.assertTrue(np.isfinite(z).all())
        pd.testing.assert_frame_equal(x, saved)
        self.assertEqual(rows[0]["missing"], 1)

    def test_bobot_blok(self):
        z = bobot_blok(np.array([[0., 1.], [2., 3.]]), .25)
        self.assertAlmostEqual(np.mean(np.sum(z * z, axis=1)), .25)
        self.assertTrue(np.isfinite(bobot_blok(np.ones((3, 2)))).all())


if __name__ == "__main__":
    unittest.main()
