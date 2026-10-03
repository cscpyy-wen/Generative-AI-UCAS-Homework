"""Run with python -m unittest discover -s tests -v after pip install -e ."""
import unittest

from ucas_generative_ai.smoke import run_smoke


class ReproductionTests(unittest.TestCase):
    def test_independent_synthetic_checks(self):
        result=run_smoke()
        self.assertEqual(result["overall"],"PASS")
        self.assertEqual(result["standard_complete_epoch_resume_all_states_exact"],"PASS")
        self.assertEqual(result["dropout_ema_complete_epoch_resume_all_states_exact"],"PASS")


if __name__ == "__main__":
    unittest.main()
