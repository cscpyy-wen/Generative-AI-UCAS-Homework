"""Exercise all seven CLI training branches using only tiny synthetic data."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from ucas_generative_ai import runner
from ucas_generative_ai.config import EXPERIMENTS,experiment_config
from ucas_generative_ai.data import MNISTData
from ucas_generative_ai.training import TrainingSession
from ucas_generative_ai.paths import ensure_run_output


class TrainingPipelineTests(unittest.TestCase):
    def test_released_records_are_not_an_output_directory(self):
        with tempfile.TemporaryDirectory(prefix="pixelcnn-reference-") as temporary:
            root=Path(temporary)
            (root/"results").mkdir()
            (root/"results/manifest.json").write_text(json.dumps(dict(reference_files={},checkpoints={})))
            self.assertEqual(ensure_run_output(root/"runs"),root/"runs")
            for protected in (root,root/"results",root/"results/s0",root/"checkpoints",root/"docs/new.json"):
                with self.assertRaises(ValueError):ensure_run_output(protected)

    def test_all_seven_experiments_and_selection_before_test(self):
        torch.set_num_threads(1)
        pixels=torch.randint(0,2,(16,1,4,4),generator=torch.Generator().manual_seed(137)).float()
        def dataset(part):return TensorDataset(part,torch.zeros(len(part),dtype=torch.long))
        data=MNISTData(dataset(pixels[:8]),dataset(pixels[8:12]),dataset(pixels[12:]),
                       torch.arange(8),torch.arange(8,12),{})
        def tiny_config(name):
            value=experiment_config(name)
            value.update(channels=4,n_blocks=2,batch_size=4,epochs=2,
                         n_train=8,n_val=4,n_test=4,image_numel=16)
            if value["architecture"] == "dilated_gated":value["dilations"]=[1,2]
            return value
        evaluate=TrainingSession.evaluate
        with tempfile.TemporaryDirectory(prefix="pixelcnn-pipeline-") as temporary:
            directory=Path(temporary)/"runs"
            test_evaluations=[]
            def checked_evaluation(session,model,loader):
                if loader.dataset is data.test:
                    self.assertTrue((directory/"selection.json").exists())
                    test_evaluations.append(session.output_dir.name)
                return evaluate(session,model,loader)
            with patch.object(runner,"load_mnist",return_value=data),\
                 patch.object(runner,"experiment_config",side_effect=tiny_config),\
                 patch.object(TrainingSession,"evaluate",checked_evaluation),\
                 contextlib.redirect_stdout(io.StringIO()):
                summary=runner.run_experiments("all",directory,device="cpu")
            self.assertEqual(set(summary["results"]),set(EXPERIMENTS))
            self.assertEqual(summary["selection"]["selection_basis"],"validation_nll")
            self.assertIn("f0",test_evaluations)
            self.assertIn("f02",test_evaluations)
            self.assertTrue((directory/"f02/pixelcnn_epoch_01.pt").exists())
            g=json.loads((directory/"g/training_summary.json").read_text())
            first=json.loads((directory/"f0/config.json").read_text())
            second=json.loads((directory/"f02/config.json").read_text())
            self.assertEqual(first["initial_epoch"],g["best_epoch"])
            self.assertEqual(second["initial_epoch"],g["best_epoch"])
            self.assertEqual(first["initial_weights_sha256"],second["initial_weights_sha256"])
            # Released records are never treated as optimizer/resume state.
            self.assertFalse((directory/"results").exists())
            with patch.object(runner,"load_mnist",return_value=data),\
                 patch.object(runner,"experiment_config",side_effect=tiny_config),\
                 patch.object(TrainingSession,"evaluate",checked_evaluation),\
                 contextlib.redirect_stdout(io.StringIO()):
                resumed=runner.run_experiments("all",directory,device="cpu")
            self.assertEqual(resumed["selection"],summary["selection"])
            for name in EXPERIMENTS:
                self.assertEqual(resumed["results"][name]["metrics"]["best_val_nll"],
                                 summary["results"][name]["metrics"]["best_val_nll"])


if __name__ == "__main__":
    unittest.main()
