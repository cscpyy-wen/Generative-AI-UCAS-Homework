"""Keep released reference records separate from new experiment output."""
import json
from pathlib import Path


def ensure_run_output(path):
    target=Path(path).resolve()
    for ancestor in (target,*target.parents):
        manifest=ancestor/"manifest.json"
        if manifest.is_file():
            try:metadata=json.loads(manifest.read_text())
            except (ValueError,OSError):metadata={}
            if "reference_files" in metadata and "checkpoints" in metadata:
                raise ValueError("Output is inside released reference records; choose runs/ or reproduced/.")
        reference_manifest=ancestor/"results/manifest.json"
        if reference_manifest.is_file():
            try:metadata=json.loads(reference_manifest.read_text())
            except (ValueError,OSError):metadata={}
            if "reference_files" in metadata and "checkpoints" in metadata:
                if target == ancestor:
                    raise ValueError("Choose a separate runs/ or reproduced/ output directory.")
                for name in ("results","checkpoints","configs","docs","notebooks","src","tests","scripts"):
                    protected=ancestor/name
                    if target == protected or protected in target.parents:
                        raise ValueError("Output would modify released project files; choose runs/ or reproduced/.")
    return Path(path)
