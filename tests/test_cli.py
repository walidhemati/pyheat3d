import json

from pyheat3d.cli import main


def test_list(capsys):
    assert main(["list"]) == 0
    assert "numpy/slicing" in capsys.readouterr().out


def test_run_numpy(capsys):
    assert main(["run", "--variant", "numpy/slicing", "--dims", "2", "--n", "33",
                 "--steps", "10"]) == 0
    assert "GLUPS" in capsys.readouterr().out


def test_run_jacobi(capsys):
    assert main(["run", "--variant", "numpy/slicing", "--dims", "3", "--n", "9",
                 "--steps", "10", "--mode", "jacobi"]) == 0


def test_registry_json_serialisable():
    from pyheat3d.registry import VARIANTS

    json.dumps({k: v.description for k, v in VARIANTS.items()})
