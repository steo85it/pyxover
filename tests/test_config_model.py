import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from config import XovOpt, XovOptions


def test_mismatched_body_and_vecopts_rejected():
    with pytest.raises(ValueError):
        XovOptions(body="MOON")


def test_msrm_sampl_must_be_even():
    with pytest.raises(ValueError):
        vecopts = XovOptions.base_vecopts()
        XovOptions(msrm_sampl=3, vecopts=vecopts)


def test_yaml_round_trip_preserves_values(tmp_path: Path):
    vecopts = XovOptions.base_vecopts()
    vecopts["PLANETNAME"] = "MOON"
    opts = XovOptions(body="MOON", vecopts=vecopts, msrm_sampl=4)

    yaml_path = tmp_path / "config.yaml"
    opts.to_yaml(yaml_path)

    loaded = XovOptions.from_yaml(yaml_path)
    assert loaded.body == "MOON"
    assert loaded.vecopts["PLANETNAME"] == "MOON"
    assert loaded.rawdir.endswith("raw/")


def test_legacy_wrapper_preserved():
    vecopts = XovOptions.base_vecopts()
    vecopts["PLANETNAME"] = "MOON"
    cloned = XovOptions(body="MOON", vecopts=vecopts, msrm_sampl=4).to_dict()

    XovOpt.clone(cloned)

    assert XovOpt.get("body") == "MOON"
    assert XovOpt.get("msrm_sampl") == 4

    with pytest.raises(NameError):
        XovOpt.set("unknown", 1)

    XovOpt.set("msrm_sampl", 6)
    assert XovOpt.get("msrm_sampl") == 6
