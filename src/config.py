# Options configuration for pyxover applications
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List

import multiprocessing as mp
import copy
import yaml
import numpy as np


_DEFAULTS_PATH = Path(__file__).with_name('default.yaml')


def _load_default_options() -> Dict[str, Any]:
    if not _DEFAULTS_PATH.exists():
        raise FileNotFoundError(f"Default configuration file not found at {_DEFAULTS_PATH}")
    loaded: Dict[str, Any] = yaml.safe_load(_DEFAULTS_PATH.read_text()) or {}
    return loaded


_DEFAULT_OPTIONS = _load_default_options()


def _default_vecopts() -> Dict[str, Any]:
    if 'vecopts' not in _DEFAULT_OPTIONS:
        raise KeyError("'vecopts' missing from default configuration")

    vecopts = copy.deepcopy(_DEFAULT_OPTIONS['vecopts'])

    for key in ('INSTID', 'INSTNAME'):
        if isinstance(vecopts.get(key), list):
            vecopts[key] = tuple(vecopts[key])

    return vecopts


def _normalize_types(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _normalize_types(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_types(v) for v in value]
    return value


@dataclass
class XovOptions:
    # env opt
    debug: bool | None = None
    local: bool | None = None
    parallel: bool | None = None
    partials: bool | None = None
    unittest: bool | None = None

    body: str | None = None
    instrument: str | None = None
    selected_hemisphere: str | None = None

    # directories
    basedir: str | None = None
    rawdir: str | None = None
    outdir: str | None = None
    auxdir: str | None = None
    tmpdir: str | None = None
    spauxdir: str | None = None

    # pyxover options
    n_proc: int | str | None = None

    expopt: str | None = None
    resopt: List[int] | None = None
    amplopt: List[int] | None = None

    parOrb: Dict[str, float] | None = None
    parGlo: Dict[str, List[float]] | None = None

    par_constr: Dict[str, float] | None = None
    mean_constr: Dict[str, float] | None = None

    cloop_sim: bool | None = None
    pert_cloop_orb: Dict[str, Any] | None = None
    pert_cloop_glo: Dict[str, Any] | None = None
    pert_cloop: Dict[str, Any] | None = None
    pert_tracks: List[Any] | None = None

    sol4_orb: List[Any] | None = None
    sol4_orbpar: List[Any] | None = None
    sol4_glo: List[str] | None = None

    OrbRep: str | None = None

    SpInterp: int | None = None
    spice_meta: str | None = None
    spice_spk: List[Any] | None = None

    new_gtrack: int | None = None

    import_proj: bool | None = None
    import_abmat: str | None = None
    new_xov: int | None = None
    weekly_sets: bool | None = None
    monthly_sets: bool | None = None
    multi_xov: bool | None = None
    new_algo: bool | None = None
    compute_input_xov: bool | None = None
    msrm_sampl: int | None = None
    n_interp: int | None = None

    full_covar: bool | None = None
    roughn_map: bool | None = None

    new_illumNG: bool | None = None
    apply_topo: bool | None = None
    small_scale_topo: bool | None = None
    range_noise: bool | None = None
    range_noise_mean_std: List[float] | None = None
    local_dem: bool | None = None
    max_range_altitude: int | None = None
    sampling_rate: int | None = None

    vecopts: Dict[str, Any] | None = field(default_factory=_default_vecopts)

    @staticmethod
    def base_vecopts() -> Dict[str, Any]:
        return _default_vecopts().copy()

    def __post_init__(self) -> None:
        self.apply_defaults()
        self.sync_paths()
        self.validate()

    def apply_defaults(self) -> None:
        for field_info in fields(self):
            name = field_info.name
            if getattr(self, name) is None and name in _DEFAULT_OPTIONS:
                setattr(self, name, copy.deepcopy(_DEFAULT_OPTIONS[name]))

        self._normalize_n_proc()

    def _normalize_n_proc(self) -> None:
        if self.n_proc is None or self.n_proc == 'auto':
            self.n_proc = max(1, mp.cpu_count() - 3)

    def sync_paths(self) -> None:
        base = self.basedir if self.basedir.endswith('/') else f"{self.basedir}/"
        self.rawdir = f'{base}raw/'
        self.outdir = f'{base}out/'
        self.auxdir = f'{base}aux/'
        self.tmpdir = f'{base}tmp/'
        self.pert_cloop = {'orb': self.pert_cloop_orb, 'glo': self.pert_cloop_glo}

    def validate(self) -> None:
        planet_name = str(self.vecopts.get('PLANETNAME', '')).upper()
        if self.body.upper() != planet_name:
            raise ValueError(
                f"Body name {self.body} is inconsistent with vecopts PLANETNAME {self.vecopts.get('PLANETNAME')}"
            )

        if not isinstance(self.msrm_sampl, int) or self.msrm_sampl % 2 != 0:
            raise ValueError('msrm_sampl config not accepted! Should be an even int')

    def to_dict(self) -> Dict[str, Any]:
        return _normalize_types(asdict(self))

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'XovOptions':
        options = cls()
        for key, value in data.items():
            if hasattr(options, key):
                setattr(options, key, value)
            else:
                raise NameError(f"Name {key} not accepted in XovOptions.from_dict()")
        options._normalize_n_proc()
        options.sync_paths()
        options.validate()
        return options

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'XovOptions':
        loaded: Dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(loaded)


class XovOpt:
    _options = XovOptions()

    @staticmethod
    def check_consistency():
        XovOpt._options.sync_paths()
        XovOpt._options.validate()

    @staticmethod
    def get(name):
        if hasattr(XovOpt._options, name):
            return getattr(XovOpt._options, name)
        raise NameError(f"Name {name} not accepted in XovOpt.get() method")

    @staticmethod
    def set(name, value):
        if hasattr(XovOpt._options, name):
            setattr(XovOpt._options, name, value)
            XovOpt._options.sync_paths()
            XovOpt._options.validate()
            print(f"### XovOpt.{name} updated to {value}.")
        else:
            raise NameError("Name not accepted in XovOpt.set() method")

    @staticmethod
    def display():
        for key, value in XovOpt._options.to_dict().items():
            print(f"{key}: {value}")

    @staticmethod
    def to_dict():
        return XovOpt._options.to_dict()

    @staticmethod
    def to_yaml(path: str | Path):
        XovOpt._options.to_yaml(path)

    @staticmethod
    def clone(opts):
        XovOpt._options = XovOptions.from_dict(opts)

    @staticmethod
    def from_yaml(path: str | Path):
        XovOpt._options = XovOptions.from_yaml(path)


if __name__ == '__main__':
    print(XovOpt.get("vecopts"))
    opt = XovOpt()
    print(opt.get("vecopts"))
    print(opt.get("body"))
    opt.set("body", "MOON")
    print(opt.get("body"))
