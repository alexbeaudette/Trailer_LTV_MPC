"""Tests for YAML-backed controller config loading."""

from pathlib import Path

import numpy as np
import pytest

from trailer_ltv_mpc import TrailerLtvMpcConfig
from trailer_ltv_mpc.config_loader import load_controller_config


def test_load_default_yaml_matches_dataclass_defaults():
    loaded = load_controller_config("configs/default.yaml")
    expected = TrailerLtvMpcConfig()

    assert loaded.N == expected.N
    assert loaded.Ts == expected.Ts
    assert loaded.geom == expected.geom
    assert np.allclose(loaded.Q, expected.Q)
    assert np.allclose(loaded.Qf, expected.Qf)
    assert np.allclose(loaded.Q_rev, expected.Q_rev)
    assert np.allclose(loaded.Qf_rev, expected.Qf_rev)
    assert np.allclose(loaded.Q_fwd, expected.Q_fwd)
    assert np.allclose(loaded.Qf_fwd, expected.Qf_fwd)
    assert np.allclose(loaded.R, expected.R)
    assert np.allclose(loaded.Rd, expected.Rd)
    assert np.isnan(loaded.forward_correction_target_pass_radius_m)
    assert np.isnan(loaded.forward_correction_path_start_offset_m)


def test_partial_yaml_override_uses_defaults(tmp_path: Path):
    override = tmp_path / "override.yaml"
    override.write_text(
        """
N: 12
Ts: 0.1
geom:
  L2: 12.5
""".lstrip(),
        encoding="utf-8",
    )

    loaded = load_controller_config(override)

    assert loaded.N == 12
    assert loaded.Ts == 0.1
    assert loaded.geom.L1 == TrailerLtvMpcConfig().geom.L1
    assert loaded.geom.L2 == 12.5
    assert np.allclose(loaded.R, TrailerLtvMpcConfig().R)


def test_unknown_controller_key_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_config_key: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown controller config key"):
        load_controller_config(bad)


def test_unknown_geometry_key_raises(tmp_path: Path):
    bad = tmp_path / "bad_geom.yaml"
    bad.write_text(
        """
geom:
  hitch_behind: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown geometry config key"):
        load_controller_config(bad)
