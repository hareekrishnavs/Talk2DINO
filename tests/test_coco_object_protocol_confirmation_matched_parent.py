"""Adversarial tests for the matched k11/k12 PARENT identity binding used
by the COCO-Object protocol-confirmation identity (fix for the former raw
tomllib.loads() bypass). Exercises _validate_matched_parent, which is the
exact function validate_static_configuration -- and therefore
_run_evaluation, before any CUDA/model/checkpoint work -- calls. Malformed
parent content must produce a clean domain failure -- either
CocoObjectProtocolConfirmationIdentityError or the real loader's own
MatchedK11K12Error (both explicitly caught by main()'s exception boundary)
-- never a bare KeyError/TypeError/raw TOML traceback."""

from __future__ import annotations

import copy
import hashlib
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.coco_object_protocol_confirmation_identity import (  # noqa: E402
    CocoObjectProtocolConfirmationIdentityError,
    _validate_matched_parent,
    load_identity,
)
from src.matched_k11_k12_identity import MatchedK11K12Error  # noqa: E402

# _validate_matched_parent forwards the real loader's own domain exception
# without re-wrapping it (matching the established _validate_materialization_parent
# convention); diagnostics/run_coco_object_protocol_confirmation.py's main()
# catches both types explicitly at the CLI boundary.
DOMAIN_ERRORS = (CocoObjectProtocolConfirmationIdentityError, MatchedK11K12Error)

IDENTITY_PATH = ROOT / "evaluation_identities/e12_coco_object_protocol_confirmation.toml"
MATCHED_IDENTITY_PATH = ROOT / "evaluation_identities/e12_matched_k11_k12_t320.toml"
pytestmark = pytest.mark.skipif(
    not (IDENTITY_PATH.exists() and MATCHED_IDENTITY_PATH.exists()),
    reason="requires the protocol-confirmation and matched k11/k12 identities",
)


def _emit(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_emit(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value type: {type(value)}")


def _dump_flat_toml(document: dict) -> str:
    """Dump a single-level-of-tables TOML document (matches the shape of
    the matched k11/k12 identity: no nested sub-tables)."""
    lines = []
    for key, value in document.items():
        if isinstance(value, dict):
            lines.append(f"\n[{key}]")
            for k, v in value.items():
                lines.append(f"{k} = {_emit(v)}")
        else:
            lines.append(f"{key} = {_emit(value)}")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def identity():
    return load_identity(repo_root=ROOT)


def _load_raw_matched() -> dict:
    with MATCHED_IDENTITY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _confirmation_identity_pointing_at(tmp_path, identity: dict, matched_path: Path, *, sha256: str | None = None):
    """_validate_matched_parent only ever reads
    parent_identities.matched_identity_{path,sha256,name} -- build the
    modified mapping directly (bypassing load_identity's own repo-relative-
    path safety constraint, which is orthogonal to what this test targets:
    the matched-identity loader's own content validation)."""
    document = copy.deepcopy(identity)
    document["parent_identities"] = dict(document["parent_identities"])
    document["parent_identities"]["matched_identity_path"] = str(matched_path)
    document["parent_identities"]["matched_identity_sha256"] = (
        sha256 if sha256 is not None else hashlib.sha256(matched_path.read_bytes()).hexdigest()
    )
    return document


def _probe(tmp_path, identity, mutate) -> None:
    document = _load_raw_matched()
    mutate(document)
    matched_path = tmp_path / "matched.toml"
    matched_path.write_text(_dump_flat_toml(document))
    confirmation_identity = _confirmation_identity_pointing_at(tmp_path, identity, matched_path)
    with pytest.raises(DOMAIN_ERRORS):
        _validate_matched_parent(ROOT, confirmation_identity)


def test_missing_propagation_section_rejected(tmp_path, identity):
    _probe(tmp_path, identity, lambda d: d.pop("propagation"))


def test_missing_graph_section_rejected(tmp_path, identity):
    _probe(tmp_path, identity, lambda d: d.pop("graph"))


def test_malformed_crop_rejected(tmp_path, identity):
    _probe(tmp_path, identity, lambda d: d["geometry"].__setitem__("crop", [448]))


def test_malformed_stride_rejected(tmp_path, identity):
    _probe(tmp_path, identity, lambda d: d["geometry"].__setitem__("stride", [224, 224, 224]))


def test_maximum_rank_bool_as_int_rejected(tmp_path, identity):
    _probe(tmp_path, identity, lambda d: d["graph"].__setitem__("maximum_rank", True))


def test_alpha_wrong_type_rejected(tmp_path, identity):
    _probe(tmp_path, identity, lambda d: d["propagation"].__setitem__("alpha", "0.98"))


def test_steps_wrong_type_rejected(tmp_path, identity):
    _probe(tmp_path, identity, lambda d: d["propagation"].__setitem__("steps", 320.0))


def test_unknown_field_rejected(tmp_path, identity):
    _probe(tmp_path, identity, lambda d: d["propagation"].__setitem__("extra_bogus_field", 1))


def test_missing_matched_identity_file_rejected(tmp_path, identity):
    missing = tmp_path / "does-not-exist.toml"
    confirmation_identity = _confirmation_identity_pointing_at(tmp_path, identity, missing, sha256="0" * 64)
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _validate_matched_parent(ROOT, confirmation_identity)


def test_malformed_toml_matched_identity_rejected(tmp_path, identity):
    matched_path = tmp_path / "bad_matched.toml"
    matched_path.write_text("this is not [valid toml")
    confirmation_identity = _confirmation_identity_pointing_at(tmp_path, identity, matched_path)
    with pytest.raises(DOMAIN_ERRORS):
        _validate_matched_parent(ROOT, confirmation_identity)


def test_parent_identity_name_mismatch_rejected(tmp_path, identity):
    """The matched identity file is well-formed and its own SHA256 pin
    matches, but its declared name disagrees with what the confirmation
    identity expects."""
    document = _load_raw_matched()
    document["identity"]["name"] = document["identity"]["name"] + "-renamed"
    matched_path = tmp_path / "matched_renamed.toml"
    matched_path.write_text(_dump_flat_toml(document))
    confirmation_identity = _confirmation_identity_pointing_at(tmp_path, identity, matched_path)
    with pytest.raises(CocoObjectProtocolConfirmationIdentityError):
        _validate_matched_parent(ROOT, confirmation_identity)


def test_real_matched_parent_validates_cleanly(identity):
    """Positive control: the real, unmodified matched k11/k12 identity
    must still validate cleanly through the fixed loader-based path."""
    matched_identity = _validate_matched_parent(ROOT, identity)
    assert matched_identity["identity"]["name"] == identity["parent_identities"]["matched_identity_name"]
    assert matched_identity["graph"]["maximum_rank"] == 12
