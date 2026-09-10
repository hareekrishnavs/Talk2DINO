"""Adversarial tests for the DFP Phase 1B conservation-TOLERANCE
successor identity (E19) loader/validator + CPU-only CLI.

CPU/data-read-only. Never imports torch, never initializes CUDA, never
loads a model/dataset/checkpoint, never executes recurrence/graph/
masked propagation, never submits SLURM. Every mutation test writes
its own temporary copy -- the committed TOML and the /scratch
calibration artifact are never edited by any test."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import src.dfp_phase1b_conservation_tolerance as m  # noqa: E402

IDENTITY_PATH = ROOT / m.IDENTITY_RELATIVE_PATH
CLI_PATH = ROOT / "diagnostics/run_dfp_phase1b_conservation_tolerance.py"
CALIBRATION_ARTIFACT_PATH = Path(
    "/scratch/haree/dfp_phase1b_conservation_calibration/coco_stuff_200/phase1b_conservation_calibration.json"
)
_CALIB_AVAILABLE = CALIBRATION_ARTIFACT_PATH.exists()
requires_calib = pytest.mark.skipif(not _CALIB_AVAILABLE, reason="calibration artifact not present on this host")

E = m.DfpPhase1bConservationToleranceError


def _identity_text() -> str:
    return IDENTITY_PATH.read_text()


def _write(tmp_path: Path, text: str, name: str = "e19.toml") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def _mutate_line(text: str, old_line: str, new_line: str) -> str:
    # Mutate the FIRST occurrence only. Several anchors legitimately appear
    # twice (once in [operational_scope], once in [authorization_boundary]);
    # the module's closed exact-value (_CANONICAL) check rejects a mutation
    # in EITHER location, so mutating the first is sufficient and unambiguous.
    assert old_line in text, f"anchor not found: {old_line!r}"
    return text.replace(old_line, new_line, 1)


# =====================================================================
# 1. Baseline / source reconstruction
# =====================================================================

def test_committed_identity_loads_clean():
    identity = m.load_identity()
    assert identity["identity"]["name"] == "e19-dfp-phase1b-conservation-tolerance"
    assert identity["governed_residual"]["governed_family"] == "rho_g"
    assert identity["tolerance"]["conservation_tolerance"] == 9.5367431640625e-07
    assert identity["tolerance"]["comparison_operator"] == "<="


def test_frozen_tolerance_is_two_pow_minus_20():
    info = m.frozen_tolerance()
    assert info["conservation_tolerance"] == 2.0 ** -20
    assert info["conservation_tolerance_binary_exponent"] == -20
    assert info["governed_family"] == "rho_g"
    assert info["label_post_calibration"] == "POST-CALIBRATION"
    assert info["label_pre_evaluation"] == "PRE-EVALUATION"


@requires_calib
def test_official_calibration_artifact_verifies_and_binding_matches():
    summary = m.verify_calibration_binding(
        calibration_artifact_path=CALIBRATION_ARTIFACT_PATH, repo_root=ROOT
    )
    assert summary["row_count"] == 200
    assert summary["governed_family"] == "rho_g"
    assert summary["conservation_tolerance"] == 2.0 ** -20
    assert summary["conservation_tolerance_binary_exponent"] == -20
    assert summary["governed_observed_maximum"] == 5.889683609150431e-07


@requires_calib
def test_exactly_200_rows_and_row_level_maxima_reconstruct():
    doc = json.loads(CALIBRATION_ARTIFACT_PATH.read_bytes())
    assert len(doc["selected_window_records"]) == 200
    rec = m.reconstruct_family_maxima(doc)
    for fam in m.CANONICAL_FAMILY_ORDER:
        rows_max = max(
            r[fam]["maximum_absolute_residual"]
            for r in doc["selected_window_records"]
            if r[fam]["sample_count"] > 0
        )
        assert rec[fam]["maximum_absolute_residual"] == rows_max
        # row reconstruction agrees with the artifact's own aggregate
        assert rec[fam]["maximum_absolute_residual"] == doc["aggregate_residual_summaries"][fam]["maximum_absolute_residual"]
        assert rec[fam]["sample_count"] == doc["aggregate_residual_summaries"][fam]["sample_count"]
        assert rec[fam]["non_finite_count"] == 0


@requires_calib
def test_source_artifact_left_byte_and_mtime_identical():
    before_bytes = CALIBRATION_ARTIFACT_PATH.read_bytes()
    before_stat = CALIBRATION_ARTIFACT_PATH.stat()
    m.verify_calibration_binding(calibration_artifact_path=CALIBRATION_ARTIFACT_PATH, repo_root=ROOT)
    after_stat = CALIBRATION_ARTIFACT_PATH.stat()
    assert CALIBRATION_ARTIFACT_PATH.read_bytes() == before_bytes
    assert (before_stat.st_mtime_ns, before_stat.st_size) == (after_stat.st_mtime_ns, after_stat.st_size)


@requires_calib
def test_calibration_digest_exact_in_e19():
    doc = json.loads(CALIBRATION_ARTIFACT_PATH.read_bytes())
    identity = m.load_identity()
    assert identity["calibration_source_binding"]["selected_windows_digest"] == doc["selected_windows_digest"]
    assert identity["calibration_source_binding"]["selected_ids_digest"] == doc["selected_ids_digest"]
    live_hash = hashlib.sha256(CALIBRATION_ARTIFACT_PATH.read_bytes()).hexdigest()
    assert identity["dependency_bindings"]["calibration_artifact"]["sha256"] == live_hash


# =====================================================================
# 2. Rule boundaries -- pow2_ceil
# =====================================================================

@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, 0.0),
        (2.0 ** -20, 2.0 ** -20),
        (2.0 ** -19, 2.0 ** -19),
        (1.0, 1.0),
        (2.0, 2.0),
        (2.0 ** 100, 2.0 ** 100),
        (math.nextafter(2.0 ** -20, 0.0), 2.0 ** -20),
        (math.nextafter(2.0 ** -20, math.inf), 2.0 ** -19),
        (math.nextafter(1.0, 0.0), 1.0),
        (math.nextafter(1.0, math.inf), 2.0),
        (5e-324, 5e-324),  # smallest subnormal
        (math.nextafter(5e-324, math.inf), 1e-323),
        (2.2250738585072014e-308, 2.2250738585072014e-308),  # smallest normal
        (1e300, 2.0 ** math.ceil(math.log2(1e300))),
        (5.889683609150431e-07, 2.0 ** -20),
        (1.3007687971189341e-06, 2.0 ** -19),
    ],
)
def test_pow2_ceil_boundaries(value, expected):
    assert m.pow2_ceil(value) == expected
    if value > 0:
        got = m.pow2_ceil(value)
        assert got >= value
        # got/2 < value unless value is exactly a power of two
        assert got == value or (got / 2.0) < value


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0, -1e-30])
def test_pow2_ceil_rejects_nonfinite_and_negative(bad):
    with pytest.raises(E):
        m.pow2_ceil(bad)


@pytest.mark.parametrize("bad", [1, True, "1.0", None, 2 ** 40])
def test_pow2_ceil_rejects_non_float(bad):
    with pytest.raises(E):
        m.pow2_ceil(bad)


def test_pow2_ceil_deterministic_across_processes_and_hashseed():
    code = "import sys; sys.path.insert(0,'%s'); import src.dfp_phase1b_conservation_tolerance as t; print(repr(t.pow2_ceil(5.889683609150431e-07)))" % ROOT
    outs = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip())
    assert outs == {repr(9.5367431640625e-07)}


# --- F1 regression: pow2_ceil overflow / binary64-representability guard ---

_TWO_POW_1023 = 2.0 ** 1023   # largest exact power of two representable as binary64
_MAX_FINITE = sys.float_info.max  # (2 - 2**-52) * 2**1023


def test_pow2_ceil_exact_two_pow_1023_returned_unchanged():
    got = m.pow2_ceil(_TWO_POW_1023)
    assert got == _TWO_POW_1023
    assert got.hex() == _TWO_POW_1023.hex()   # bit-identical, not doubled / clamped
    assert math.isfinite(got)


@pytest.mark.parametrize(
    "value",
    [
        math.nextafter(_TWO_POW_1023, math.inf),  # just above 2**1023
        _MAX_FINITE,                              # largest finite binary64
        math.nextafter(_MAX_FINITE, 0.0),         # one ULP below max finite
        1.5 * _TWO_POW_1023,                      # squarely inside (2**1023, 2**1024)
    ],
)
def test_pow2_ceil_finite_input_with_unrepresentable_ceiling_raises_domain_error(value):
    assert math.isfinite(value) and value > 0.0
    with pytest.raises(E) as excinfo:
        m.pow2_ceil(value)
    assert "not representable as a finite binary64" in str(excinfo.value)
    assert not isinstance(excinfo.value, OverflowError)  # no raw OverflowError, ever


def test_pow2_ceil_no_raw_overflowerror_escapes_near_the_top_of_the_range():
    probes = [_TWO_POW_1023, math.nextafter(_TWO_POW_1023, math.inf), _MAX_FINITE,
              math.nextafter(_MAX_FINITE, 0.0)]
    for e in range(1015, 1024):
        p = 2.0 ** e
        probes.extend([p, math.nextafter(p, 0.0), math.nextafter(p, math.inf)])
    for x in probes:
        try:
            r = m.pow2_ceil(x)
        except E:
            continue
        except OverflowError:  # pragma: no cover -- the exact bug this guards
            raise AssertionError(f"raw OverflowError escaped pow2_ceil({x!r})")
        assert math.isfinite(r) and r > 0.0
        assert math.frexp(r)[0] == 0.5   # an exact power of two
        assert r >= x


@pytest.mark.parametrize(
    "value,expected",
    [
        (2.0 ** 1022, 2.0 ** 1022),                              # exact power below the max
        (math.nextafter(2.0 ** 1022, 0.0), 2.0 ** 1022),         # just below -> that power
        (math.nextafter(2.0 ** 1022, math.inf), 2.0 ** 1023),    # just above -> next power
        (2.0 ** 512, 2.0 ** 512),
        (math.nextafter(2.0 ** 512, math.inf), 2.0 ** 513),
        (math.nextafter(2.0 ** 200, 0.0), 2.0 ** 200),
        (5.889683609150431e-07, 2.0 ** -20),                     # governed rho_g maximum
        (1.3007687971189341e-06, 2.0 ** -19),
    ],
)
def test_pow2_ceil_large_but_representable_still_follows_frozen_rule(value, expected):
    got = m.pow2_ceil(value)
    assert got == expected
    assert got.hex() == expected.hex()


def test_pow2_ceil_governed_maximum_bit_exact_two_pow_minus_20():
    assert float.hex(m.pow2_ceil(5.889683609150431e-07)) == "0x1.0000000000000p-20"
    assert m.pow2_ceil(5.889683609150431e-07) == 9.5367431640625e-07 == 2.0 ** -20


def test_pow2_ceil_established_behavior_unchanged_after_f1_guard():
    assert m.pow2_ceil(0.0) == 0.0
    assert m.pow2_ceil(5e-324) == 5e-324                          # smallest subnormal
    assert m.pow2_ceil(math.nextafter(5e-324, math.inf)) == 1e-323  # next subnormal
    assert m.pow2_ceil(2.225073858507201e-308) == 2.2250738585072014e-308   # largest subnormal
    assert m.pow2_ceil(2.2250738585072014e-308) == 2.2250738585072014e-308  # smallest normal
    for bad in (float("nan"), float("inf"), float("-inf"), -1.0, -1e-30, -_MAX_FINITE):
        with pytest.raises(E):
            m.pow2_ceil(bad)
    for bad in (1, True, False, "1.0", None, 2 ** 40):
        with pytest.raises(E):
            m.pow2_ceil(bad)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_pow2_ceil_overflow_guard_still_lets_fatal_exceptions_propagate(monkeypatch, exc_type):
    real_ldexp = math.ldexp

    def boom(*a, **k):
        raise exc_type("fatal during ldexp")

    monkeypatch.setattr(math, "ldexp", boom)
    try:
        with pytest.raises(exc_type):
            m.pow2_ceil(1.5e-6)
    finally:
        monkeypatch.setattr(math, "ldexp", real_ldexp)


# =====================================================================
# 3. Relational mutation matrix on E19 (temp copies only)
# =====================================================================

@pytest.mark.parametrize(
    "old,new",
    [
        # tolerance one representable float lower / higher
        ("conservation_tolerance = 9.5367431640625e-07", f"conservation_tolerance = {math.nextafter(2.0**-20, 0.0)!r}"),
        ("conservation_tolerance = 9.5367431640625e-07", f"conservation_tolerance = {math.nextafter(2.0**-20, math.inf)!r}"),
        # arbitrary multiplier / not a power of two
        ("conservation_tolerance = 9.5367431640625e-07", "conservation_tolerance = 1.1920928955078125e-06"),
        # frozen exponent wrong
        ("conservation_tolerance_binary_exponent = -20", "conservation_tolerance_binary_exponent = -19"),
        # governed M mutated (tolerance would no longer be pow2_ceil(M))
        ("reconstructed_maximum_absolute_residual = 5.889683609150431e-07\ngoverned = true",
         "reconstructed_maximum_absolute_residual = 2.5e-07\ngoverned = true"),
        ("observed_maximum_absolute_residual = 5.889683609150431e-07\nobserved_maximum_is_exact_power_of_two = false",
         "observed_maximum_absolute_residual = 2.5e-07\nobserved_maximum_is_exact_power_of_two = false"),
        # rule name changed
        ('name = "smallest_exact_power_of_two_ge_observed_maximum_absolute_governed_residual"',
         'name = "p99_percentile_times_safety_factor"'),
        # aggregation operator changed
        ('maximum_operation = "max_over_all_200_rows_of_per_window_maximum_absolute_residual_for_the_governed_family"',
         'maximum_operation = "mean_over_all_200_rows"'),
        # rounding operator asserted
        ("decimal_rounding_used = false", "decimal_rounding_used = true"),
        # zero behavior changed
        ('zero_input_behavior = "if M == 0, tolerance = 0"', 'zero_input_behavior = "if M == 0, tolerance = smallest_subnormal"'),
        # comparison operator changed to strict
        ('comparison_operator = "<="\nacceptance_relation = "absolute_residual <= conservation_tolerance"',
         'comparison_operator = "<"\nacceptance_relation = "absolute_residual <= conservation_tolerance"'),
        # sample count mutated
        ("sample_count = 7887805", "sample_count = 7887804"),
        # non-finite count mutated
        ("non_finite_count = 0\nreconstructed_maximum_absolute_residual = 5.889683609150431e-07",
         "non_finite_count = 1\nreconstructed_maximum_absolute_residual = 5.889683609150431e-07"),
        # family order mutated
        ('residual_family_names_in_canonical_order = ["rho_M", "rho_0", "rho_320", "rho_g"]',
         'residual_family_names_in_canonical_order = ["rho_g", "rho_M", "rho_0", "rho_320"]'),
        # governed family name mutated
        ('governed_family = "rho_g"\ngoverned_family_source', 'governed_family = "rho_320"\ngoverned_family_source'),
        # source artifact hash mutated (pinned copy in [tolerance])
        ('source_calibration_artifact_sha256 = "f065b143933cf7881e5a31dd663783719a684e6d430e9e7b2bb314ed7e792d22"',
         'source_calibration_artifact_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"'),
        # E16 hash mutated (dependency binding -- caught by live re-hash)
        ('sha256 = "01add2460ad032055db2bd73dcc4e3f40b1b7e3dede07d25afbea4f268ecb679"\nuntouched_by_this_identity = true\n\n[dependency_bindings.e15_protocol]',
         'sha256 = "1111111111111111111111111111111111111111111111111111111111111111"\nuntouched_by_this_identity = true\n\n[dependency_bindings.e15_protocol]'),
        # authorization boundary flipped
        ("full_phase1b_execution_authorized = false", "full_phase1b_execution_authorized = true"),
        ("phase1b_5000_image_evaluation_authorized = false", "phase1b_5000_image_evaluation_authorized = true"),
        ("phase1b_500_image_diagnostic_authorized = false", "phase1b_500_image_diagnostic_authorized = true"),
        ("numeric_tolerance_defined = true\ntolerance_derived_from_5000_image_evaluation = false",
         "numeric_tolerance_defined = true\ntolerance_derived_from_5000_image_evaluation = true"),
        # unknown field
        ('conservation_tolerance = 9.5367431640625e-07',
         'conservation_tolerance = 9.5367431640625e-07\nsurprise_extra_field = 1'),
        # bool-as-int for schema_version
        ("schema_version = 1\nstatus =", "schema_version = true\nstatus ="),
        # malformed string (whitespace) in a hash-shaped field
        ('source_calibration_repository_commit = "ef738016f7c2d0b7d37cd85cb675854c18a8135c"',
         'source_calibration_repository_commit = "ef738016f7c2d0b7d37cd85cb675854c18a8135 "'),
    ],
)
def test_e19_relational_mutation_matrix_rejected(tmp_path, old, new):
    text = _mutate_line(_identity_text(), old, new)
    p = _write(tmp_path, text)
    with pytest.raises(E):
        m.load_identity(path=p, repo_root=ROOT)


def test_e19_missing_field_rejected(tmp_path):
    text = _mutate_line(_identity_text(), "conservation_tolerance_is_power_of_two = true\n", "")
    p = _write(tmp_path, text)
    with pytest.raises(E):
        m.load_identity(path=p, repo_root=ROOT)


@pytest.mark.parametrize(
    "old,new",
    [
        ('relative_path = "evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml"',
         'relative_path = "../evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml"'),
        ('relative_path = "evaluation_identities/e16_dfp_phase1b_conservation_calibration.toml"',
         'relative_path = "/etc/passwd"'),
        ('relative_path = "src/dfp_phase1b_masked_forward.py"',
         'relative_path = "src/../src/dfp_phase1b_masked_forward.py"'),
    ],
)
def test_e19_path_traversal_and_absolute_rejected(tmp_path, old, new):
    text = _mutate_line(_identity_text(), old, new)
    p = _write(tmp_path, text)
    with pytest.raises(E):
        m.load_identity(path=p, repo_root=ROOT)


# =====================================================================
# 4. Calibration-document mutation matrix (dict-level; no hash needed
#    because reconstruct_family_maxima checks row/aggregate consistency)
# =====================================================================

@requires_calib
def _base_doc() -> dict:
    return json.loads(CALIBRATION_ARTIFACT_PATH.read_bytes())


@requires_calib
def test_reconstruct_rejects_mutated_row_residual_max():
    doc = _base_doc()
    doc["selected_window_records"][0]["rho_g"]["maximum_absolute_residual"] = 9.9e-3
    with pytest.raises(E):
        m.reconstruct_family_maxima(doc)


@requires_calib
def test_reconstruct_rejects_mutated_aggregate_max():
    doc = _base_doc()
    doc["aggregate_residual_summaries"]["rho_g"]["maximum_absolute_residual"] = 1.0e-3
    with pytest.raises(E):
        m.reconstruct_family_maxima(doc)


@requires_calib
def test_reconstruct_rejects_mutated_sample_count():
    doc = _base_doc()
    doc["selected_window_records"][3]["rho_320"]["sample_count"] += 1
    with pytest.raises(E):
        m.reconstruct_family_maxima(doc)


@requires_calib
def test_reconstruct_rejects_injected_nonfinite_count():
    doc = _base_doc()
    doc["selected_window_records"][5]["rho_g"]["non_finite_count"] = 2
    with pytest.raises(E):
        m.reconstruct_family_maxima(doc)


@requires_calib
def test_reconstruct_rejects_row_count_not_200():
    doc = _base_doc()
    doc["selected_window_records"] = doc["selected_window_records"][:199]
    with pytest.raises(E):
        m.reconstruct_family_maxima(doc)


@requires_calib
def test_verify_binding_rejects_hash_mismatched_artifact(tmp_path):
    doc = _base_doc()
    doc["selected_window_records"][0]["rho_g"]["maximum_absolute_residual"] *= 1.0000001
    p = tmp_path / "mutated_calib.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(E):
        m.verify_calibration_binding(calibration_artifact_path=p, repo_root=ROOT)


@requires_calib
def test_verify_binding_recomputation_cross_check_rejects_tampered_identity_dict():
    # The committed E19 is internally exact (every leaf pinned by _CANONICAL),
    # so a tampered *file* is rejected by load_identity before the artifact is
    # ever read. To exercise the LAST line of defence -- the row-level
    # recomputation cross-check inside verify_calibration_binding -- hand it a
    # mutated identity *dict* directly (bypassing load_identity) and confirm
    # the real 200 rows no longer reproduce the frozen numbers.
    good = m.load_identity()

    tampered = copy.deepcopy(good)
    tampered["tolerance"]["conservation_tolerance"] = 4.76837158203125e-07  # 2**-21, wrong
    with pytest.raises(E):
        m.verify_calibration_binding(
            identity=tampered, calibration_artifact_path=CALIBRATION_ARTIFACT_PATH, repo_root=ROOT
        )

    tampered2 = copy.deepcopy(good)
    tampered2["derivation_inputs"]["rho_g"]["reconstructed_maximum_absolute_residual"] = 3.0e-07
    with pytest.raises(E):
        m.verify_calibration_binding(
            identity=tampered2, calibration_artifact_path=CALIBRATION_ARTIFACT_PATH, repo_root=ROOT
        )


# =====================================================================
# 5. Isolation -- zero forbidden imports / calls
# =====================================================================

# Hermetic child program run by ``test_no_torch_or_heavy_modules_hermetic``.
# It proves the Task 6A verification path imports none of
# torch/torchvision/cv2/mmcv/mmengine/mmseg, in a FRESH isolated
# interpreter with an ``__import__`` guard installed *before* the Task 6A
# module is imported -- so the result is independent of whatever the
# parent pytest process has already loaded. A bare ``name in sys.modules``
# check in-process is a false pass the moment a sibling test module has
# imported torch first.
_HERMETIC_ISOLATION_CHILD = r'''
import sys, os, builtins

FORBIDDEN = ("torch", "torchvision", "cv2", "mmcv", "mmengine", "mmseg")

# (1) prohibited modules must be absent BEFORE the Task 6A import
_pre = sorted(n for n in FORBIDDEN if n in sys.modules)
if _pre:
    sys.stderr.write("FORBIDDEN PRESENT BEFORE TASK6A IMPORT: %r\n" % (_pre,))
    os._exit(11)

# (2)+(3)+(9) install an import guard BEFORE importing the Task 6A module;
# any attempt to import a forbidden top-level module aborts the child
# immediately with a non-zero status.
_real_import = builtins.__import__
def _guard(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split(".", 1)[0]
    if top in FORBIDDEN:
        sys.stderr.write("FORBIDDEN IMPORT ATTEMPTED: %s\n" % name)
        sys.stderr.flush()
        os._exit(12)
    return _real_import(name, globals, locals, fromlist, level)
builtins.__import__ = _guard

# (8/10) direct repository path (works under `python -I`, which ignores PYTHONPATH)
sys.path.insert(0, os.environ["IV_ROOT"])

# (4) import the real Task 6A module under the guard
import src.dfp_phase1b_conservation_tolerance as t

root = os.environ["IV_ROOT"]
cal = os.environ["IV_CAL"]

# (5)+(6) exercise the real verification path against the real artifact
ident = t.load_identity(repo_root=root)
summary = t.verify_calibration_binding(calibration_artifact_path=cal, repo_root=root)
info = t.frozen_tolerance(repo_root=root)

# (7) prohibited modules must still be absent AFTERWARD
_post = sorted(n for n in FORBIDDEN if n in sys.modules)
if _post:
    sys.stderr.write("FORBIDDEN PRESENT AFTER TASK6A VERIFICATION: %r\n" % (_post,))
    os._exit(13)

# the scientific result the subprocess just recomputed must be intact
if summary["conservation_tolerance"] != 2.0 ** -20:
    sys.stderr.write("tolerance drifted in subprocess: %r\n" % (summary["conservation_tolerance"],))
    os._exit(14)
if summary["governed_family"] != "rho_g" or info["comparison_operator"] != "<=":
    sys.stderr.write("governed-family / operator drift in subprocess\n")
    os._exit(15)

print("HERMETIC_OK M=%r tol=%r" % (summary["governed_observed_maximum"], summary["conservation_tolerance"]))
sys.exit(0)
'''


@requires_calib
def test_no_torch_or_heavy_modules_hermetic(tmp_path):
    """F2 (hermetic): a fresh isolated interpreter installs an
    ``__import__`` guard, then imports and fully exercises the Task 6A
    module (load_identity + verify_calibration_binding + frozen_tolerance)
    against the real verified calibration artifact. The child aborts
    non-zero if any of torch/torchvision/cv2/mmcv/mmengine/mmseg is
    imported, or is present in ``sys.modules`` before or after. Being a
    separate process, it stays correct even when a sibling Phase 1B test
    module has already imported torch into the parent interpreter."""
    child = tmp_path / "iv_hermetic_isolation.py"
    child.write_text(_HERMETIC_ISOLATION_CHILD)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "IV_ROOT": str(ROOT),
        "IV_CAL": str(CALIBRATION_ARTIFACT_PATH),
    }
    for passthrough in ("SYSTEMROOT", "LD_LIBRARY_PATH"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]
    r = subprocess.run(
        [sys.executable, "-I", str(child)],
        capture_output=True, text=True, cwd=str(tmp_path), env=env,
    )
    assert r.returncode == 0, (
        "hermetic isolation subprocess failed (returncode=%s)\n"
        "--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout, r.stderr)
    )
    assert "HERMETIC_OK" in r.stdout, (
        "hermetic subprocess did not confirm success\n"
        "--- stdout ---\n%s\n--- stderr ---\n%s" % (r.stdout, r.stderr)
    )
    assert "Traceback" not in r.stderr, r.stderr


def test_module_source_has_no_forbidden_tokens():
    src_text = (ROOT / "src/dfp_phase1b_conservation_tolerance.py").read_text()
    cli_text = CLI_PATH.read_text()
    for token in ("import torch", "torch.cuda", ".cuda(", "sbatch", "srun", "salloc", "load_checkpoint", "build_model"):
        assert token not in src_text, f"{token!r} in tolerance module"
        assert token not in cli_text, f"{token!r} in tolerance CLI"


def test_cli_from_fresh_cwd_sanitized_env(tmp_path):
    if not _CALIB_AVAILABLE:
        pytest.skip("no calibration artifact")
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", str(tmp_path))}
    r = subprocess.run(
        [sys.executable, str(CLI_PATH), "--repo-root", str(ROOT), "verify-calibration-binding",
         "--artifact", str(CALIBRATION_ARTIFACT_PATH)],
        capture_output=True, text=True, cwd=str(tmp_path), env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "EXACTLY matches" in r.stdout
    assert "Traceback" not in r.stderr


def test_cli_ordinary_failure_exit_2_no_traceback(tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", str(tmp_path))}
    r = subprocess.run(
        [sys.executable, str(CLI_PATH), "--repo-root", str(ROOT), "verify-calibration-binding",
         "--artifact", "/no/such/artifact.json"],
        capture_output=True, text=True, cwd=str(tmp_path), env=env,
    )
    assert r.returncode == 2
    assert "Traceback" not in r.stderr
    assert "DFP PHASE 1B CONSERVATION TOLERANCE FAIL" in r.stderr


def test_cli_has_no_execution_subcommand():
    r = subprocess.run([sys.executable, str(CLI_PATH), "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    help_text = r.stdout
    # the argparse subcommand list is rendered as "{a,b,c}" -- extract and
    # check it exactly, rather than substring-scanning the prose description.
    choice_blocks = re.findall(r"\{([a-z0-9,\-]+)\}", help_text)
    assert choice_blocks, f"no subcommand choice block in help:\n{help_text}"
    subcommands = set(choice_blocks[0].split(","))
    assert subcommands == {"verify-protocol", "verify-calibration-binding", "show-tolerance"}, subcommands
    for forbidden in ("execute", "run", "sbatch", "srun", "salloc", "evaluate", "inference", "diagnostic"):
        assert forbidden not in subcommands, f"forbidden subcommand exposed: {forbidden}"


def test_no_sbatch_launcher_created_for_task6a():
    assert not (ROOT / "run_dfp_phase1b_conservation_tolerance.sbatch").exists()
    assert not (ROOT / "dfp_phase1b_conservation_tolerance.sbatch").exists()


# =====================================================================
# 6. Fatal exceptions propagate unchanged
# =====================================================================

@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exceptions_not_swallowed_in_verify_binding(monkeypatch, exc_type):
    def boom(*a, **k):
        raise exc_type("fatal during calibration validation")
    monkeypatch.setattr(m, "validate_calibration_artifact", boom)
    if not _CALIB_AVAILABLE:
        pytest.skip("no calibration artifact")
    with pytest.raises(exc_type):
        m.verify_calibration_binding(calibration_artifact_path=CALIBRATION_ARTIFACT_PATH, repo_root=ROOT)


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_fatal_exceptions_not_swallowed_in_pow2_ceil_path(monkeypatch, exc_type):
    real_frexp = math.frexp

    def boom(x):
        raise exc_type("fatal during frexp")
    monkeypatch.setattr(math, "frexp", boom)
    try:
        with pytest.raises(exc_type):
            m.pow2_ceil(1.5e-6)
    finally:
        monkeypatch.setattr(math, "frexp", real_frexp)


def test_ordinary_errors_wrapped_in_domain_exception(tmp_path):
    p = tmp_path / "not_toml.toml"
    p.write_text("{{{ this is not valid toml")
    with pytest.raises(E):
        m.load_identity(path=p, repo_root=ROOT)
