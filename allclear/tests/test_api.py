"""Tests for the high-level ``allclear.api`` surface.

Covers the status-ladder query methods (``query_azalt``, ``query_radec``,
``query_satellite``, ``query_pass_window``), the OBSCURED vs CLOUDY
disambiguation against an ``ObscurationMask``, the staleness gate, and
TLE propagation when ``sgp4`` is available.

Tests that depend on ``sgp4`` are skipped when the optional satellite
extra is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

from allclear.api import (
    HORIZON_ALT_DEG,
    STATUS_BELOW_HORIZON,
    STATUS_CLEAR,
    STATUS_CLOUDY,
    STATUS_NO_DATA,
    STATUS_OBSCURED,
    STATUS_SGP4_ERROR,
    STATUS_STALE,
    SkyTransmissionResult,
    _coerce_time,
    get_test_transmission,
)
from allclear.obscuration import ObscurationMask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_TIME = Time("2024-01-01T12:00:00", scale="utc")
SITE_LAT = 20.7458   # Haleakala
SITE_LON = -156.4317


def _mask_blocking(az_lo, az_hi, alt_lo, alt_hi):
    """ObscurationMask that blocks the given az/alt rectangle (weight=0)
    and leaves everything else visible (weight=1)."""
    m = ObscurationMask.empty(az_step_deg=2.0, alt_step_deg=2.0)
    az_centers = 0.5 * (m.az_edges_deg[:-1] + m.az_edges_deg[1:])
    alt_centers = 0.5 * (m.alt_edges_deg[:-1] + m.alt_edges_deg[1:])
    AZ, ALT = np.meshgrid(az_centers, alt_centers)
    blocked = (AZ >= az_lo) & (AZ <= az_hi) & (ALT >= alt_lo) & (ALT <= alt_hi)
    m.weight = np.where(blocked, 0.0, 1.0)
    return m


@pytest.fixture
def clear_result():
    """Synthetic clear-sky result pinned to a known obs_time/site."""
    r = get_test_transmission("clear")
    r.obs_time = OBS_TIME
    r.site_lat = SITE_LAT
    r.site_lon = SITE_LON
    return r


@pytest.fixture
def overcast_result():
    r = get_test_transmission("overcast")
    r.obs_time = OBS_TIME
    r.site_lat = SITE_LAT
    r.site_lon = SITE_LON
    return r


# ---------------------------------------------------------------------------
# Status ladder: query_azalt
# ---------------------------------------------------------------------------

class TestQueryAzAlt:
    def test_clear(self, clear_result):
        out = clear_result.query_azalt(180.0, 60.0)
        assert out["status"] == STATUS_CLEAR
        assert out["transmission"] is not None
        assert out["transmission"] >= clear_result.threshold

    def test_cloudy(self, overcast_result):
        out = overcast_result.query_azalt(180.0, 60.0)
        assert out["status"] == STATUS_CLOUDY
        assert out["transmission"] is not None
        assert out["transmission"] < overcast_result.threshold

    def test_below_horizon(self, clear_result):
        out = clear_result.query_azalt(180.0, HORIZON_ALT_DEG - 1.0)
        assert out["status"] == STATUS_BELOW_HORIZON
        assert out["transmission"] is None

    def test_obscured_overrides_transmission(self, clear_result):
        # Without the mask the synthetic 'clear' scenario reports CLEAR
        # everywhere — attaching a mask must flip that direction to
        # OBSCURED, not leave it as a misleading CLEAR.
        clear_result.obscuration = _mask_blocking(0, 30, 5, 30)
        out = clear_result.query_azalt(15.0, 15.0)
        assert out["status"] == STATUS_OBSCURED
        assert out["transmission"] is None

    def test_obscuration_does_not_affect_unblocked_direction(
            self, clear_result):
        clear_result.obscuration = _mask_blocking(0, 30, 5, 30)
        out = clear_result.query_azalt(200.0, 60.0)
        assert out["status"] == STATUS_CLEAR

    def test_no_data_returns_none(self):
        # Empty transmission map → query() returns NaN → status NO_DATA.
        from allclear.transmission import interpolate_transmission
        empty = interpolate_transmission(
            np.array([]), np.array([]), np.array([])
        )
        r = SkyTransmissionResult(transmission_map=empty,
                                  obs_time=OBS_TIME,
                                  site_lat=SITE_LAT, site_lon=SITE_LON)
        out = r.query_azalt(180.0, 60.0)
        assert out["status"] == STATUS_NO_DATA
        assert out["transmission"] is None

    def test_az_wraps_modulo_360(self, clear_result):
        # az=370 should be the same as az=10.
        a = clear_result.query_azalt(10.0, 45.0)
        b = clear_result.query_azalt(370.0, 45.0)
        assert a["az_deg"] == b["az_deg"]
        assert a["status"] == b["status"]


# ---------------------------------------------------------------------------
# Status ladder: query_radec
# ---------------------------------------------------------------------------

class TestQueryRADec:
    def test_includes_radec_in_output(self, clear_result):
        # Pick a position somewhere above the horizon at OBS_TIME/Haleakala.
        # Vega (RA≈279.23, Dec≈+38.78) — visibility depends on the epoch,
        # but the test only checks the dict shape.
        out = clear_result.query_radec(279.23, 38.78)
        assert "ra_deg" in out
        assert "dec_deg" in out
        assert out["ra_deg"] == pytest.approx(279.23, abs=0.01)
        assert out["dec_deg"] == pytest.approx(38.78, abs=0.01)
        assert "az_deg" in out
        assert "alt_deg" in out
        assert "status" in out

    def test_obscuration_path(self, clear_result):
        # Mask the entire sky → any above-horizon target returns OBSCURED.
        clear_result.obscuration = _mask_blocking(0, 360, 0, 90)
        out = clear_result.query_radec(279.23, 38.78)
        assert out["status"] in {STATUS_OBSCURED, STATUS_BELOW_HORIZON}

    def test_staleness_short_circuits_before_coord_transform(
            self, clear_result):
        # If the result is older than max_age_seconds, query_radec must
        # return STALE without bothering with the SkyCoord conversion.
        # We pass a max_age of 1 second; obs_time is in 2024.
        out = clear_result.query_radec(279.23, 38.78, max_age_seconds=1)
        assert out["status"] == STATUS_STALE
        # ra/dec are echoed even on stale results so the caller can log
        # which target the answer was for.
        assert out["ra_deg"] == pytest.approx(279.23, abs=0.01)


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_age_seconds_with_explicit_now(self, clear_result):
        now = OBS_TIME + 330 * (Time("2024-01-01T12:00:01", scale="utc")
                                - Time("2024-01-01T12:00:00", scale="utc"))
        age = clear_result.age_seconds(now=now)
        assert age == pytest.approx(330.0, abs=0.01)

    def test_age_seconds_obs_time_none(self):
        r = SkyTransmissionResult(
            transmission_map=get_test_transmission("clear").transmission_map,
            obs_time=None,
        )
        assert r.age_seconds(now=OBS_TIME) == float("inf")

    def test_is_stale_threshold(self, clear_result):
        now_fresh = OBS_TIME + 60 * (Time("2024-01-01T12:00:01", scale="utc")
                                     - Time("2024-01-01T12:00:00", scale="utc"))
        now_stale = OBS_TIME + 400 * (Time("2024-01-01T12:00:01", scale="utc")
                                      - Time("2024-01-01T12:00:00", scale="utc"))
        assert not clear_result.is_stale(300.0, now=now_fresh)
        assert clear_result.is_stale(300.0, now=now_stale)

    def test_query_returns_nan_when_stale(self, clear_result):
        # query() is the low-level float-returning method; stale → NaN.
        val = clear_result.query(180.0, 60.0, max_age_seconds=1)
        assert np.isnan(val)

    def test_query_azalt_returns_stale_status(self, clear_result):
        out = clear_result.query_azalt(180.0, 60.0, max_age_seconds=1)
        assert out["status"] == STATUS_STALE
        assert "age_seconds" in out


# ---------------------------------------------------------------------------
# Time coercion
# ---------------------------------------------------------------------------

class TestCoerceTime:
    def test_passthrough_astropy_time(self):
        t = Time("2024-01-01T12:00:00", scale="utc")
        assert _coerce_time(t) is t

    def test_iso_string(self):
        t = _coerce_time("2024-01-01T12:00:00")
        assert isinstance(t, Time)
        assert t.isot.startswith("2024-01-01T12:00:00")

    def test_iso_string_with_tz(self):
        # +05:00 → 07:00 UTC.
        t = _coerce_time("2024-01-01T12:00:00+05:00")
        assert t.isot.startswith("2024-01-01T07:00:00")


# ---------------------------------------------------------------------------
# Satellite queries (require sgp4)
# ---------------------------------------------------------------------------

sgp4 = pytest.importorskip("sgp4")

# ISS TLE.  Epoch 2024-001 (Jan 1 2024) — chosen so SGP4 propagation
# stays well within tolerance at OBS_TIME.  The exact element values
# matter only that they're internally consistent; assertions below only
# check propagation succeeds and the result has the right structure.
ISS_TLE_1 = "1 25544U 98067A   24001.50000000  .00018000  00000-0  32000-3 0  9990"
ISS_TLE_2 = "2 25544  51.6400 100.0000 0005000  10.0000 350.0000 15.50000000000000"

# A TLE that fails SGP4 propagation (eccentricity > 1, etc.).  Used to
# exercise the error-handling branch without depending on sgp4 internals.
BAD_TLE_1 = "1 99999U 00000A   99001.00000000  .00000000  00000-0  00000-0 0  9990"
BAD_TLE_2 = "2 99999 999.9999 999.9999 9999999 999.9999 999.9999 99.99999999999990"


class TestQuerySatellite:
    def test_basic_shape(self, clear_result):
        out = clear_result.query_satellite(
            ISS_TLE_1, ISS_TLE_2, name="ISS")
        assert set(out.keys()) >= {
            "sat_name", "time", "az_deg", "alt_deg",
            "transmission", "status",
        }
        assert out["sat_name"] == "ISS"
        assert out["status"] in {
            STATUS_CLEAR, STATUS_CLOUDY, STATUS_OBSCURED,
            STATUS_BELOW_HORIZON, STATUS_NO_DATA,
        }

    def test_default_time_is_obs_time(self, clear_result):
        out = clear_result.query_satellite(ISS_TLE_1, ISS_TLE_2, name="ISS")
        assert out["time"].startswith("2024-01-01T12:00:00")

    def test_explicit_time_moves_position(self, clear_result):
        # Two epochs separated by an hour should produce different az/alt.
        a = clear_result.query_satellite(
            ISS_TLE_1, ISS_TLE_2,
            time=Time("2024-01-01T12:00:00", scale="utc"))
        b = clear_result.query_satellite(
            ISS_TLE_1, ISS_TLE_2,
            time=Time("2024-01-01T13:00:00", scale="utc"))
        # Az/alt may be None if SGP4_ERROR — guard before comparing.
        if a["az_deg"] is not None and b["az_deg"] is not None:
            assert (a["az_deg"], a["alt_deg"]) != (b["az_deg"], b["alt_deg"])

    def test_sgp4_error_branch(self, clear_result):
        out = clear_result.query_satellite(
            BAD_TLE_1, BAD_TLE_2, name="BAD")
        assert out["status"] == STATUS_SGP4_ERROR
        assert out["error_code"] != 0
        assert out["transmission"] is None

    def test_staleness_relative_to_queried_time(self, clear_result):
        # Querying a satellite an hour after the frame's obs_time with
        # max_age_seconds=60 should be flagged STALE.  The age reported
        # is relative to the QUERIED time, not wall-clock now — that's
        # the right semantic for pass scheduling.
        out = clear_result.query_satellite(
            ISS_TLE_1, ISS_TLE_2,
            time=Time("2024-01-01T13:00:00", scale="utc"),
            max_age_seconds=60,
        )
        assert out["status"] == STATUS_STALE
        assert out["age_seconds"] == pytest.approx(3600.0, abs=1.0)

    def test_obscuration_applies_to_satellite_path(self, clear_result):
        # Mask everything above horizon → if SGP4 places the satellite
        # above horizon, status must be OBSCURED (not CLEAR/CLOUDY).
        clear_result.obscuration = _mask_blocking(0, 360, 0, 90)
        out = clear_result.query_satellite(
            ISS_TLE_1, ISS_TLE_2, name="ISS")
        assert out["status"] in {STATUS_OBSCURED, STATUS_BELOW_HORIZON}

    def test_missing_sgp4_raises_importerror(self, clear_result, monkeypatch):
        # Simulate sgp4 not being installed.  We can't actually uninstall
        # it; instead we patch the import to fail.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("sgp4"):
                raise ImportError("simulated: sgp4 not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="sgp4"):
            clear_result.query_satellite(ISS_TLE_1, ISS_TLE_2)


# ---------------------------------------------------------------------------
# Pass window
# ---------------------------------------------------------------------------

class TestQueryPassWindow:
    def test_sample_count_matches_window(self, clear_result):
        start = Time("2024-01-01T12:00:00", scale="utc")
        end = Time("2024-01-01T12:02:00", scale="utc")  # 120 s
        pw = clear_result.query_pass_window(
            ISS_TLE_1, ISS_TLE_2, start=start, end=end, step=30.0)
        # floor(120 / 30) + 1 = 5
        assert pw["n_samples"] == 5
        assert len(pw["samples"]) == 5
        assert pw["step_seconds"] == 30.0
        assert pw["duration_seconds"] == pytest.approx(120.0, abs=0.5)

    def test_status_counts_sum_to_n_samples(self, clear_result):
        start = Time("2024-01-01T12:00:00", scale="utc")
        end = Time("2024-01-01T12:01:00", scale="utc")
        pw = clear_result.query_pass_window(
            ISS_TLE_1, ISS_TLE_2, start=start, end=end, step=10.0)
        total = sum(pw[k] for k in (
            "n_clear", "n_cloudy", "n_obscured", "n_below_horizon",
            "n_no_data", "n_stale", "n_sgp4_error",
        ))
        assert total == pw["n_samples"]

    def test_longest_clear_window_with_synthetic_alts(self, clear_result):
        # Build a pass window whose status sequence is known by stubbing
        # query_satellite directly.  Sequence (8 samples):
        #   C C C . C - C C   (.=cloudy, -=below-horizon)
        # → 6 CLEAR, 1 CLOUDY, 1 BELOW_HORIZON; longest CLEAR run = 3.
        seq = [
            STATUS_CLEAR, STATUS_CLEAR, STATUS_CLEAR,
            STATUS_CLOUDY,
            STATUS_CLEAR,
            STATUS_BELOW_HORIZON,
            STATUS_CLEAR, STATUS_CLEAR,
        ]
        idx = {"i": 0}

        def fake_query_satellite(*args, **kwargs):
            s = seq[idx["i"]]
            idx["i"] += 1
            return {
                "sat_name": kwargs.get("name"),
                "time": "2024-01-01T12:00:00",
                "az_deg": 180.0, "alt_deg": 45.0,
                "transmission": 0.9 if s == STATUS_CLEAR else 0.1,
                "status": s,
            }

        clear_result.query_satellite = fake_query_satellite
        start = Time("2024-01-01T12:00:00", scale="utc")
        end = Time("2024-01-01T12:01:10", scale="utc")  # 70 s → 8 samples at 10s
        pw = clear_result.query_pass_window(
            ISS_TLE_1, ISS_TLE_2, start=start, end=end, step=10.0)
        assert pw["n_samples"] == 8
        assert pw["n_clear"] == 6
        assert pw["n_cloudy"] == 1
        assert pw["n_below_horizon"] == 1
        # Longest contiguous CLEAR run = 3 samples × 10s = 30s
        assert pw["longest_clear_window_seconds"] == 30.0
        # clear_fraction is over above-horizon samples: 6 / 7
        assert pw["clear_fraction"] == pytest.approx(6 / 7, abs=0.001)

    def test_rejects_bad_window(self, clear_result):
        t = Time("2024-01-01T12:00:00", scale="utc")
        with pytest.raises(ValueError, match="end must be"):
            clear_result.query_pass_window(
                ISS_TLE_1, ISS_TLE_2, start=t, end=t, step=10.0)

    def test_rejects_nonpositive_step(self, clear_result):
        start = Time("2024-01-01T12:00:00", scale="utc")
        end = Time("2024-01-01T12:01:00", scale="utc")
        with pytest.raises(ValueError, match="step"):
            clear_result.query_pass_window(
                ISS_TLE_1, ISS_TLE_2, start=start, end=end, step=0.0)


# ---------------------------------------------------------------------------
# Plumbing: obscuration field is present on SkyTransmissionResult
# ---------------------------------------------------------------------------

class TestObscurationPlumbing:
    def test_field_exists_and_defaults_to_none(self):
        from allclear.api import SkyTransmissionResult
        from allclear.transmission import interpolate_transmission
        empty = interpolate_transmission(np.array([]), np.array([]),
                                         np.array([]))
        r = SkyTransmissionResult(transmission_map=empty)
        assert r.obscuration is None

    def test_to_dict_summarizes_obscuration(self, clear_result):
        # No mask attached → obscuration key is None.
        d = clear_result.to_dict()
        assert d["obscuration"] is None
        # With a mask attached → summary appears.
        clear_result.obscuration = _mask_blocking(0, 90, 0, 30)
        d = clear_result.to_dict()
        assert d["obscuration"] is not None
        assert "occluded_fraction" in d["obscuration"]
        assert 0.0 < d["obscuration"]["occluded_fraction"] < 1.0
