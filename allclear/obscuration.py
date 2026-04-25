"""Persistent obscuration mask in sky coordinates.

Represents directions that systematically cannot see stars regardless
of weather: dome, trees, rooflines, masts, horizon terrain, dead
columns, and the region outside the horizon circle.  The mask is
stored in the ground frame ``(azimuth, altitude)`` — not pixel space —
so it stays valid across pointing drifts and mount movements.  On
access, ``project_to_pixel`` renders it into the current frame's pixel
grid using the camera model.

The mask is used at three stages of the pipeline:

  1. **Detection** — zero out obscured pixels before DAOStarFinder so
     dome edges and dead columns do not dominate the top-N brightest
     sources.
  2. **Matching** — skip catalog stars that project into obscured sky
     so the solver does not fight structural artifacts.
  3. **Transmission** — render obscured directions as an "occluded"
     category distinct from measured cloud opacity.

A Tier 1 workflow (single frame → cloud map) uses either an empty
mask (no-op) or a static cold-start mask built from the camera's
horizon circle.  Tier 2 builds a learned mask by aggregating per-star
detection outcomes from many clear-sky frames; the learned mask is
stacked with the static mask via ``combined_with``.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np


DEFAULT_AZ_STEP_DEG = 2.0
DEFAULT_ALT_STEP_DEG = 2.0
OCCLUDED_THRESHOLD = 0.3


@dataclass
class ObscurationMask:
    """All-sky obscuration stored in (alt, az) ground-frame bins.

    Attributes
    ----------
    alt_edges_deg : ndarray, shape (N_alt + 1,)
        Bin edges in altitude, degrees.  Monotonically increasing.
    az_edges_deg : ndarray, shape (N_az + 1,)
        Bin edges in azimuth, degrees.  Covers 0..360.
    weight : ndarray, shape (N_alt, N_az)
        Per-bin transmission weight in [0, 1].  1 = fully visible,
        0 = fully obscured.  NaN for bins with no observations; those
        are treated as visible (weight = 1) by ``query``.
    n_visits : ndarray, shape (N_alt, N_az), optional
        Clear-sky star visits used to fit each bin.  Diagnostic only.
    n_frames : int
        Number of frames that contributed to the learned component.
    version : str
    """

    alt_edges_deg: np.ndarray
    az_edges_deg: np.ndarray
    weight: np.ndarray
    n_visits: Optional[np.ndarray] = None
    n_frames: int = 0
    version: str = "0.1"

    def __post_init__(self):
        self.alt_edges_deg = np.asarray(self.alt_edges_deg, dtype=np.float64)
        self.az_edges_deg = np.asarray(self.az_edges_deg, dtype=np.float64)
        self.weight = np.asarray(self.weight, dtype=np.float64)
        if self.n_visits is not None:
            self.n_visits = np.asarray(self.n_visits, dtype=np.int64)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, az_step_deg=DEFAULT_AZ_STEP_DEG,
              alt_step_deg=DEFAULT_ALT_STEP_DEG):
        """All-visible mask on a regular (az, alt) grid.  A no-op."""
        alt_edges = np.arange(-5.0, 90.0 + alt_step_deg, alt_step_deg)
        az_edges = np.arange(0.0, 360.0 + az_step_deg, az_step_deg)
        weight = np.ones((len(alt_edges) - 1, len(az_edges) - 1),
                         dtype=np.float64)
        return cls(alt_edges_deg=alt_edges, az_edges_deg=az_edges,
                   weight=weight)

    @classmethod
    def from_camera(cls, camera_model, image_shape,
                    horizon_alt_deg=3.0,
                    az_step_deg=DEFAULT_AZ_STEP_DEG,
                    alt_step_deg=DEFAULT_ALT_STEP_DEG):
        """Static cold-start mask from a CameraModel + image shape.

        Marks directions as obscured (weight = 0) if they fall below a
        minimum altitude (horizon cut) or project outside the image
        rectangle (beyond the fisheye circle).  Everything else gets
        weight = 1.  Works without any calibration data.
        """
        obs = cls.empty(az_step_deg=az_step_deg,
                        alt_step_deg=alt_step_deg)
        ny, nx = image_shape
        alt_centers = 0.5 * (obs.alt_edges_deg[:-1]
                             + obs.alt_edges_deg[1:])
        az_centers = 0.5 * (obs.az_edges_deg[:-1]
                            + obs.az_edges_deg[1:])
        AZ, ALT = np.meshgrid(az_centers, alt_centers)
        az_rad = np.radians(AZ)
        alt_rad = np.radians(ALT)
        x, y = camera_model.sky_to_pixel(az_rad, alt_rad)
        in_frame = (np.isfinite(x) & np.isfinite(y)
                    & (x >= 0) & (x < nx)
                    & (y >= 0) & (y < ny))
        below_horizon = ALT < horizon_alt_deg
        obs.weight = np.where(in_frame & ~below_horizon, 1.0, 0.0)
        return obs

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, az_deg, alt_deg):
        """Weight at a sky position (scalar or array inputs)."""
        az_deg = np.asarray(az_deg, dtype=np.float64) % 360.0
        alt_deg = np.asarray(alt_deg, dtype=np.float64)

        below = alt_deg < self.alt_edges_deg[0]
        above = alt_deg > self.alt_edges_deg[-1]
        in_range = ~below & ~above & np.isfinite(alt_deg) & np.isfinite(az_deg)

        alt_idx = np.clip(
            np.digitize(alt_deg, self.alt_edges_deg) - 1,
            0, len(self.alt_edges_deg) - 2,
        )
        az_idx = np.clip(
            np.digitize(az_deg, self.az_edges_deg) - 1,
            0, len(self.az_edges_deg) - 2,
        )

        w = self.weight[alt_idx, az_idx]
        w = np.where(np.isnan(w), 1.0, w)

        out = np.where(in_range, w, np.where(below, 0.0, 1.0))
        return out

    def is_visible(self, az_deg, alt_deg, threshold=OCCLUDED_THRESHOLD):
        """Boolean mask: True where weight >= threshold."""
        return self.query(az_deg, alt_deg) >= threshold

    # ------------------------------------------------------------------
    # Pixel projection
    # ------------------------------------------------------------------

    def project_to_pixel(self, camera_model, image_shape):
        """Render the mask onto a pixel grid for the given camera.

        Returns a 2D float array the same shape as ``image_shape`` with
        values in [0, 1].  Pixels that do not correspond to a valid sky
        direction (outside the fisheye circle) receive weight 0.
        """
        ny, nx = image_shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        with np.errstate(invalid="ignore"):
            az_rad, alt_rad = camera_model.pixel_to_sky(
                xx.astype(np.float64), yy.astype(np.float64))
        az_deg = np.degrees(az_rad)
        alt_deg = np.degrees(alt_rad)
        w = self.query(az_deg, alt_deg)
        # Mark non-finite sky positions (outside fisheye) as obscured
        invalid = ~(np.isfinite(az_rad) & np.isfinite(alt_rad))
        w = np.where(invalid, 0.0, w)
        return w

    def project_to_pixel_mask(self, camera_model, image_shape,
                              threshold=OCCLUDED_THRESHOLD):
        """Boolean pixel array: True where the pixel is obscured."""
        return self.project_to_pixel(camera_model, image_shape) < threshold

    # ------------------------------------------------------------------
    # Derived products
    # ------------------------------------------------------------------

    def radial_response(self):
        """Azimuthally-averaged weight as a function of altitude.

        Returns ``(alt_center_deg, mean_weight, std_weight)``.  Useful
        as a 1-D diagnostic of the effective throughput curve.  NaN
        bins are ignored in both the mean and the standard deviation.
        """
        alt_centers = 0.5 * (self.alt_edges_deg[:-1]
                             + self.alt_edges_deg[1:])
        with np.errstate(invalid="ignore", all="ignore"):
            mean_w = np.nanmean(self.weight, axis=1)
            std_w = np.nanstd(self.weight, axis=1)
        return alt_centers, mean_w, std_w

    # ------------------------------------------------------------------
    # Combination
    # ------------------------------------------------------------------

    def combined_with(self, other):
        """Element-wise minimum of two masks on the same grid.

        Use this to stack a learned mask on top of a static cold-start
        mask: an obscuration found by either source disqualifies the
        bin.  NaN values are treated as 1.0 (unknown ⇒ visible).
        """
        if (len(self.alt_edges_deg) != len(other.alt_edges_deg)
                or len(self.az_edges_deg) != len(other.az_edges_deg)
                or not np.allclose(self.alt_edges_deg, other.alt_edges_deg)
                or not np.allclose(self.az_edges_deg, other.az_edges_deg)):
            raise ValueError("ObscurationMask grids do not match")
        a = np.where(np.isfinite(self.weight), self.weight, 1.0)
        b = np.where(np.isfinite(other.weight), other.weight, 1.0)
        combined = np.minimum(a, b)
        return ObscurationMask(
            alt_edges_deg=self.alt_edges_deg.copy(),
            az_edges_deg=self.az_edges_deg.copy(),
            weight=combined,
            n_visits=self.n_visits,
            n_frames=max(self.n_frames, other.n_frames),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self):
        d = {
            "version": self.version,
            "alt_edges_deg": self.alt_edges_deg.tolist(),
            "az_edges_deg": self.az_edges_deg.tolist(),
            "weight": [
                [None if not np.isfinite(v) else round(float(v), 4)
                 for v in row]
                for row in self.weight
            ],
            "n_frames": int(self.n_frames),
        }
        if self.n_visits is not None:
            d["n_visits"] = self.n_visits.astype(int).tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        alt_edges = np.asarray(d["alt_edges_deg"], dtype=np.float64)
        az_edges = np.asarray(d["az_edges_deg"], dtype=np.float64)
        raw = d["weight"]
        weight = np.array(
            [[(np.nan if v is None else float(v)) for v in row]
             for row in raw],
            dtype=np.float64,
        )
        n_visits = None
        if "n_visits" in d and d["n_visits"] is not None:
            n_visits = np.asarray(d["n_visits"], dtype=np.int64)
        return cls(
            alt_edges_deg=alt_edges, az_edges_deg=az_edges,
            weight=weight, n_visits=n_visits,
            n_frames=int(d.get("n_frames", 0)),
            version=str(d.get("version", "0.1")),
        )

    def save(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        path = pathlib.Path(path)
        return cls.from_dict(json.loads(path.read_text()))


# ----------------------------------------------------------------------
# Builder (Tier 2)
# ----------------------------------------------------------------------

def build_from_observations(az_deg, alt_deg, detected, clear_fraction,
                            vmag, clear_gate=0.7, vmag_min=1.5,
                            vmag_max=6.0, min_visits=8,
                            az_step_deg=DEFAULT_AZ_STEP_DEG,
                            alt_step_deg=DEFAULT_ALT_STEP_DEG,
                            n_frames=0):
    """Build a learned ObscurationMask from per-star detection outcomes.

    Each input row describes one ``(frame, catalog star)`` observation:
    the predicted ground-frame position at the frame's timestamp, whether
    the star was matched, the frame's clear-sky fraction (used as a
    gate so cloudy frames do not contaminate the mask), and the star's
    catalog magnitude.

    Parameters
    ----------
    az_deg, alt_deg : ndarray
        Predicted sky position of the catalog star in the frame.
    detected : ndarray of int
        1 if the star was matched by the solver, 0 otherwise.
    clear_fraction : ndarray
        Frame-level clear-sky fraction for the observing frame.
    vmag : ndarray
        Catalog magnitude.
    clear_gate : float
        Minimum clear fraction for an observation to count.
    vmag_min : float
        Exclude saturated bright stars (default 1.5; at 16 bit they
        clip and often fail matching for photometric reasons, not
        obscuration).
    vmag_max : float
        Exclude faint stars that might just be below the detection
        threshold even in clear sky.
    min_visits : int
        Bins with fewer clear-sky visits are marked NaN (unknown).
    az_step_deg, alt_step_deg : float
        Grid bin sizes in degrees.
    n_frames : int
        Number of frames contributing (stored as metadata).

    Returns
    -------
    ObscurationMask
    """
    az = np.asarray(az_deg, dtype=np.float64) % 360.0
    alt = np.asarray(alt_deg, dtype=np.float64)
    det = np.asarray(detected, dtype=np.int32)
    cf = np.asarray(clear_fraction, dtype=np.float64)
    vm = np.asarray(vmag, dtype=np.float64)

    ok = (cf >= clear_gate) & (vm >= vmag_min) & (vm <= vmag_max) \
        & np.isfinite(alt) & np.isfinite(az)
    az, alt, det = az[ok], alt[ok], det[ok]

    alt_edges = np.arange(-5.0, 90.0 + alt_step_deg, alt_step_deg)
    az_edges = np.arange(0.0, 360.0 + az_step_deg, az_step_deg)
    n_alt, n_az = len(alt_edges) - 1, len(az_edges) - 1

    ai = np.clip(np.digitize(alt, alt_edges) - 1, 0, n_alt - 1)
    zi = np.clip(np.digitize(az, az_edges) - 1, 0, n_az - 1)

    visits = np.zeros((n_alt, n_az), dtype=np.int64)
    detects = np.zeros((n_alt, n_az), dtype=np.int64)
    np.add.at(visits, (ai, zi), 1)
    np.add.at(detects, (ai, zi), det)

    weight = np.full((n_alt, n_az), np.nan, dtype=np.float64)
    enough = visits >= min_visits
    weight[enough] = detects[enough] / np.maximum(visits[enough], 1)

    # Extrapolate downward: in each azimuth column, altitudes below
    # the lowest fitted bin are assumed obscured (weight = 0).  The
    # rationale is that stars sweep through every (az, low-alt) bin
    # via sidereal motion over a season, so consistent absence of any
    # clear-sky visit means the direction is persistently blocked
    # (horizon terrain, tree canopy, neighbouring structures) rather
    # than under-sampled.  Without this the rendered mask shows a
    # floating obscuration ring disconnected from the horizon cut.
    for zi in range(n_az):
        col_filled = np.where(enough[:, zi])[0]
        if col_filled.size == 0:
            # No data at all in this azimuth; leave as NaN so callers
            # can choose their own handling (query() defaults to visible).
            continue
        lowest = int(col_filled[0])
        if lowest > 0:
            weight[:lowest, zi] = 0.0

    return ObscurationMask(
        alt_edges_deg=alt_edges, az_edges_deg=az_edges,
        weight=weight, n_visits=visits,
        n_frames=int(n_frames),
    )
