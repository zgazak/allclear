"""Instrument model: camera parameters saved/loaded as JSON."""

import json
import pathlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from .projection import CameraModel, ProjectionType


@dataclass
class InstrumentModel:
    """Persistent camera characterization saved to/loaded from JSON.

    Stores all camera geometry parameters, site information, and
    fit-quality metadata so that ``allclear solve`` can process frames
    without re-running the blind solver.
    """

    # --- Site ---
    site_lat: float = 0.0
    site_lon: float = 0.0
    site_name: str = ""

    # --- Camera geometry ---
    projection: str = "equidistant"
    focal_length_px: float = 750.0
    center_x: float = 1548.0
    center_y: float = 1040.0
    az0_deg: float = 0.0
    alt0_deg: float = 90.0
    roll_deg: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    image_width: int = 3096
    image_height: int = 2080
    mirrored: bool = False  # True if image is E-W flipped

    # --- Detection settings (reused in solve mode) ---
    detection_fwhm: float = 5.0
    detection_threshold_sigma: float = 5.0
    detection_n_brightest: int = 200

    # --- Photometry ---
    photometric_zeropoint: float = 0.0  # inst_mag - cat_vmag offset for clear sky

    # --- Fit quality ---
    n_stars_matched: int = 0
    n_stars_expected: int = 0
    rms_residual_px: float = 0.0
    median_residual_px: float = 0.0
    fit_timestamp: str = ""
    frame_used: str = ""

    # --- Bookkeeping ---
    allclear_version: str = ""

    def __post_init__(self):
        if not self.allclear_version:
            from allclear import __version__
            self.allclear_version = __version__

    # ---- Conversions ----

    def to_camera_model(self) -> CameraModel:
        """Convert to a CameraModel for projection calculations."""
        import numpy as np
        return CameraModel(
            cx=self.center_x,
            cy=self.center_y,
            az0=np.radians(self.az0_deg),
            alt0=np.radians(self.alt0_deg),
            rho=np.radians(self.roll_deg),
            f=self.focal_length_px,
            proj_type=ProjectionType(self.projection),
            k1=self.k1,
            k2=self.k2,
        )

    @classmethod
    def from_camera_model(cls, model: CameraModel, **kwargs) -> "InstrumentModel":
        """Build from a CameraModel plus optional metadata."""
        import numpy as np
        inst = cls(
            projection=model.proj_type.value,
            focal_length_px=model.f,
            center_x=model.cx,
            center_y=model.cy,
            az0_deg=float(np.degrees(model.az0)),
            alt0_deg=float(np.degrees(model.alt0)),
            roll_deg=float(np.degrees(model.rho)),
            k1=model.k1,
            k2=model.k2,
        )
        for k, v in kwargs.items():
            if hasattr(inst, k):
                setattr(inst, k, v)
        return inst

    # ---- Persistence ----

    def save(self, path):
        """Save instrument model to a JSON file."""
        path = pathlib.Path(path)
        data = {
            "allclear_version": self.allclear_version,
            "site": {
                "latitude": self.site_lat,
                "longitude": self.site_lon,
                "name": self.site_name,
            },
            "camera": {
                "projection": self.projection,
                "focal_length_px": self.focal_length_px,
                "center_x": self.center_x,
                "center_y": self.center_y,
                "az0_deg": self.az0_deg,
                "alt0_deg": self.alt0_deg,
                "roll_deg": self.roll_deg,
                "k1": self.k1,
                "k2": self.k2,
                "image_width": self.image_width,
                "image_height": self.image_height,
                "mirrored": self.mirrored,
            },
            "detection": {
                "fwhm": self.detection_fwhm,
                "threshold_sigma": self.detection_threshold_sigma,
                "n_brightest": self.detection_n_brightest,
            },
            "photometry": {
                "zeropoint": self.photometric_zeropoint,
            },
            "fit_quality": {
                "n_stars_matched": self.n_stars_matched,
                "n_stars_expected": self.n_stars_expected,
                "rms_residual_px": self.rms_residual_px,
                "median_residual_px": self.median_residual_px,
                "fit_timestamp": self.fit_timestamp,
                "frame_used": self.frame_used,
            },
        }
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path) -> "InstrumentModel":
        """Load instrument model from a JSON file."""
        path = pathlib.Path(path)
        data = json.loads(path.read_text())
        site = data.get("site", {})
        cam = data.get("camera", {})
        det = data.get("detection", {})
        phot = data.get("photometry", {})
        fq = data.get("fit_quality", {})
        return cls(
            allclear_version=data.get("allclear_version", "0.2.0"),
            site_lat=site.get("latitude", 0.0),
            site_lon=site.get("longitude", 0.0),
            site_name=site.get("name", ""),
            projection=cam.get("projection", "equidistant"),
            focal_length_px=cam.get("focal_length_px", 750.0),
            center_x=cam.get("center_x", 1548.0),
            center_y=cam.get("center_y", 1040.0),
            az0_deg=cam.get("az0_deg", 0.0),
            alt0_deg=cam.get("alt0_deg", 90.0),
            roll_deg=cam.get("roll_deg", 0.0),
            k1=cam.get("k1", 0.0),
            k2=cam.get("k2", 0.0),
            image_width=cam.get("image_width", 3096),
            image_height=cam.get("image_height", 2080),
            mirrored=cam.get("mirrored", False),
            detection_fwhm=det.get("fwhm", 5.0),
            detection_threshold_sigma=det.get("threshold_sigma", 5.0),
            detection_n_brightest=det.get("n_brightest", 200),
            photometric_zeropoint=phot.get("zeropoint", 0.0),
            n_stars_matched=fq.get("n_stars_matched", 0),
            n_stars_expected=fq.get("n_stars_expected", 0),
            rms_residual_px=fq.get("rms_residual_px", 0.0),
            median_residual_px=fq.get("median_residual_px", 0.0),
            fit_timestamp=fq.get("fit_timestamp", ""),
            frame_used=fq.get("frame_used", ""),
        )
