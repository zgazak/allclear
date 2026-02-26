"""Utility functions used across allclear."""

import pathlib
import warnings

import numpy as np
from astropy.io import fits

# File extensions recognized as FITS
_FITS_EXTS = {".fits", ".fit", ".fts"}
# All supported image extensions
SUPPORTED_EXTS = _FITS_EXTS | {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_image(path):
    """Load an image file, returning (data_float64, header_or_None).

    Supports FITS (.fits/.fit/.fts) and common image formats
    (.jpg/.jpeg/.png/.tif/.tiff).  For FITS files the header is returned;
    for other formats the header is None.
    """
    path = pathlib.Path(path)
    ext = path.suffix.lower()

    if ext in _FITS_EXTS:
        return load_fits_image(path)

    # Generic image via Pillow
    try:
        from PIL import Image
        img = Image.open(path)
    except Exception as exc:
        raise ValueError(f"Cannot read image {path}: {exc}") from exc

    # Convert to grayscale if needed (star detection needs single channel)
    if img.mode not in ("L", "I", "F"):
        img = img.convert("L")

    data = np.asarray(img, dtype=np.float64)
    # FITS convention: origin at bottom-left, y increases upward
    data = data[::-1]
    return data, None


def load_fits_image(path):
    """Load a FITS image, returning (data, header).

    Handles 16/32-bit integer and float data. Raises ValueError for
    truncated or unreadable files.
    """
    try:
        with fits.open(str(path), ignore_missing_simple=True) as hdul:
            data = hdul[0].data
            header = hdul[0].header
    except Exception as exc:
        raise ValueError(f"Cannot read FITS file {path}: {exc}") from exc

    if data is None:
        raise ValueError(f"No image data in FITS file {path}")

    # Ensure native byte order float64
    data = np.asarray(data, dtype=np.float64)
    return data, header


def extract_obs_time(path):
    """Try to extract observation time from image EXIF metadata.

    Returns an astropy Time or None if no timestamp found.
    """
    from astropy.time import Time
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return None
        # Map tag IDs to names
        exif_dict = {TAGS.get(k, k): v for k, v in exif.items()}
        # Prefer DateTimeOriginal, then DateTime
        dt_str = exif_dict.get("DateTimeOriginal") or exif_dict.get("DateTime")
        if dt_str:
            # EXIF format: "YYYY:MM:DD HH:MM:SS"
            dt_str = dt_str.replace(":", "-", 2)  # fix date separators
            return Time(dt_str, scale="utc")
    except Exception:
        pass
    return None


def parse_fits_header(header):
    """Extract observation metadata from a FITS header.

    Returns dict with keys: obs_time, lat_deg, lon_deg, exposure,
    focal_mm, pixel_um.
    """
    from astropy.time import Time

    obs_time_str = header.get("DATE-OBS") or header.get("DATE")
    obs_time = Time(obs_time_str, scale="utc")

    lat_deg = _parse_dms(header["SITELAT"])
    lon_deg = _parse_dms(header["SITELONG"])

    exposure = float(header.get("EXPOSURE", 1.0))
    focal_mm = float(header.get("FOCAL", 1.8))
    pixel_um = float(header.get("XPIXELSZ", 2.4))

    return {
        "obs_time": obs_time,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "exposure": exposure,
        "focal_mm": focal_mm,
        "pixel_um": pixel_um,
    }


def _parse_dms(value):
    """Parse a DMS string like '+20:44:45:00' or a numeric value to degrees."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    parts = s.split(":")
    sign = -1 if s.startswith("-") else 1
    parts_f = [abs(float(p)) for p in parts]
    deg = parts_f[0]
    if len(parts_f) > 1:
        deg += parts_f[1] / 60.0
    if len(parts_f) > 2:
        deg += parts_f[2] / 3600.0
    # Fourth field (fractional arcsec) if present
    if len(parts_f) > 3:
        deg += parts_f[3] / 3600.0 / 100.0  # centiarcsec
    return sign * deg


def altaz_to_direction(az_rad, alt_rad):
    """Convert azimuth/altitude (radians) to unit direction vector (x, y, z).

    Convention: x=East, y=North, z=Up.
    """
    cos_alt = np.cos(alt_rad)
    x = cos_alt * np.sin(az_rad)
    y = cos_alt * np.cos(az_rad)
    z = np.sin(alt_rad)
    return np.array([x, y, z])


def direction_to_altaz(d):
    """Convert direction vector(s) to (az_rad, alt_rad).

    Input shape: (3,) or (3, N).
    """
    d = np.asarray(d, dtype=np.float64)
    if d.ndim == 1:
        x, y, z = d
    else:
        x, y, z = d[0], d[1], d[2]
    az = np.arctan2(x, y) % (2 * np.pi)
    alt = np.arcsin(np.clip(z / np.sqrt(x**2 + y**2 + z**2), -1, 1))
    return az, alt


def airmass_bemporad(alt_rad):
    """Bemporad airmass formula for altitude in radians.

    Returns airmass X. For alt <= 0 returns a large value (100).
    """
    alt_deg = np.degrees(alt_rad)
    alt_deg = np.asarray(alt_deg, dtype=np.float64)
    z_deg = 90.0 - alt_deg
    z_rad = np.radians(z_deg)

    # Kasten & Young (1989) formula
    scalar = alt_deg.ndim == 0
    alt_d = np.atleast_1d(alt_deg)
    result = np.where(
        alt_d > 0,
        1.0 / (np.cos(np.radians(np.clip(90.0 - alt_d, 0, 89.9)))
               + 0.50572 * (6.07995 + alt_d) ** (-1.6364)),
        100.0,
    )
    return float(result) if scalar else result


def extinction_correction(vmag, airmass, k=0.20):
    """Apply atmospheric extinction: m_observed = vmag + k * airmass."""
    return vmag + k * airmass
