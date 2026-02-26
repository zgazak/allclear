"""Synthetic all-sky image generation for testing."""

import numpy as np
from astropy.table import Table

from .projection import CameraModel, ProjectionType


def generate_synthetic_frame(
    camera_model=None,
    star_table=None,
    nx=3096,
    ny=2080,
    sky_background=200.0,
    read_noise=15.0,
    flux_scale=1e5,
    psf_sigma=2.5,
    cloud_patches=None,
    seed=42,
):
    """Generate a synthetic all-sky camera frame.

    Parameters
    ----------
    camera_model : CameraModel, optional
        Camera geometry. Defaults to a zenith-pointing equidistant lens.
    star_table : astropy.table.Table, optional
        Must have columns: az_deg, alt_deg, vmag (or vmag_extinct).
        If None, generates random stars.
    nx, ny : int
        Image dimensions.
    sky_background : float
        Mean sky background in ADU.
    read_noise : float
        Read noise standard deviation in ADU.
    flux_scale : float
        Flux of a mag=0 star in ADU.
    psf_sigma : float
        PSF Gaussian sigma in pixels.
    cloud_patches : list of dict, optional
        Each dict has keys: az_deg, alt_deg, radius_deg, opacity (0-1).
    seed : int
        Random seed.

    Returns
    -------
    image : ndarray (ny, nx)
        Synthetic image in float64.
    truth : Table
        Table of planted stars with columns: az_deg, alt_deg, vmag,
        x_true, y_true, flux, in_frame.
    """
    rng = np.random.default_rng(seed)

    if camera_model is None:
        camera_model = CameraModel()

    # Generate random stars if none provided
    if star_table is None:
        n_stars = 500
        az = rng.uniform(0, 360, n_stars)
        alt = rng.uniform(10, 90, n_stars)
        vmag = rng.uniform(-1, 6.0, n_stars)
        star_table = Table({"az_deg": az, "alt_deg": alt, "vmag": vmag})

    az_rad = np.radians(np.asarray(star_table["az_deg"], dtype=np.float64))
    alt_rad = np.radians(np.asarray(star_table["alt_deg"], dtype=np.float64))
    mag_col = "vmag_extinct" if "vmag_extinct" in star_table.colnames else "vmag"
    vmag = np.asarray(star_table[mag_col], dtype=np.float64)

    # Project stars
    x_true, y_true = camera_model.sky_to_pixel(az_rad, alt_rad)

    # Determine which are in frame
    in_frame = (x_true >= 0) & (x_true < nx) & (y_true >= 0) & (y_true < ny)
    in_frame &= np.isfinite(x_true) & np.isfinite(y_true)

    # Compute flux
    flux = flux_scale * 10 ** (-0.4 * vmag)

    # Apply cloud patches
    transmission = np.ones(len(vmag))
    if cloud_patches:
        for patch in cloud_patches:
            paz_rad = np.radians(patch["az_deg"])
            palt_rad = np.radians(patch["alt_deg"])
            pradius = np.radians(patch["radius_deg"])
            opacity = patch["opacity"]

            # Angular distance from patch center
            cos_sep = (np.sin(alt_rad) * np.sin(palt_rad) +
                       np.cos(alt_rad) * np.cos(palt_rad) *
                       np.cos(az_rad - paz_rad))
            cos_sep = np.clip(cos_sep, -1, 1)
            sep = np.arccos(cos_sep)

            # Gaussian opacity profile
            atten = opacity * np.exp(-0.5 * (sep / pradius) ** 2)
            transmission *= (1.0 - atten)

    flux_attenuated = flux * transmission

    # Build image
    image = np.full((ny, nx), sky_background, dtype=np.float64)

    # Stamp size
    stamp_half = int(5 * psf_sigma) + 1

    for i in range(len(vmag)):
        if not in_frame[i]:
            continue
        xi, yi = x_true[i], y_true[i]
        fi = flux_attenuated[i]

        ix_lo = max(0, int(xi) - stamp_half)
        ix_hi = min(nx, int(xi) + stamp_half + 1)
        iy_lo = max(0, int(yi) - stamp_half)
        iy_hi = min(ny, int(yi) + stamp_half + 1)

        yy, xx = np.mgrid[iy_lo:iy_hi, ix_lo:ix_hi]
        r2 = (xx - xi) ** 2 + (yy - yi) ** 2
        psf = np.exp(-0.5 * r2 / psf_sigma ** 2)
        psf /= psf.sum()
        image[iy_lo:iy_hi, ix_lo:ix_hi] += fi * psf

    # Add Poisson noise (approximate with Gaussian for large counts)
    image = np.maximum(image, 0)
    image = rng.poisson(image.astype(np.int64)).astype(np.float64)

    # Add read noise
    image += rng.normal(0, read_noise, image.shape)

    # Build truth table
    truth = Table()
    truth["az_deg"] = star_table["az_deg"]
    truth["alt_deg"] = star_table["alt_deg"]
    truth["vmag"] = star_table["vmag"] if "vmag" in star_table.colnames else vmag
    truth["x_true"] = x_true
    truth["y_true"] = y_true
    truth["flux"] = flux_attenuated
    truth["in_frame"] = in_frame
    truth["transmission"] = transmission

    return image, truth
