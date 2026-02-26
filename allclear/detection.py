"""Star detection in all-sky camera images."""

import numpy as np
from astropy.table import Table
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder


def create_horizon_mask(ny, nx, cx, cy, radius):
    """Create a circular mask: True inside radius from (cx, cy)."""
    yy, xx = np.ogrid[:ny, :nx]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return dist <= radius


def create_rough_mask(image, sigma_clip_sigma=3.0):
    """Create a mask of saturated or dead pixels."""
    med = np.median(image)
    std = np.std(image)
    mask = np.zeros(image.shape, dtype=bool)
    mask[image > med + 50 * std] = True  # saturated
    return mask


def detect_stars(
    image,
    fwhm=5.0,
    threshold_sigma=5.0,
    n_brightest=200,
    sharplo=0.2,
    sharphi=1.0,
    roundlo=-1.0,
    roundhi=1.0,
    mask=None,
    box_size=64,
):
    """Detect stars in an all-sky camera image.

    Parameters
    ----------
    image : ndarray (ny, nx)
        Input image.
    fwhm : float
        Expected FWHM of stellar PSFs in pixels.
    threshold_sigma : float
        Detection threshold in sigma above background.
    n_brightest : int
        Maximum number of sources to return.
    sharplo, sharphi : float
        Sharpness bounds for filtering.
    roundlo, roundhi : float
        Roundness bounds for filtering.
    mask : ndarray bool, optional
        Pixels to ignore (True = masked).
    box_size : int
        Background estimation box size.

    Returns
    -------
    Table with columns: x, y, flux, peak, sharpness, roundness
        Sorted by flux (brightest first).
    """
    image = np.asarray(image, dtype=np.float64)
    ny, nx = image.shape

    # Background subtraction
    bkg = Background2D(
        image,
        box_size=(box_size, box_size),
        filter_size=(3, 3),
        bkg_estimator=MedianBackground(),
        mask=mask,
    )
    image_sub = image - bkg.background

    # Estimate noise
    bkg_rms = bkg.background_rms
    median_rms = np.median(bkg_rms)
    threshold = threshold_sigma * median_rms

    # Detect
    finder = DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold,
        sharplo=sharplo,
        sharphi=sharphi,
        roundlo=roundlo,
        roundhi=roundhi,
    )
    sources = finder(image_sub, mask=mask)

    if sources is None or len(sources) == 0:
        return Table(
            names=["x", "y", "flux", "peak", "sharpness", "roundness"],
            dtype=[float, float, float, float, float, float],
        )

    # Sort by flux descending
    sources.sort("flux")
    sources.reverse()

    # Spatial deduplication: keep only the brightest detection within
    # min_sep pixels.  Extended objects (domes, reflections) produce many
    # detections in a small area; this collapses them to one and frees
    # detection slots for real point sources.
    min_sep = fwhm * 3.0
    xs = np.array(sources["xcentroid"], dtype=np.float64)
    ys = np.array(sources["ycentroid"], dtype=np.float64)
    keep = np.ones(len(sources), dtype=bool)
    for i in range(len(sources)):
        if not keep[i]:
            continue
        dists = np.sqrt((xs[i + 1:] - xs[i]) ** 2 + (ys[i + 1:] - ys[i]) ** 2)
        too_close = np.where(dists < min_sep)[0] + (i + 1)
        keep[too_close] = False
    sources = sources[keep]

    # Take brightest N
    if len(sources) > n_brightest:
        sources = sources[:n_brightest]

    out = Table()
    out["x"] = np.array(sources["xcentroid"], dtype=np.float64)
    out["y"] = np.array(sources["ycentroid"], dtype=np.float64)
    out["flux"] = np.array(sources["flux"], dtype=np.float64)
    out["peak"] = np.array(sources["peak"], dtype=np.float64)
    out["sharpness"] = np.array(sources["sharpness"], dtype=np.float64)
    out["roundness"] = np.array(sources["roundness1"], dtype=np.float64)

    return out
