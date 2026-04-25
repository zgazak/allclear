"""Bright star catalog for all-sky camera matching."""

import pathlib
import re
import warnings

import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.table import Table
from astropy.time import Time
import astropy.units as u

from .utils import airmass_bemporad, expected_apparent_mag

DATA_DIR = pathlib.Path(__file__).parent / "data"
CATALOG_PATH = DATA_DIR / "hipparcos_bright.ecsv"

# Also check for BSC5 in a sibling 'catalog' directory
BSC5_PATH = pathlib.Path(__file__).resolve().parent.parent / "catalog" / "bsc5.txt"

MAG_LIMIT = 6.5


class BrightStarCatalog:
    """Bright star catalog for all-sky camera matching.

    Loads in priority order:
    1. Cached ECSV at allclear/data/hipparcos_bright.ecsv
    2. BSC5 text file at catalog/bsc5.txt
    3. VizieR Hipparcos download (requires astroquery + internet)
    """

    def __init__(self, mag_limit=MAG_LIMIT):
        self.mag_limit = mag_limit
        self._table = None

    @property
    def table(self):
        if self._table is None:
            self._table = self._load()
        return self._table

    def _load(self):
        if CATALOG_PATH.exists():
            return Table.read(CATALOG_PATH, format="ascii.ecsv")
        if BSC5_PATH.exists():
            return self._load_bsc5(BSC5_PATH)
        return self._download_and_cache()

    def _load_bsc5(self, path):
        """Parse the BSC5 text catalog."""
        hr_ids = []
        ra_deg = []
        dec_deg = []
        vmags = []

        with open(path) as f:
            for line in f:
                # Skip header lines (first 2 lines)
                line = line.rstrip("\n")
                if not line or line.startswith("Line#") or line.startswith("HR#"):
                    continue

                try:
                    hr = int(line[0:5])
                    ra_str = line[5:17].strip()
                    dec_str = line[17:30].strip()
                    mag_str = line[52:57].strip()

                    if not ra_str or not dec_str or not mag_str:
                        continue

                    vmag = float(mag_str)
                    if vmag > self.mag_limit:
                        continue

                    # Parse RA "HH:MM:SS.SS"
                    ra_parts = ra_str.split(":")
                    ra_h = float(ra_parts[0])
                    ra_m = float(ra_parts[1])
                    ra_s = float(ra_parts[2])
                    ra = 15.0 * (ra_h + ra_m / 60.0 + ra_s / 3600.0)

                    # Parse DEC "+DD:MM:SS.SS"
                    dec_parts = dec_str.split(":")
                    dec_sign = -1 if dec_parts[0].startswith("-") else 1
                    dec_d = abs(float(dec_parts[0]))
                    dec_m = float(dec_parts[1])
                    dec_s = float(dec_parts[2])
                    dec = dec_sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)

                    hr_ids.append(hr)
                    ra_deg.append(ra)
                    dec_deg.append(dec)
                    vmags.append(vmag)
                except (ValueError, IndexError):
                    continue

        out = Table()
        out["hip_id"] = hr_ids  # HR numbers (close enough for matching)
        out["ra_deg"] = np.array(ra_deg, dtype=np.float64)
        out["dec_deg"] = np.array(dec_deg, dtype=np.float64)
        out["vmag"] = np.array(vmags, dtype=np.float64)

        # Cache as ECSV for fast future loads
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out.write(CATALOG_PATH, format="ascii.ecsv", overwrite=True)
        return out

    def _download_and_cache(self):
        """Download Hipparcos catalog from VizieR, filter, and cache."""
        try:
            from astroquery.vizier import Vizier

            vizier = Vizier(
                columns=["HIP", "Vmag", "_RAJ2000", "_DEJ2000"],
                row_limit=-1,
            )
            result = vizier.get_catalogs("I/239/hip_main")
            hip = result[0]
        except Exception as exc:
            raise RuntimeError(
                f"Cannot download Hipparcos catalog: {exc}. "
                f"Place a pre-built catalog at {CATALOG_PATH} "
                f"or BSC5 at {BSC5_PATH}"
            ) from exc

        mask = hip["Vmag"] <= self.mag_limit
        hip = hip[mask]

        out = Table()
        out["hip_id"] = hip["HIP"]
        out["ra_deg"] = np.array(hip["_RAJ2000"], dtype=np.float64)
        out["dec_deg"] = np.array(hip["_DEJ2000"], dtype=np.float64)
        out["vmag"] = np.array(hip["Vmag"], dtype=np.float64)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out.write(CATALOG_PATH, format="ascii.ecsv", overwrite=True)
        return out

    def get_visible_stars(self, lat_deg, lon_deg, obs_time, alt_limit=5.0,
                          response_k=0.20):
        """Return catalog stars visible from the given location and time.

        Parameters
        ----------
        lat_deg, lon_deg : float
            Observer latitude/longitude in degrees.
        obs_time : astropy.time.Time
            Observation time.
        alt_limit : float
            Minimum altitude in degrees.
        response_k : float
            Default effective response slope (mag/airmass) used to
            predict expected apparent magnitudes when no per-instrument
            response calibration is available.  Bundles nominal
            atmospheric extinction with zenith-angle-dependent
            throughput rolloff (vignetting, projection Jacobian).

        Returns
        -------
        astropy.table.Table
            Columns: hip_id, ra_deg, dec_deg, vmag, az_deg, alt_deg,
            airmass, vmag_expected
        """
        cat = self.table
        location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg,
                                 height=0 * u.m)
        frame = AltAz(obstime=obs_time, location=location)
        coords = SkyCoord(ra=cat["ra_deg"] * u.deg, dec=cat["dec_deg"] * u.deg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            altaz = coords.transform_to(frame)

        alt_deg_arr = altaz.alt.deg
        az_deg_arr = altaz.az.deg

        mask = alt_deg_arr >= alt_limit

        out = Table()
        out["hip_id"] = cat["hip_id"][mask]
        out["ra_deg"] = cat["ra_deg"][mask]
        out["dec_deg"] = cat["dec_deg"][mask]
        out["vmag"] = cat["vmag"][mask]
        out["az_deg"] = az_deg_arr[mask]
        out["alt_deg"] = alt_deg_arr[mask]
        out["airmass"] = airmass_bemporad(np.radians(alt_deg_arr[mask]))
        out["vmag_expected"] = expected_apparent_mag(
            out["vmag"], out["airmass"], k=response_k
        )

        # Sort by apparent brightness (brightest first)
        out.sort("vmag_expected")
        return out
