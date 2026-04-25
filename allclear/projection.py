"""Camera projection model for all-sky lenses."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ProjectionType(Enum):
    EQUIDISTANT = "equidistant"      # r = f * theta
    EQUISOLID = "equisolid"          # r = 2f * sin(theta/2)
    STEREOGRAPHIC = "stereographic"  # r = 2f * tan(theta/2)
    ORTHOGRAPHIC = "orthographic"    # r = f * sin(theta)


def _theta_to_r(theta, f, proj_type):
    """Map zenith angle theta to radial distance r for given projection."""
    if proj_type == ProjectionType.EQUIDISTANT:
        return f * theta
    elif proj_type == ProjectionType.EQUISOLID:
        return 2.0 * f * np.sin(theta / 2.0)
    elif proj_type == ProjectionType.STEREOGRAPHIC:
        return 2.0 * f * np.tan(theta / 2.0)
    elif proj_type == ProjectionType.ORTHOGRAPHIC:
        return f * np.sin(theta)
    raise ValueError(f"Unknown projection type: {proj_type}")


def _r_to_theta(r, f, proj_type):
    """Inverse: radial distance r back to zenith angle theta."""
    r_f = r / f
    if proj_type == ProjectionType.EQUIDISTANT:
        return r_f
    elif proj_type == ProjectionType.EQUISOLID:
        return 2.0 * np.arcsin(np.clip(r_f / 2.0, -1, 1))
    elif proj_type == ProjectionType.STEREOGRAPHIC:
        return 2.0 * np.arctan(r_f / 2.0)
    elif proj_type == ProjectionType.ORTHOGRAPHIC:
        return np.arcsin(np.clip(r_f, -1, 1))
    raise ValueError(f"Unknown projection type: {proj_type}")


def _apply_distortion(r, k1, k2):
    """Apply radial distortion: r' = r * (1 + k1*r^2 + k2*r^4)."""
    r2 = r * r
    return r * (1.0 + k1 * r2 + k2 * r2 * r2)


def _invert_distortion(r_dist, k1, k2, iterations=20):
    """Invert radial distortion via Newton's method."""
    r = r_dist.copy() if hasattr(r_dist, 'copy') else np.float64(r_dist)
    for _ in range(iterations):
        r2 = r * r
        r4 = r2 * r2
        f_val = r * (1.0 + k1 * r2 + k2 * r4) - r_dist
        f_prime = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4
        # Guard against zero derivative
        f_prime = np.where(np.abs(f_prime) < 1e-15, 1.0, f_prime)
        r = r - f_val / f_prime
    return r


@dataclass
class CameraModel:
    """All-sky camera geometric model.

    Parameters
    ----------
    cx, cy : float
        Optical center in pixel coordinates.
    az0 : float
        Azimuth of the camera boresight (radians). For a zenith-pointing
        camera this is the azimuth corresponding to the "up" direction
        in the image (the rotation of the camera).
    alt0 : float
        Altitude of boresight (radians). Typically ~pi/2 for zenith.
    rho : float
        Camera roll angle (radians). Rotation of the sensor about the
        optical axis.
    f : float
        Focal length in pixels (focal_mm / pixel_mm).
    proj_type : ProjectionType
        Lens projection model.
    k1 : float
        Radial distortion coefficient (3rd order).
    k2 : float
        Radial distortion coefficient (5th order).
    """

    cx: float = 1548.0
    cy: float = 1040.0
    az0: float = 0.0
    alt0: float = np.pi / 2
    rho: float = 0.0
    f: float = 750.0  # ~1.8mm / 0.0024mm
    proj_type: ProjectionType = ProjectionType.EQUIDISTANT
    k1: float = 0.0
    k2: float = 0.0

    # --- Parameter vector interface for optimizer ---

    @staticmethod
    def param_names():
        return ["cx", "cy", "az0", "alt0", "rho", "f", "k1", "k2"]

    def get_params_array(self):
        return np.array([self.cx, self.cy, self.az0, self.alt0,
                         self.rho, self.f, self.k1, self.k2])

    @classmethod
    def from_params_array(cls, params, proj_type=ProjectionType.EQUIDISTANT):
        return cls(
            cx=params[0], cy=params[1], az0=params[2], alt0=params[3],
            rho=params[4], f=params[5], proj_type=proj_type,
            k1=params[6], k2=params[7],
        )

    # --- Projection ---

    def sky_to_pixel(self, az, alt):
        """Project sky coordinates (az, alt in radians) to pixel (x, y).

        Parameters can be scalar or array. Returns (x, y) same shape as input.
        """
        az = np.asarray(az, dtype=np.float64)
        alt = np.asarray(alt, dtype=np.float64)

        # Direction vectors in ground frame (x=E, y=N, z=Up)
        cos_alt = np.cos(alt)
        dx = cos_alt * np.sin(az)
        dy = cos_alt * np.cos(az)
        dz = np.sin(alt)

        # Rotate into camera frame.
        # Camera boresight points at (az0, alt0), roll = rho.
        # Build rotation: first rotate about z by -az0, then about x by
        # -(pi/2 - alt0), then about z by -rho.
        d_cam = self._rotate_to_camera(dx, dy, dz)

        # Camera frame: z_cam = optical axis, x_cam/y_cam = sensor plane
        xc, yc, zc = d_cam

        # Zenith angle from optical axis
        theta = np.arctan2(np.sqrt(xc**2 + yc**2), zc)
        phi = np.arctan2(xc, yc)  # angle in sensor plane

        # Radial distance
        r = _theta_to_r(theta, self.f, self.proj_type)

        # Apply distortion
        r_dist = _apply_distortion(r, self.k1, self.k2)

        # Pixel coordinates
        x = self.cx + r_dist * np.sin(phi)
        y = self.cy + r_dist * np.cos(phi)

        return x, y

    def pixel_to_sky(self, x, y):
        """Inverse projection: pixel (x, y) to sky (az, alt) in radians.

        Returns (az, alt). Points behind the camera get NaN.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        dx_pix = x - self.cx
        dy_pix = y - self.cy

        r_dist = np.sqrt(dx_pix**2 + dy_pix**2)
        phi = np.arctan2(dx_pix, dy_pix)

        # Invert distortion
        r = _invert_distortion(r_dist, self.k1, self.k2)

        # Invert radial projection
        theta = _r_to_theta(r, self.f, self.proj_type)

        # Direction in camera frame
        sin_theta = np.sin(theta)
        xc = sin_theta * np.sin(phi)
        yc = sin_theta * np.cos(phi)
        zc = np.cos(theta)

        # Rotate back to ground frame
        dx, dy, dz = self._rotate_to_ground(xc, yc, zc)

        az = np.arctan2(dx, dy) % (2 * np.pi)
        alt = np.arcsin(np.clip(dz, -1, 1))

        return az, alt

    # --- WCS conversion ---

    # Mapping from AllClear projection types to FITS WCS codes (Paper III,
    # Calabretta & Greisen 2002).
    _PROJ_TO_WCS = {
        ProjectionType.EQUIDISTANT: "ARC",    # zenithal equidistant
        ProjectionType.EQUISOLID: "ZEA",      # zenithal equal-area
        ProjectionType.STEREOGRAPHIC: "STG",   # stereographic
        ProjectionType.ORTHOGRAPHIC: "SIN",    # orthographic / slant
    }
    _WCS_TO_PROJ = {v: k for k, v in _PROJ_TO_WCS.items()}

    def to_wcs(self, obs_time, site_lat, site_lon, naxis=None):
        """Convert CameraModel to an astropy WCS with SIP distortion.

        Parameters
        ----------
        obs_time : astropy.time.Time or str
            Observation UTC (for az/alt → RA/Dec conversion).
        site_lat, site_lon : float
            Observer latitude / longitude in **degrees**.
        naxis : tuple of int, optional
            (NAXIS1, NAXIS2) image dimensions.

        Returns
        -------
        wcs : astropy.wcs.WCS
        """
        from astropy.wcs import WCS
        from astropy.time import Time
        from astropy.coordinates import EarthLocation, AltAz, SkyCoord
        import astropy.units as u

        if isinstance(obs_time, str):
            obs_time = Time(obs_time)

        proj_code = self._PROJ_TO_WCS[self.proj_type]

        # Observer location / frame
        location = EarthLocation.from_geodetic(
            lon=site_lon * u.deg, lat=site_lat * u.deg
        )
        altaz_frame = AltAz(obstime=obs_time, location=location)

        # ---- helpers -------------------------------------------------------
        def _altaz_to_radec(az_rad, alt_rad):
            """(az, alt) in radians → (RA, Dec) in degrees."""
            sc = SkyCoord(
                az=np.degrees(az_rad) * u.deg,
                alt=np.degrees(alt_rad) * u.deg,
                frame=altaz_frame,
            ).icrs
            return sc.ra.deg, sc.dec.deg

        # ---- CRVAL: boresight RA/Dec --------------------------------------
        # NOTE: self.az0 is a rotation-matrix parameter, NOT the boresight
        # azimuth direction.  Compute the true boresight from the rotation.
        az_bore, alt_bore = self.pixel_to_sky(self.cx, self.cy)
        ra0, dec0 = _altaz_to_radec(float(az_bore), float(alt_bore))

        # ---- CRPIX (FITS 1-indexed) ---------------------------------------
        crpix1 = self.cx + 1.0
        crpix2 = self.cy + 1.0

        # ---- CD matrix via numerical Jacobian ------------------------------
        # Build an undistorted model (SIP handles distortion separately).
        m0 = CameraModel(
            cx=self.cx, cy=self.cy, az0=self.az0, alt0=self.alt0,
            rho=self.rho, f=self.f, proj_type=self.proj_type, k1=0.0, k2=0.0,
        )

        # Temporary WCS with identity CD — used only to map (RA,Dec) → IWC.
        hdr = {
            "NAXIS": 2,
            "CTYPE1": f"RA---{proj_code}",
            "CTYPE2": f"DEC--{proj_code}",
            "CRPIX1": crpix1,
            "CRPIX2": crpix2,
            "CRVAL1": float(ra0),
            "CRVAL2": float(dec0),
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CD1_1": 1.0,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 1.0,
        }
        w_unit = WCS(hdr)

        delta = 50.0  # pixel perturbation for finite differences
        perturbs = [(delta, 0), (-delta, 0), (0, delta), (0, -delta)]
        iwc_pts = []
        for du, dv in perturbs:
            az_s, alt_s = m0.pixel_to_sky(self.cx + du, self.cy + dv)
            ra_s, dec_s = _altaz_to_radec(az_s, alt_s)
            # With identity CD, pixel = IWC + CRPIX (1-indexed), so
            # IWC = pixel_from_wcs − (CRPIX − 1)  in 0-indexed output.
            pix = w_unit.wcs_world2pix(np.array([[ra_s, dec_s]]), 0)[0]
            iwc_pts.append(pix - np.array([self.cx, self.cy]))

        # Centered differences: dIWC / d(pixel offset)
        diwc_dx = (iwc_pts[0] - iwc_pts[1]) / (2.0 * delta)
        diwc_dy = (iwc_pts[2] - iwc_pts[3]) / (2.0 * delta)

        cd = np.array([[diwc_dx[0], diwc_dy[0]],
                        [diwc_dx[1], diwc_dy[1]]])

        # ---- SIP distortion polynomials ------------------------------------
        sip_order = 7
        sip_a, sip_b = self._fit_sip(sip_order)

        # Inverse SIP (AP, BP) for world → pixel direction
        sip_ap, sip_bp = self._fit_sip_inverse(sip_order)

        # ---- Assemble final header -----------------------------------------
        header = {
            "NAXIS": 2,
            "CTYPE1": f"RA---{proj_code}-SIP",
            "CTYPE2": f"DEC--{proj_code}-SIP",
            "CRPIX1": crpix1,
            "CRPIX2": crpix2,
            "CRVAL1": float(ra0),
            "CRVAL2": float(dec0),
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "EQUINOX": 2000.0,
            "RADESYS": "ICRS",
            "CD1_1": cd[0, 0],
            "CD1_2": cd[0, 1],
            "CD2_1": cd[1, 0],
            "CD2_2": cd[1, 1],
            "A_ORDER": sip_order,
            "B_ORDER": sip_order,
            "AP_ORDER": sip_order,
            "BP_ORDER": sip_order,
        }
        if naxis is not None:
            header["NAXIS1"] = int(naxis[0])
            header["NAXIS2"] = int(naxis[1])

        # Write SIP coefficient keywords
        for (p, q), val in sip_a.items():
            header[f"A_{p}_{q}"] = val
        for (p, q), val in sip_b.items():
            header[f"B_{p}_{q}"] = val
        for (p, q), val in sip_ap.items():
            header[f"AP_{p}_{q}"] = val
        for (p, q), val in sip_bp.items():
            header[f"BP_{p}_{q}"] = val

        return WCS(header)

    # ---- SIP fitting helpers -----------------------------------------------

    @staticmethod
    def _radial_sip_coeffs(c_radial, max_order):
        """Expand radial coefficients into SIP A and B dictionaries.

        The fit produces coefficients for the basis functions
        ``u * r^{2(n+1)}`` (n = 0, 1, 2, ...).  Expanding via the
        binomial theorem::

            u * (u² + v²)^{n+1} = Σ_k C(n+1,k) u^{2k+1} v^{2(n+1-k)}

        gives the SIP polynomial terms with constrained coefficients
        that enforce purely radial distortion.
        """
        from math import comb
        sip_a, sip_b = {}, {}
        for n, c_n in enumerate(c_radial):
            power = n + 1                # exponent of r²
            sip_ord = 2 * power + 1      # SIP order = 3, 5, 7, ...
            if sip_ord > max_order:
                break
            if abs(c_n) < 1e-25:
                continue
            for k in range(power + 1):
                bk = comb(power, k)
                # A terms: u^(2k+1) * v^(2*(power-k))
                p_a = 2 * k + 1
                q_a = 2 * (power - k)
                sip_a[(p_a, q_a)] = sip_a.get((p_a, q_a), 0.0) + c_n * bk
                # B terms: u^(2k) * v^(2*(power-k)+1)
                p_b = 2 * k
                q_b = 2 * (power - k) + 1
                sip_b[(p_b, q_b)] = sip_b.get((p_b, q_b), 0.0) + c_n * bk
        sip_a = {k: v for k, v in sip_a.items() if abs(v) > 1e-25}
        sip_b = {k: v for k, v in sip_b.items() if abs(v) > 1e-25}
        return sip_a, sip_b

    def _fit_radial_correction(self, n_terms):
        """Fit the 1D radial correction function for SIP.

        The CameraModel distortion is purely radial, so the SIP
        correction has the form:

            du = u * g(r²),  dv = v * g(r²)

        where g(r²) = c₀r² + c₁r⁴ + c₂r⁶ + ...

        We fit g(r²) as a 1D polynomial, avoiding ill-conditioned 2D fits.

        Returns (c_forward, c_inverse) — radial coefficient arrays for
        forward (distorted→undistorted) and inverse (undistorted→distorted)
        SIP directions.
        """
        # Sample radii from near-zero to the max usable radius.  We
        # sample in the *undistorted* space because _apply_distortion
        # is monotonic and well-defined there, then pair each point
        # (r_undist, r_dist) to get both the forward and inverse
        # mappings without relying on Newton inversion, which is
        # unstable for cameras whose distortion approaches its own
        # critical radius (e.g. Haleakala's k1*r_max^2 ~ 0.15).
        r_max = self.f * np.pi / 2 * 0.92  # ~92% of horizon
        r_undist = np.linspace(1.0, r_max, 400)
        r_dist = _apply_distortion(r_undist, self.k1, self.k2)

        # Drop points where the distortion polynomial is non-monotonic
        # (past its critical radius, r_dist stops increasing with
        # r_undist); those samples are outside the usable field.
        ok = np.concatenate(([True], np.diff(r_dist) > 0))
        r_undist = r_undist[ok]
        r_dist = r_dist[ok]

        # Forward SIP (distorted -> undistorted): evaluate at r_dist.
        g_fwd = r_undist / r_dist - 1.0
        # Inverse SIP (undistorted -> distorted): evaluate at r_undist.
        g_inv = r_dist / r_undist - 1.0
        # Below, we fit each polynomial in its own sample grid.
        # Use r_dist for the forward fit and r_undist for the inverse.
        radii = r_dist

        # Fit each polynomial in its own natural variable.  Work in
        # normalized t = (r/r_max)^2 to avoid 25-orders-of-magnitude
        # conditioning issues with raw pixel units.
        t_fwd = (r_dist / r_max) ** 2
        t_inv = (r_undist / r_max) ** 2
        M_fwd = np.column_stack([t_fwd ** (n + 1) for n in range(n_terms)])
        M_inv = np.column_stack([t_inv ** (n + 1) for n in range(n_terms)])

        d_fwd, _, _, _ = np.linalg.lstsq(M_fwd, g_fwd, rcond=None)
        d_inv, _, _, _ = np.linalg.lstsq(M_inv, g_inv, rcond=None)

        # Convert back: d_n = c_n * r_max^{2(n+1)}
        c_fwd = np.array([d_fwd[n] / r_max ** (2 * (n + 1))
                          for n in range(n_terms)])
        c_inv = np.array([d_inv[n] / r_max ** (2 * (n + 1))
                          for n in range(n_terms)])

        return c_fwd, c_inv

    def _fit_sip(self, order):
        """Fit forward SIP (A, B): distorted pixel → undistorted pixel."""
        n_terms = (order - 1) // 2
        c_fwd, _ = self._fit_radial_correction(n_terms)
        return self._radial_sip_coeffs(c_fwd, order)

    def _fit_sip_inverse(self, order):
        """Fit inverse SIP (AP, BP): undistorted pixel → distorted pixel."""
        n_terms = (order - 1) // 2
        _, c_inv = self._fit_radial_correction(n_terms)
        return self._radial_sip_coeffs(c_inv, order)

    @classmethod
    def from_wcs(cls, wcs, obs_time, site_lat, site_lon):
        """Reconstruct a CameraModel from a WCS + observation context.

        Parameters
        ----------
        wcs : astropy.wcs.WCS
            WCS with a zenithal projection (ARC/ZEA/STG/SIN), optional SIP.
        obs_time : astropy.time.Time or str
        site_lat, site_lon : float   (degrees)

        Returns
        -------
        CameraModel
        """
        from astropy.time import Time
        from astropy.coordinates import EarthLocation, AltAz, SkyCoord
        import astropy.units as u

        if isinstance(obs_time, str):
            obs_time = Time(obs_time)

        # ---- projection type -----------------------------------------------
        ctype1 = wcs.wcs.ctype[0]
        # Strip SIP suffix and extract 3-letter code
        proj_code = ctype1.split("-")[-1].replace("SIP", "")
        if not proj_code or proj_code not in cls._WCS_TO_PROJ:
            # Try second-to-last token
            parts = [p for p in ctype1.split("-") if p and p != "SIP"]
            proj_code = parts[-1] if parts else "ARC"
        proj_type = cls._WCS_TO_PROJ.get(proj_code, ProjectionType.EQUIDISTANT)

        # ---- optical center ------------------------------------------------
        cx = wcs.wcs.crpix[0] - 1.0   # FITS 1-indexed → 0-indexed
        cy = wcs.wcs.crpix[1] - 1.0

        # ---- focal length from CD matrix -----------------------------------
        cd = wcs.wcs.cd if hasattr(wcs.wcs, "cd") and wcs.wcs.cd.size else np.eye(2)
        # Pixel scale (deg/pix) = sqrt(|det(CD)|)
        pixel_scale_deg = np.sqrt(abs(np.linalg.det(cd)))
        f = np.degrees(1.0) / pixel_scale_deg  # (180/pi) / scale

        # ---- boresight az/alt ----------------------------------------------
        ra0, dec0 = wcs.wcs.crval
        location = EarthLocation.from_geodetic(
            lon=site_lon * u.deg, lat=site_lat * u.deg
        )
        altaz_frame = AltAz(obstime=obs_time, location=location)
        boresight = SkyCoord(ra=ra0 * u.deg, dec=dec0 * u.deg,
                             frame="icrs").transform_to(altaz_frame)
        az0 = np.radians(boresight.az.deg)
        alt0 = np.radians(boresight.alt.deg)

        # ---- roll angle (rho) ----------------------------------------------
        # The CD matrix encodes scale + rotation + possible parity flip.
        # Total rotation angle of the WCS = angle of pixel axes relative to
        # the celestial axes.  This combines the camera roll (rho) with the
        # parallactic angle at observation time.
        #
        # WCS rotation angle (from CD matrix):
        wcs_angle = np.arctan2(-cd[0, 1], cd[1, 1])
        # For the parallactic angle, use small offsets.
        # North direction in pixel frame: offset Dec by +1 arcmin
        north_sc = SkyCoord(ra=ra0 * u.deg, dec=(dec0 + 1.0 / 60) * u.deg,
                            frame="icrs").transform_to(altaz_frame)
        az_n, alt_n = np.radians(north_sc.az.deg), np.radians(north_sc.alt.deg)
        # Build a zero-distortion model with rho=0 to find the pixel PA of north
        m_trial = cls(cx=cx, cy=cy, az0=az0, alt0=alt0, rho=0.0,
                      f=f, proj_type=proj_type, k1=0.0, k2=0.0)
        px0, py0 = m_trial.sky_to_pixel(az0, alt0)
        px_n, py_n = m_trial.sky_to_pixel(az_n, alt_n)
        pa_north_rho0 = np.arctan2(px_n - px0, py_n - py0)  # PA in pixel frame with rho=0
        # The WCS angle tells us what PA north *should* have in the image.
        # rho is the difference.
        rho = (wcs_angle - pa_north_rho0) % (2 * np.pi)

        # ---- distortion (k1, k2 from SIP) ---------------------------------
        k1, k2 = 0.0, 0.0
        if wcs.sip is not None:
            # Forward SIP A maps distorted→undistorted. Our convention:
            # distorted = undistorted * (1 + k1*r² + k2*r⁴)
            # SIP A_3_0 ≈ -k1, so k1 ≈ -A_3_0
            a = wcs.sip.a
            if a is not None and a.shape[0] > 3:
                k1 = -a[3, 0] if abs(a[3, 0]) > 1e-20 else 0.0
            if a is not None and a.shape[0] > 5:
                k2 = -a[5, 0] if abs(a[5, 0]) > 1e-20 else 0.0

        return cls(
            cx=cx, cy=cy, az0=az0, alt0=alt0, rho=rho,
            f=f, proj_type=proj_type, k1=k1, k2=k2,
        )

    def _rotate_to_camera(self, dx, dy, dz):
        """Rotate ground-frame direction to camera frame."""
        # Step 1: rotate about z-axis by -az0
        ca, sa = np.cos(-self.az0), np.sin(-self.az0)
        x1 = ca * dx - sa * dy
        y1 = sa * dx + ca * dy
        z1 = dz

        # Step 2: rotate about x-axis by -(pi/2 - alt0) to tilt boresight to z
        tilt = -(np.pi / 2 - self.alt0)
        ct, st = np.cos(tilt), np.sin(tilt)
        x2 = x1
        y2 = ct * y1 - st * z1
        z2 = st * y1 + ct * z1

        # Step 3: rotate about z-axis by -rho (camera roll)
        cr, sr = np.cos(-self.rho), np.sin(-self.rho)
        xc = cr * x2 - sr * y2
        yc = sr * x2 + cr * y2
        zc = z2

        return xc, yc, zc

    def _rotate_to_ground(self, xc, yc, zc):
        """Rotate camera-frame direction back to ground frame."""
        # Inverse of _rotate_to_camera: apply inverse rotations in reverse order

        # Step 3 inverse: rotate about z by +rho
        cr, sr = np.cos(self.rho), np.sin(self.rho)
        x2 = cr * xc - sr * yc
        y2 = sr * xc + cr * yc
        z2 = zc

        # Step 2 inverse: rotate about x by +(pi/2 - alt0)
        tilt = np.pi / 2 - self.alt0
        ct, st = np.cos(tilt), np.sin(tilt)
        x1 = x2
        y1 = ct * y2 - st * z2
        z1 = st * y2 + ct * z2

        # Step 1 inverse: rotate about z by +az0
        ca, sa = np.cos(self.az0), np.sin(self.az0)
        dx = ca * x1 - sa * y1
        dy = sa * x1 + ca * y1
        dz = z1

        return dx, dy, dz
