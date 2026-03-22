"""Web-based interactive viewer for the allclear manual-fit tool.

Replaces the matplotlib GUI with a browser-based interface using a local
HTTP server and Leaflet.js for smooth zoom/pan of all-sky images.

Usage (from cli.py cmd_manual_fit):
    from .manual_fit_web import ManualFitWeb
    viewer = ManualFitWeb(data, objects, cat, meta, output_path,
                          initial_model=initial_model, mirrored=mirrored)
    viewer.run()
"""

import io
import json
import logging
import socket
import webbrowser
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np

from .projection import CameraModel, ProjectionType
from .manual_fit import solve_from_clicks

logger = logging.getLogger(__name__)


def _render_image_png(image, mirrored=False):
    """Render a 2D numpy array to PNG bytes with percentile stretch.

    Parameters
    ----------
    image : ndarray
        2D image array (float or int).
    mirrored : bool
        If True, flip left-right before rendering.

    Returns
    -------
    bytes
        PNG image data.
    """
    from PIL import Image

    data = image.copy()
    if mirrored:
        data = data[:, ::-1]

    # Percentile stretch
    vmin, vmax = np.percentile(data, [1, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = np.clip((data - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    # Flip vertically: FITS has origin="lower" but PNG has origin at top.
    # We will handle this in Leaflet by setting the image bounds so that
    # y=0 (bottom row of FITS) maps to Leaflet y=0. Leaflet's CRS.Simple
    # has y increasing upward, so we need the PNG to be stored in the
    # normal top-down order and tell Leaflet the bounds go from [0,0] to
    # [ny, nx].  Since PNG row 0 = top = FITS row ny-1, we flip here.
    scaled = scaled[::-1]

    img = Image.fromarray(scaled, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def _find_free_port(start=8765, end=8865):
    """Find a free TCP port in the given range."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


class ManualFitWeb:
    """Web-based interactive manual-fit viewer.

    Parameters
    ----------
    image : ndarray
        2D image array.
    objects : list of dict
        From get_identifiable_objects(). Each has name, az_deg, alt_deg, vmag, category.
    cat_table : astropy.table.Table
        Full star catalog.
    meta : dict
        Frame metadata (obs_time, lat_deg, lon_deg, etc.).
    output_path : str
        Default output path for saving the model JSON.
    initial_model : CameraModel, optional
        If provided, start with this model.
    mirrored : bool
        If True, the image should be flipped left-right for display.
    """

    def __init__(self, image, objects, cat_table, meta, output_path,
                 initial_model=None, mirrored=False):
        self.image = image
        self.objects = objects
        self.cat_table = cat_table
        self.meta = meta
        self.output_path = output_path
        self.model = initial_model
        self.mirrored = mirrored

        # Corrections: list of dicts {star_id, az_rad, alt_rad, actual_x, actual_y}
        self.corrections = []
        self.n_guided = 0
        self.guided_rms = 0.0

        # Pre-render image
        ny, nx = image.shape
        self.ny = ny
        self.nx = nx
        self._png_bytes = _render_image_png(image, mirrored=mirrored)

    def _get_stars_json(self):
        """Build JSON response with all catalog star predictions and named objects."""
        result = {
            "model_params": None,
            "catalog_stars": [],
            "named_objects": [],
            "corrections": [],
            "n_guided": self.n_guided,
            "guided_rms": self.guided_rms,
            "image_width": self.nx,
            "image_height": self.ny,
        }

        if self.model is not None:
            result["model_params"] = {
                "cx": self.model.cx, "cy": self.model.cy,
                "az0_deg": float(np.degrees(self.model.az0)),
                "alt0_deg": float(np.degrees(self.model.alt0)),
                "rho_deg": float(np.degrees(self.model.rho)),
                "f": self.model.f,
                "k1": self.model.k1, "k2": self.model.k2,
                "proj_type": self.model.proj_type.value,
            }

            # Project catalog stars
            cat_az = np.radians(
                np.asarray(self.cat_table["az_deg"], dtype=np.float64))
            cat_alt = np.radians(
                np.asarray(self.cat_table["alt_deg"], dtype=np.float64))
            cat_x, cat_y = self.model.sky_to_pixel(cat_az, cat_alt)
            vmag = np.asarray(self.cat_table["vmag_extinct"], dtype=np.float64)

            # Sort by brightness and limit to 500
            order = np.argsort(vmag)
            n_shown = 0
            for i in order:
                if n_shown >= 500:
                    break
                cx, cy = float(cat_x[i]), float(cat_y[i])
                if not (np.isfinite(cx) and np.isfinite(cy)
                        and 0 <= cx < self.nx and 0 <= cy < self.ny):
                    continue
                result["catalog_stars"].append({
                    "idx": int(i),
                    "x": cx, "y": cy,
                    "az_deg": float(self.cat_table["az_deg"][i]),
                    "alt_deg": float(self.cat_table["alt_deg"][i]),
                    "vmag": float(vmag[i]),
                })
                n_shown += 1

            # Project named objects
            for idx, obj in enumerate(self.objects):
                az_rad = np.radians(obj["az_deg"])
                alt_rad = np.radians(obj["alt_deg"])
                px, py = self.model.sky_to_pixel(
                    np.array([az_rad]), np.array([alt_rad]))
                x, y = float(px[0]), float(py[0])
                if not (np.isfinite(x) and np.isfinite(y)
                        and -50 < x < self.nx + 50
                        and -50 < y < self.ny + 50):
                    continue
                result["named_objects"].append({
                    "idx": idx,
                    "name": obj["name"],
                    "x": x, "y": y,
                    "az_deg": obj["az_deg"],
                    "alt_deg": obj["alt_deg"],
                    "vmag": obj["vmag"],
                    "category": obj["category"],
                })

        # Include current corrections with UPDATED predicted positions
        # from the current model (so arrows track model changes).
        if self.model is not None:
            for c in self.corrections:
                az_r = np.radians(c["az_deg"])
                alt_r = np.radians(c["alt_deg"])
                px_c, py_c = self.model.sky_to_pixel(
                    np.array([az_r]), np.array([alt_r]))
                cc = dict(c)
                cc["pred_x"] = float(px_c[0])
                cc["pred_y"] = float(py_c[0])
                result["corrections"].append(cc)
        else:
            for c in self.corrections:
                result["corrections"].append(c)

        return result

    def _handle_correct(self, data):
        """Handle a correction from the user."""
        self.corrections.append(data)
        n = len(self.corrections)
        print(f"  Correction #{n}: {data.get('label', '?')} -> "
              f"({data['actual_x']:.0f}, {data['actual_y']:.0f})")

        # Auto-solve when we have 3+ corrections
        if n >= 3:
            return self._auto_solve()
        return {"status": "ok", "n_corrections": n}

    def _auto_solve(self):
        """Re-solve model from correction clicks."""
        click_az = [np.radians(c["az_deg"]) for c in self.corrections]
        click_alt = [np.radians(c["alt_deg"]) for c in self.corrections]
        click_px = [c["actual_x"] for c in self.corrections]
        click_py = [c["actual_y"] for c in self.corrections]

        try:
            self.model = solve_from_clicks(
                click_px, click_py, click_az, click_alt,
                self.image.shape)
            self.n_guided = 0
            self.guided_rms = 0.0

            mx, my = self.model.sky_to_pixel(
                np.array(click_az), np.array(click_alt))
            rms = float(np.sqrt(np.mean(
                (mx - np.array(click_px))**2 +
                (my - np.array(click_py))**2)))
            print(f"  Solved: f={self.model.f:.1f}, "
                  f"cx={self.model.cx:.0f}, cy={self.model.cy:.0f}, "
                  f"alt0={np.degrees(self.model.alt0):.1f}°, "
                  f"rho={np.degrees(self.model.rho):.1f}°, "
                  f"RMS={rms:.1f} px")

            # Auto-refine with distortion when we have enough corrections
            # spread across the sky (coverage check)
            n = len(self.corrections)
            if n >= 8 and self._has_sky_coverage():
                print(f"  Auto-refining with distortion "
                      f"({n} corrections with sky coverage)...")
                result = self._handle_refine({"fit_distortion": True})
                if result.get("status") == "refined":
                    rms = result.get("rms", rms)

            stars = self._get_stars_json()
            return {"status": "solved", "rms": rms, "stars": stars}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  Solve failed: {e}")
            return {"status": "error", "message": str(e)}

    def _has_sky_coverage(self):
        """Check if corrections span at least 3 quadrants of the sky."""
        if len(self.corrections) < 6:
            return False
        azimuths = [c["az_deg"] for c in self.corrections]
        quadrants = set()
        for az in azimuths:
            quadrants.add(int(az / 90) % 4)
        return len(quadrants) >= 3

    def _handle_refine(self, data):
        """Run guided_refine."""
        if self.model is None:
            return {"status": "error",
                    "message": "No model to refine. Identify at least 3 objects first."}

        fit_distortion = data.get("fit_distortion", False)
        label = "+distortion" if fit_distortion else "geometry"
        print(f"  Running guided refine ({label})...")

        from .strategies import guided_refine, _adaptive_min_peak_offset

        cat_az = np.radians(
            np.asarray(self.cat_table["az_deg"], dtype=np.float64))
        cat_alt = np.radians(
            np.asarray(self.cat_table["alt_deg"], dtype=np.float64))

        background = float(np.median(self.image))
        p999 = float(np.percentile(self.image, 99.9))
        min_peak_offset = _adaptive_min_peak_offset(background, p999)

        # Use the possibly-mirrored image for guided refine
        img = self.image
        if self.mirrored:
            img = img[:, ::-1]

        refined, n_matched, rms = guided_refine(
            img, cat_az, cat_alt, self.model,
            n_iterations=15,
            min_peak_offset=min_peak_offset,
            fit_distortion=fit_distortion,
            initial_search_radius=30,
        )

        if n_matched >= 6:
            self.model = refined
            self.n_guided = n_matched
            self.guided_rms = rms
            print(f"  Guided refine: {n_matched} matches, RMS={rms:.2f} px")
            print(f"    f={refined.f:.1f}, "
                  f"rho={np.degrees(refined.rho):.1f} deg, "
                  f"k1={refined.k1:.2e}")
            stars = self._get_stars_json()
            return {"status": "ok", "n_matched": n_matched,
                    "rms": rms, "stars": stars}
        else:
            msg = f"Guided refine failed: only {n_matched} matches"
            print(f"  {msg}")
            return {"status": "error", "message": msg}

    def _handle_mirror(self):
        """Flip the model East-West."""
        if self.model is None:
            return {"status": "error", "message": "No model to mirror."}

        self.model = CameraModel(
            cx=self.model.cx, cy=self.model.cy,
            az0=-self.model.az0, alt0=self.model.alt0,
            rho=-self.model.rho, f=self.model.f,
            proj_type=self.model.proj_type,
            k1=self.model.k1, k2=self.model.k2,
        )
        # Clear corrections since they are now invalid
        self.corrections = []
        self.n_guided = 0
        self.guided_rms = 0.0

        print(f"  Mirrored! rho={np.degrees(self.model.rho):.1f} deg, "
              f"az0={np.degrees(self.model.az0):.1f} deg")

        stars = self._get_stars_json()
        return {"status": "ok", "stars": stars}

    def _handle_undo(self):
        """Undo the last correction."""
        if not self.corrections:
            return {"status": "error", "message": "Nothing to undo."}
        removed = self.corrections.pop()
        print(f"  Undone last correction")

        if len(self.corrections) >= 3:
            result = self._auto_solve()
            return result
        else:
            stars = self._get_stars_json()
            return {"status": "ok", "stars": stars}

    def _handle_save(self):
        """Save the current model as JSON."""
        if self.model is None:
            return {"status": "error", "message": "No model to save."}

        from .instrument import InstrumentModel
        from datetime import datetime, timezone
        import pathlib

        ny, nx = self.image.shape
        inst = InstrumentModel.from_camera_model(
            self.model,
            site_lat=self.meta.get("lat_deg", 0.0),
            site_lon=self.meta.get("lon_deg", 0.0),
            image_width=nx,
            image_height=ny,
            mirrored=self.mirrored,
            n_stars_matched=self.n_guided or len(self.corrections),
            rms_residual_px=self.guided_rms,
            fit_timestamp=datetime.now(timezone.utc).isoformat(),
            frame_used="manual-fit-web",
        )

        pathlib.Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        inst.save(self.output_path)
        print(f"  Model saved to {self.output_path}")
        return {"status": "ok", "path": self.output_path}

    def run(self):
        """Start the web server and open the browser."""
        port = _find_free_port()
        handler_class = partial(_RequestHandler, viewer=self)
        server = HTTPServer(('127.0.0.1', port), handler_class)
        url = f"http://127.0.0.1:{port}"

        print(f"\n  Manual Fit Web Viewer running at: {url}")
        print("  Press Ctrl+C to stop.\n")

        # Open browser
        webbrowser.open(url)

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")
        finally:
            server.server_close()


class _RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the manual-fit web viewer."""

    def __init__(self, *args, viewer=None, **kwargs):
        self.viewer = viewer
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Suppress default request logging."""
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_png(self, png_bytes):
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', len(png_bytes))
        self.send_header('Cache-Control', 'max-age=3600')
        self.end_headers()
        self.wfile.write(png_bytes)

    def _send_html(self, html):
        body = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._send_html(HTML_PAGE)
        elif self.path == '/api/image':
            self._send_png(self.viewer._png_bytes)
        elif self.path == '/api/stars':
            self._send_json(self.viewer._get_stars_json())
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        if self.path == '/api/correct':
            result = self.viewer._handle_correct(data)
            self._send_json(result)
        elif self.path == '/api/refine':
            result = self.viewer._handle_refine(data)
            self._send_json(result)
        elif self.path == '/api/mirror':
            result = self.viewer._handle_mirror()
            self._send_json(result)
        elif self.path == '/api/undo':
            result = self.viewer._handle_undo()
            self._send_json(result)
        elif self.path == '/api/save':
            result = self.viewer._handle_save()
            self._send_json(result)
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


# ---------------------------------------------------------------------------
# Embedded HTML/CSS/JS
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AllClear Manual Fit</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    overflow: hidden;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

#toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: #16213e;
    border-bottom: 1px solid #333;
    flex-shrink: 0;
    z-index: 1000;
}
#toolbar button {
    background: #0f3460;
    color: #e0e0e0;
    border: 1px solid #444;
    padding: 5px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    white-space: nowrap;
}
#toolbar button:hover { background: #1a5276; }
#toolbar button:active { background: #2471a3; }
#toolbar button.danger { background: #7b2d26; }
#toolbar button.danger:hover { background: #922b21; }
#toolbar button.success { background: #1e6e3e; }
#toolbar button.success:hover { background: #27864e; }

#status-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    flex: 1;
    font-size: 13px;
    padding-left: 8px;
}
#status-bar .stat {
    color: #aaa;
}
#status-bar .stat b {
    color: #fff;
}

#map-container {
    flex: 1;
    position: relative;
}
#map {
    width: 100%;
    height: 100%;
    background: #111;
}

#prompt-bar {
    padding: 6px 12px;
    background: #16213e;
    border-top: 1px solid #333;
    font-size: 13px;
    color: #ffd700;
    text-align: center;
    flex-shrink: 0;
    z-index: 1000;
    min-height: 30px;
}
#prompt-bar.active { color: #00ffff; background: #1a2a3e; }

#model-info {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(0,0,0,0.8);
    color: #ccc;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-family: 'Consolas', 'Monaco', monospace;
    z-index: 1000;
    pointer-events: none;
    line-height: 1.5;
    border: 1px solid #444;
}

/* Leaflet dark theme overrides */
.leaflet-container { background: #111 !important; }
.leaflet-control-zoom a {
    background: #222 !important;
    color: #ccc !important;
    border-color: #444 !important;
}
.leaflet-control-zoom a:hover { background: #333 !important; }

/* Custom marker labels */
.star-label {
    color: #ffcc44;
    font-size: 11px;
    font-weight: bold;
    text-shadow: 1px 1px 2px #000, -1px -1px 2px #000;
    white-space: nowrap;
    pointer-events: none !important;
}

/* Spinner overlay */
#spinner {
    display: none;
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.5);
    z-index: 2000;
    align-items: center;
    justify-content: center;
}
#spinner.active { display: flex; }
#spinner .inner {
    color: #fff;
    font-size: 16px;
    padding: 16px 24px;
    background: rgba(0,0,0,0.8);
    border-radius: 8px;
    border: 1px solid #555;
}
@keyframes spin { to { transform: rotate(360deg); } }
#spinner .inner::before {
    content: '';
    display: inline-block;
    width: 16px; height: 16px;
    border: 2px solid #fff;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 8px;
    vertical-align: middle;
}
</style>
</head>
<body>

<div id="toolbar">
    <button onclick="doRefine(false)" title="Run guided refine (geometry only)">Refine</button>
    <button onclick="doRefine(true)" title="Run guided refine with distortion fitting">Refine+Dist</button>
    <button onclick="doMirror()" class="danger" title="Flip East-West">Mirror</button>
    <button onclick="doUndo()" title="Undo last correction">Undo</button>
    <button onclick="doSave()" class="success" title="Save model JSON">Save</button>
    <div id="status-bar">
        <span class="stat">Corrections: <b id="n-corrections">0</b></span>
        <span class="stat">Guided: <b id="n-guided">0</b></span>
        <span class="stat">RMS: <b id="rms-val">--</b></span>
    </div>
</div>

<div id="map-container">
    <div id="map"></div>
    <div id="model-info"></div>
    <div id="spinner"><div class="inner">Working...</div></div>
</div>

<div id="prompt-bar">
    Loading image... Zoom with scroll wheel. Click any marker to select, then click actual position.
</div>

<script>
// -- State --
let selectedStar = null;    // {type: 'named'|'catalog', idx, az_deg, alt_deg, label, marker}
let starData = null;        // last /api/stars response
let catMarkers = [];        // L.circleMarker array
let namedMarkers = [];      // L.marker array
let correctionLines = [];   // L.polyline array
let correctionMarkers = []; // L.circleMarker array for actual positions
let highlightMarker = null; // highlighted selected marker

let imgW = 0, imgH = 0;
let map, imageLayer;

// -- Initialize map --
function initMap(width, height) {
    imgW = width;
    imgH = height;

    // Leaflet CRS.Simple: x = lng, y = lat.  We use pixel coordinates
    // with y increasing upward (FITS convention). Map bounds: [0,0] to [ny, nx].
    map = L.map('map', {
        crs: L.CRS.Simple,
        minZoom: -3,
        maxZoom: 5,
        zoomSnap: 0.25,
        zoomDelta: 0.5,
        wheelPxPerZoomLevel: 120,
        attributionControl: false,
    });

    // Image bounds: [[y_min, x_min], [y_max, x_max]] in Leaflet's [lat, lng]
    var bounds = [[0, 0], [imgH, imgW]];
    imageLayer = L.imageOverlay('/api/image', bounds).addTo(map);
    map.fitBounds(bounds);

    // Click handler for placing corrections
    map.on('click', onMapClick);
}

// -- Fetch star data and draw overlays --
function loadStars() {
    fetch('/api/stars')
        .then(r => r.json())
        .then(data => {
            starData = data;
            imgW = data.image_width;
            imgH = data.image_height;

            if (!map) {
                initMap(imgW, imgH);
            }

            drawOverlays(data);
            updateStats(data);
            updateModelInfo(data.model_params);
            setPrompt('Zoom with scroll wheel. Click any marker to select, then click actual position.');
        });
}

function clearOverlays() {
    catMarkers.forEach(m => map.removeLayer(m));
    catMarkers = [];
    namedMarkers.forEach(m => map.removeLayer(m));
    namedMarkers = [];
    correctionLines.forEach(m => map.removeLayer(m));
    correctionLines = [];
    correctionMarkers.forEach(m => map.removeLayer(m));
    correctionMarkers = [];
    if (highlightMarker) { map.removeLayer(highlightMarker); highlightMarker = null; }
}

function drawOverlays(data) {
    clearOverlays();

    // Draw catalog stars as red circle markers
    if (data.catalog_stars) {
        data.catalog_stars.forEach(star => {
            let radius = Math.max(2, Math.min(8, 8 - star.vmag * 0.8));
            let m = L.circleMarker([star.y, star.x], {
                radius: radius,
                color: '#ff4444',
                weight: 1,
                fillOpacity: 0,
                opacity: 0.6,
            }).addTo(map);
            m._starData = {type: 'catalog', idx: star.idx, az_deg: star.az_deg,
                           alt_deg: star.alt_deg, label: 'v=' + star.vmag.toFixed(1),
                           x: star.x, y: star.y};
            m.on('click', onMarkerClick);
            catMarkers.push(m);
        });
    }

    // Draw named objects as larger markers with labels
    if (data.named_objects) {
        data.named_objects.forEach(obj => {
            // Diamond-shaped divIcon
            let color = obj.category === 'planet' ? '#ffaa00' : '#ffcc44';
            let iconHtml = '<div style="width:12px;height:12px;border:2px solid ' + color +
                ';transform:rotate(45deg);background:transparent;"></div>';
            let icon = L.divIcon({
                html: iconHtml,
                className: '',
                iconSize: [12, 12],
                iconAnchor: [6, 6],
            });
            let m = L.marker([obj.y, obj.x], {icon: icon}).addTo(map);
            m._starData = {type: 'named', idx: obj.idx, az_deg: obj.az_deg,
                           alt_deg: obj.alt_deg, label: obj.name,
                           x: obj.x, y: obj.y};
            m.on('click', onMarkerClick);
            namedMarkers.push(m);

            // Label tooltip (permanent)
            let labelIcon = L.divIcon({
                html: '<span class="star-label">' + obj.name + '</span>',
                className: '',
                iconSize: [100, 16],
                iconAnchor: [-10, 8],
            });
            let lbl = L.marker([obj.y, obj.x], {
                icon: labelIcon,
                interactive: false,
            }).addTo(map);
            namedMarkers.push(lbl);
        });
    }

    // Draw corrections
    if (data.corrections) {
        data.corrections.forEach(c => {
            drawCorrection(c, data);
        });
    }
}

function drawCorrection(c, data) {
    // Find predicted position from current model
    // We need to search named objects and catalog stars for the original position
    let predX = c.pred_x;
    let predY = c.pred_y;
    let actX = c.actual_x;
    let actY = c.actual_y;

    if (predX !== undefined && predY !== undefined) {
        // Draw arrow line from predicted to actual
        let line = L.polyline([[predY, predX], [actY, actX]], {
            color: '#00ffff',
            weight: 2,
            opacity: 0.8,
            dashArray: '6,4',
        }).addTo(map);
        correctionLines.push(line);
    }

    // Actual position marker
    let cm = L.circleMarker([actY, actX], {
        radius: 6,
        color: '#00ffff',
        weight: 2,
        fillOpacity: 0,
    }).addTo(map);
    correctionMarkers.push(cm);
}

function onMarkerClick(e) {
    L.DomEvent.stopPropagation(e);
    let sd = e.target._starData;
    if (!sd) return;

    selectedStar = sd;

    // Highlight the selected marker
    if (highlightMarker) { map.removeLayer(highlightMarker); }
    highlightMarker = L.circleMarker([sd.y, sd.x], {
        radius: 14,
        color: '#00ffff',
        weight: 3,
        fillOpacity: 0,
        dashArray: '4,4',
    }).addTo(map);

    setPrompt('Now click where ' + sd.label + ' actually is in the image.', true);
}

function onMapClick(e) {
    if (!selectedStar) return;

    let clickY = e.latlng.lat;  // Leaflet lat = our pixel y
    let clickX = e.latlng.lng;  // Leaflet lng = our pixel x

    // Build correction data
    let corrData = {
        star_type: selectedStar.type,
        star_idx: selectedStar.idx,
        az_deg: selectedStar.az_deg,
        alt_deg: selectedStar.alt_deg,
        actual_x: clickX,
        actual_y: clickY,
        pred_x: selectedStar.x,
        pred_y: selectedStar.y,
        label: selectedStar.label,
    };

    // Clear selection highlight
    if (highlightMarker) { map.removeLayer(highlightMarker); highlightMarker = null; }
    selectedStar = null;
    setPrompt('Sending correction...');

    // Send to backend
    fetch('/api/correct', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(corrData),
    })
    .then(r => r.json())
    .then(result => {
        if (result.stars) {
            // Model was re-solved, redraw everything
            starData = result.stars;
            drawOverlays(result.stars);
            updateStats(result.stars);
            updateModelInfo(result.stars.model_params);
        } else {
            // Just add the correction arrow locally
            drawCorrection(corrData, starData);
        }
        setPrompt('Click any marker to select, then click actual position.');
    })
    .catch(err => {
        setPrompt('Error: ' + err.message);
    });
}

// -- Actions --

function showSpinner(msg) {
    let el = document.getElementById('spinner');
    el.querySelector('.inner').textContent = msg || 'Working...';
    el.classList.add('active');
}
function hideSpinner() {
    document.getElementById('spinner').classList.remove('active');
}

function doRefine(fitDistortion) {
    let label = fitDistortion ? 'Refining with distortion...' : 'Refining...';
    showSpinner(label);
    fetch('/api/refine', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({fit_distortion: fitDistortion}),
    })
    .then(r => r.json())
    .then(result => {
        hideSpinner();
        if (result.status === 'ok' && result.stars) {
            starData = result.stars;
            drawOverlays(result.stars);
            updateStats(result.stars);
            updateModelInfo(result.stars.model_params);
            setPrompt('Refined: ' + result.n_matched + ' matches, RMS=' +
                       result.rms.toFixed(2) + 'px');
        } else {
            setPrompt('Refine failed: ' + (result.message || 'unknown error'));
        }
    })
    .catch(err => { hideSpinner(); setPrompt('Error: ' + err.message); });
}

function doMirror() {
    showSpinner('Mirroring...');
    fetch('/api/mirror', { method: 'POST' })
    .then(r => r.json())
    .then(result => {
        hideSpinner();
        if (result.status === 'ok' && result.stars) {
            starData = result.stars;
            drawOverlays(result.stars);
            updateStats(result.stars);
            updateModelInfo(result.stars.model_params);
            setPrompt('Mirrored. Corrections cleared.');
        } else {
            setPrompt('Mirror failed: ' + (result.message || 'unknown error'));
        }
    })
    .catch(err => { hideSpinner(); setPrompt('Error: ' + err.message); });
}

function doUndo() {
    fetch('/api/undo', { method: 'POST' })
    .then(r => r.json())
    .then(result => {
        if (result.stars) {
            starData = result.stars;
            drawOverlays(result.stars);
            updateStats(result.stars);
            updateModelInfo(result.stars.model_params);
        }
        setPrompt('Undone last correction.');
    })
    .catch(err => { setPrompt('Error: ' + err.message); });
}

function doSave() {
    fetch('/api/save', { method: 'POST' })
    .then(r => r.json())
    .then(result => {
        if (result.status === 'ok') {
            setPrompt('Model saved to ' + result.path);
        } else {
            setPrompt('Save failed: ' + (result.message || 'unknown error'));
        }
    })
    .catch(err => { setPrompt('Error: ' + err.message); });
}

// -- UI helpers --

function setPrompt(text, isActive) {
    let el = document.getElementById('prompt-bar');
    el.textContent = text;
    if (isActive) {
        el.classList.add('active');
    } else {
        el.classList.remove('active');
    }
}

function updateStats(data) {
    document.getElementById('n-corrections').textContent = (data.corrections || []).length;
    document.getElementById('n-guided').textContent = data.n_guided || 0;
    let rms = data.guided_rms;
    if (rms && rms > 0) {
        document.getElementById('rms-val').textContent = rms.toFixed(2) + 'px';
    } else {
        document.getElementById('rms-val').textContent = '--';
    }
}

function updateModelInfo(params) {
    let el = document.getElementById('model-info');
    if (!params) {
        el.innerHTML = '<i>No model</i>';
        return;
    }
    el.innerHTML =
        'f = ' + params.f.toFixed(1) + ' px<br>' +
        'cx = ' + params.cx.toFixed(1) + ', cy = ' + params.cy.toFixed(1) + '<br>' +
        'az0 = ' + params.az0_deg.toFixed(1) + '&deg;<br>' +
        'alt0 = ' + params.alt0_deg.toFixed(1) + '&deg;<br>' +
        'rho = ' + params.rho_deg.toFixed(1) + '&deg;<br>' +
        'k1 = ' + params.k1.toExponential(2) + '<br>' +
        params.proj_type;
}

// -- Keyboard shortcuts --
document.addEventListener('keydown', function(e) {
    // Don't capture if user is typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch(e.key) {
        case 'r': doRefine(false); break;
        case 'd': doRefine(true); break;
        case 'm': doMirror(); break;
        case 'u': doUndo(); break;
        case 's': doSave(); break;
    }
});

// -- Start --
loadStars();
</script>
</body>
</html>
"""
