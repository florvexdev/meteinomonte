#!/usr/bin/env python3
"""
radar_nowcast.py
=================
Nowcasting radar "professionale" basato su RainViewer (dati) + pysteps
(algoritmo). Sostituisce l'approccio precedente (cross-correlazione FFT
grezza) con lo standard effettivamente usato dai servizi meteorologici e
nella letteratura scientifica per il nowcasting a brevissimo termine:

  1. Si scaricano N frame radar consecutivi (RainViewer, tile PNG).
  2. Si stima il campo di moto (motion field) con l'algoritmo
     Lucas-Kanade denso (Lucas & Kanade, 1981), implementato in pysteps
     e usato in produzione da MeteoSwiss, KNMI e altri servizi europei
     (Pulkkinen et al. 2019, "pysteps: an open-source Python library for
     probabilistic precipitation nowcasting", GMD).
  3. Si estrapola il campo di pioggia in avanti nel tempo per advezione
     Lagrangiana semplice lungo il campo di moto stimato (la stessa idea
     fisica alla base di S-PROG/STEPS, qui applicata in forma
     deterministica per restare leggera su un runner CI gratuito).

Riferimenti:
  - Lucas, B.D. & Kanade, T. (1981), "An iterative image registration
    technique with an application to stereo vision", IJCAI.
  - Pulkkinen, S. et al. (2019), pysteps, Geosci. Model Dev., 12, 4185-4219.
  - RainViewer public API: https://www.rainviewer.com/api.html
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("montesilvano-meteo.radar")

RAINVIEWER_INDEX_URL = "https://api.rainviewer.com/public/weather-maps.json"
TILE_SIZE_PX = 256
ZOOM = 8  # ~ buon compromesso risoluzione/copertura per un raggio di ~20-30km
FRAME_INTERVAL_MIN = 10.0  # intervallo tipico tra frame RainViewer

# Mosaico per lo screenshot inviato su Telegram: griglia NxN di tile attorno
# al target, così l'area intorno a Montesilvano è visibile per intero e non
# solo il singolo tile usato per l'analisi numerica.
MOSAIC_GRID = 3  # 3x3 tile => copertura ~ maggiore del solo tile centrale
BASEMAP_ZOOM = 8


@dataclass
class NowcastResult:
    ok: bool
    error: str | None = None
    frames_used: int = 0
    motion_speed_kmh: float | None = None
    motion_direction_compass: str | None = None
    motion_confidence: str = "n/d"  # basata su coerenza spaziale del campo di moto
    rain_now_pct_area: float = 0.0
    rain_now: bool = False
    forecast_15min_pct_area: float | None = None
    forecast_30min_pct_area: float | None = None
    forecast_60min_pct_area: float | None = None
    eta_rain_minutes: float | None = None
    method: str = "Lucas-Kanade dense optical flow (pysteps) + advezione Lagrangiana"
    frame_time_utc: str | None = None  # timestamp reale dell'ultimo frame RainViewer usato
    image_png: bytes | None = None  # screenshot radar componibile, per invio via Telegram


def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def _vector_to_compass(dx: float, dy: float) -> str:
    angle = (math.degrees(math.atan2(dx, -dy)) + 360) % 360
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((angle + 11.25) // 22.5) % 16
    return dirs[idx]


async def _get_frame_index(client: httpx.AsyncClient) -> dict:
    r = await client.get(RAINVIEWER_INDEX_URL, timeout=20.0)
    r.raise_for_status()
    data = r.json()
    return {
        "host": data.get("host", "https://tilecache.rainviewer.com"),
        "past": data.get("radar", {}).get("past", []),
    }


async def _fetch_tile(client: httpx.AsyncClient, host: str, path: str, x: int, y: int, z: int) -> np.ndarray | None:
    url = f"{host}{path}/{TILE_SIZE_PX}/{z}/{x}/{y}/4/1_1.png"
    try:
        r = await client.get(url, timeout=20.0)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")
        return np.array(img)
    except Exception as e:  # noqa: BLE001
        log.warning("Tile radar non scaricabile (%s): %s", url, e)
        return None


def _to_intensity(tile_rgba: np.ndarray) -> np.ndarray:
    """Il canale alpha di RainViewer codifica l'intensità di precipitazione."""
    return tile_rgba[:, :, 3].astype(np.float64)


def _km_per_pixel(lat: float, zoom: int) -> float:
    tile_size_km = 40075.0 * math.cos(math.radians(lat)) / (2 ** zoom)
    return tile_size_km / TILE_SIZE_PX


def _tile_to_lat_lon(xtile: float, ytile: float, zoom: int) -> tuple[float, float]:
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    return math.degrees(lat_rad), lon_deg


async def _build_radar_snapshot(
    client: httpx.AsyncClient,
    host: str,
    path: str,
    lat: float,
    lon: float,
    frame_time_iso: str,
    motion_dir: str | None,
    motion_speed_kmh: float | None,
    rain_now_pct: float,
) -> bytes | None:
    """Scarica un mosaico MOSAIC_GRID x MOSAIC_GRID di tile RainViewer attorno
    al target, li assembla in un'unica immagine e disegna: marker del target,
    legenda intensità pioggia, freccia di moto celle, timestamp reale del
    frame. Restituisce PNG bytes pronti per l'invio (sendPhoto)."""
    try:
        xt, yt = _lat_lon_to_tile(lat, lon, BASEMAP_ZOOM)
        half = MOSAIC_GRID // 2

        mosaic = Image.new("RGBA", (TILE_SIZE_PX * MOSAIC_GRID, TILE_SIZE_PX * MOSAIC_GRID), (18, 22, 30, 255))
        for row in range(MOSAIC_GRID):
            for col in range(MOSAIC_GRID):
                tx, ty = xt + (col - half), yt + (row - half)
                url = f"{host}{path}/{TILE_SIZE_PX}/{BASEMAP_ZOOM}/{tx}/{ty}/2/1_1.png"
                try:
                    r = await client.get(url, timeout=20.0)
                    r.raise_for_status()
                    tile_img = Image.open(io.BytesIO(r.content)).convert("RGBA")
                except Exception as e:  # noqa: BLE001
                    log.warning("Tile mosaico non scaricabile (%s): %s", url, e)
                    tile_img = Image.new("RGBA", (TILE_SIZE_PX, TILE_SIZE_PX), (30, 34, 42, 255))
                mosaic.paste(tile_img, (col * TILE_SIZE_PX, row * TILE_SIZE_PX))

        # Sfondo scuro dietro alle tile semi-trasparenti (RainViewer ha alpha
        # variabile: sopra a un base scuro il radar resta leggibile anche
        # dove non piove).
        base = Image.new("RGBA", mosaic.size, (15, 20, 28, 255))
        base.alpha_composite(mosaic)
        canvas = base.convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")

        w, h = canvas.size
        cx, cy = w // 2, h // 2  # il target è sempre al centro del mosaico

        # --- Marker target (Montesilvano) ---
        r_marker = 7
        draw.ellipse((cx - r_marker, cy - r_marker, cx + r_marker, cy + r_marker),
                     fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(220, 30, 30, 255))

        try:
            font = ImageFont.load_default()
        except Exception:  # noqa: BLE001
            font = None

        label = "Montesilvano (PE)"
        draw.text((cx + 12, cy - 8), label, fill=(255, 255, 255, 255), font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0, 255))

        # --- Freccia di moto celle (se disponibile) ---
        if motion_dir and motion_speed_kmh and motion_speed_kmh > 0.5:
            compass_deg = {
                "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
                "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
                "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
            }.get(motion_dir, 0)
            rad = math.radians(compass_deg)
            length = 55
            ex = cx + length * math.sin(rad)
            ey = cy - length * math.cos(rad)
            draw.line((cx, cy, ex, ey), fill=(255, 210, 0, 255), width=4)
            ah = 10
            left = math.radians(compass_deg + 150)
            right = math.radians(compass_deg - 150)
            draw.line((ex, ey, ex + ah * math.sin(left), ey - ah * math.cos(left)), fill=(255, 210, 0, 255), width=4)
            draw.line((ex, ey, ex + ah * math.sin(right), ey - ah * math.cos(right)), fill=(255, 210, 0, 255), width=4)

        # --- Riquadro info in alto a sinistra ---
        try:
            dt = datetime.fromtimestamp(int(frame_time_iso), tz=timezone.utc)
            time_txt = dt.strftime("%H:%M UTC (%d/%m)")
        except (ValueError, TypeError):
            time_txt = str(frame_time_iso)

        info_lines = [
            "RADAR - RainViewer (osservato)",
            f"Frame: {time_txt}",
            f"Copertura pioggia locale: {rain_now_pct:.1f}%",
        ]
        if motion_dir and motion_speed_kmh:
            info_lines.append(f"Moto celle: → {motion_dir} · {motion_speed_kmh:.0f} km/h")

        pad = 6
        line_h = 14
        box_w = 260
        box_h = pad * 2 + line_h * len(info_lines)
        draw.rectangle((8, 8, 8 + box_w, 8 + box_h), fill=(0, 0, 0, 160))
        for i, line in enumerate(info_lines):
            draw.text((8 + pad, 8 + pad + i * line_h), line, fill=(255, 255, 255, 255), font=font)

        # --- Legenda intensità pioggia (scala colori standard RainViewer) ---
        legend = [
            ("Debole", (100, 190, 255)),
            ("Moderata", (60, 220, 60)),
            ("Forte", (255, 210, 0)),
            ("Molto forte", (255, 60, 60)),
        ]
        lx, ly = 8, h - 8 - (len(legend) * 16 + pad)
        draw.rectangle((lx, ly - pad, lx + 130, h - 8), fill=(0, 0, 0, 160))
        for i, (name, color) in enumerate(legend):
            yy = ly + i * 16
            draw.rectangle((lx + 6, yy, lx + 18, yy + 10), fill=color + (255,))
            draw.text((lx + 24, yy - 1), name, fill=(255, 255, 255, 255), font=font)

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:  # noqa: BLE001
        log.warning("Composizione screenshot radar fallita: %s", e)
        return None


async def run_nowcast(client: httpx.AsyncClient, lat: float, lon: float, n_frames: int = 4) -> NowcastResult:
    """
    Esegue il nowcasting radar professionale sulla zona attorno a (lat, lon).
    Usa fino a `n_frames` frame passati di RainViewer (di norma ogni 10 min).
    """
    try:
        # Import qui per rendere il fallimento di pysteps/opencv non fatale
        # per il resto del programma se, in futuro, la dipendenza non fosse
        # disponibile per qualche motivo nell'ambiente CI.
        from pysteps import motion as pysteps_motion
    except ImportError as e:  # noqa: BLE001
        return NowcastResult(ok=False, error=f"pysteps non disponibile: {e}")

    try:
        idx = await _get_frame_index(client)
        past = idx["past"]
        if len(past) < 2:
            return NowcastResult(ok=False, error="Meno di 2 frame radar disponibili da RainViewer")

        frames_meta = past[-min(n_frames, len(past)):]
        xt, yt = _lat_lon_to_tile(lat, lon, ZOOM)

        arrays = []
        for frame in frames_meta:
            tile = await _fetch_tile(client, idx["host"], frame["path"], xt, yt, ZOOM)
            if tile is not None:
                arrays.append(_to_intensity(tile))

        if len(arrays) < 2:
            return NowcastResult(ok=False, error="Impossibile scaricare abbastanza tile radar validi")

        stack = np.stack(arrays, axis=0)  # (T, H, W)

        # --- Stima del campo di moto con Lucas-Kanade denso (pysteps) ---
        oflow_method = pysteps_motion.get_method("LK")
        motion_field = oflow_method(stack)  # shape (2, H, W): componenti u, v in px/frame

        u_field, v_field = motion_field[0], motion_field[1]
        valid = np.isfinite(u_field) & np.isfinite(v_field)
        # Confidenza: quanto è coerente il campo di moto (bassa varianza = alta confidenza)
        if valid.sum() > 10:
            u_valid, v_valid = u_field[valid], v_field[valid]
            u_mean, v_mean = float(np.median(u_valid)), float(np.median(v_valid))
            speed_std = float(np.std(np.hypot(u_valid, v_valid)))
            speed_mean_px = math.hypot(u_mean, v_mean)
            confidence = "alta" if speed_std < max(1.0, speed_mean_px * 0.5) else "bassa (campo di moto poco coerente)"
        else:
            u_mean, v_mean = 0.0, 0.0
            confidence = "n/d (dati radar insufficienti nella zona)"

        km_px = _km_per_pixel(lat, ZOOM)
        speed_kmh = (math.hypot(u_mean, v_mean) * km_px / FRAME_INTERVAL_MIN) * 60.0
        direction_compass = _vector_to_compass(u_mean, v_mean) if (u_mean or v_mean) else None

        # --- Stato attuale sull'area locale (centro tile ~ target) ---
        latest = arrays[-1]
        h, w = latest.shape
        # finestra centrale ristretta (~ raggio 8km attorno al target, più
        # rappresentativa del "sopra Montesilvano" rispetto all'intero tile)
        margin = int(h * 0.35)
        center = latest[margin:h - margin, margin:w - margin]
        threshold = 15.0
        rain_now_pct = 100.0 * float((center > threshold).sum()) / center.size
        rain_now = rain_now_pct > 2.0

        # --- Estrapolazione Lagrangiana per +15/+30/+60 minuti ---
        def _advect(field: np.ndarray, steps: int) -> np.ndarray:
            """Trasla il campo di intensità lungo (u_mean, v_mean) di `steps`
            intervalli di FRAME_INTERVAL_MIN, con interpolazione bilineare
            semplice via np.roll (approssimazione a griglia intera)."""
            shifted = field.copy()
            dx_total = int(round(u_mean * steps))
            dy_total = int(round(v_mean * steps))
            shifted = np.roll(shifted, shift=dy_total, axis=0)
            shifted = np.roll(shifted, shift=dx_total, axis=1)
            return shifted

        def _coverage_after(steps: int) -> float:
            projected = _advect(latest, steps)
            proj_center = projected[margin:h - margin, margin:w - margin]
            return 100.0 * float((proj_center > threshold).sum()) / proj_center.size

        fc_15 = _coverage_after(int(round(15 / FRAME_INTERVAL_MIN)))
        fc_30 = _coverage_after(int(round(30 / FRAME_INTERVAL_MIN)))
        fc_60 = _coverage_after(int(round(60 / FRAME_INTERVAL_MIN)))

        eta_minutes = None
        if not rain_now and speed_kmh > 0.5:
            for minutes in range(5, 121, 5):
                steps = minutes / FRAME_INTERVAL_MIN
                if _coverage_after(int(round(steps))) > 2.0:
                    eta_minutes = float(minutes)
                    break

        # Timestamp reale (epoch) dell'ultimo frame RainViewer effettivamente usato.
        last_frame_time = frames_meta[-1].get("time")

        image_png = await _build_radar_snapshot(
            client, idx["host"], frames_meta[-1]["path"], lat, lon,
            frame_time_iso=last_frame_time,
            motion_dir=direction_compass,
            motion_speed_kmh=round(speed_kmh, 1) if speed_kmh else None,
            rain_now_pct=rain_now_pct,
        )

        return NowcastResult(
            ok=True,
            frames_used=len(arrays),
            motion_speed_kmh=round(speed_kmh, 1),
            motion_direction_compass=direction_compass,
            motion_confidence=confidence,
            rain_now_pct_area=round(rain_now_pct, 1),
            rain_now=rain_now,
            forecast_15min_pct_area=round(fc_15, 1),
            forecast_30min_pct_area=round(fc_30, 1),
            forecast_60min_pct_area=round(fc_60, 1),
            eta_rain_minutes=eta_minutes,
            frame_time_utc=str(last_frame_time) if last_frame_time else None,
            image_png=image_png,
        )

    except Exception as e:  # noqa: BLE001
        log.exception("Nowcast radar fallito")
        return NowcastResult(ok=False, error=str(e))
