#!/usr/bin/env python3
"""
Montesilvano Meteo Watch
=========================
Orchestratore che interroga più fonti meteo gratuite in parallelo, calcola
grandezze derivate con formule scientifiche verificate, esegue un
nowcasting radar professionale (Lucas-Kanade optical flow via pysteps) e
invia/aggiorna un resoconto esteso via Telegram (edit-in-place).

Fonti dati (gratuite):
  1. Open-Meteo /v1/forecast — 5 modelli numerici indipendenti in parallelo.
  2. Open-Meteo air-quality API — qualità dell'aria e indice UV.
  3. RainViewer — radar globale in tile, usato per il nowcasting.

Grandezze calcolate localmente (vedi weather_calc.py per le fonti):
  Heat Index (Rothfusz/NWS), Wind Chill (NWS 2001), Dew Point (Magnus),
  Humidex, scala Beaufort, trend di pressione, rischio nebbia/gelata,
  interpretazione CAPE, rischio UV.

NOTA (onestà dei dati): non esiste una API REST gratuita/stabile per una
stazione fisica di Montesilvano; si usano modelli numerici multipli
(che assimilano dati stazioni/radar/satelliti a monte) + radar osservato.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

import radar_nowcast
import telegram_client
import weather_calc as wc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("montesilvano-meteo")

# --------------------------------------------------------------------------
# Configurazione
# --------------------------------------------------------------------------

LAT = float(os.environ.get("TARGET_LAT", "42.5091"))
LON = float(os.environ.get("TARGET_LON", "14.1516"))
LOCATION_NAME = os.environ.get("TARGET_NAME", "Montesilvano (PE)")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

HTTP_TIMEOUT = httpx.Timeout(25.0, connect=10.0)

OPEN_METEO_MODELS = [
    "best_match",
    "icon_seamless",
    "gfs_seamless",
    "ecmwf_ifs04",
    "meteofrance_seamless",
]

HOURLY_VARS = [
    "temperature_2m", "apparent_temperature", "precipitation_probability",
    "precipitation", "rain", "showers", "weather_code", "cloud_cover",
    "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "pressure_msl", "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "wind_gusts_10m", "relative_humidity_2m", "dew_point_2m", "visibility",
    "cape", "uv_index", "et0_fao_evapotranspiration", "vapour_pressure_deficit",
    "freezing_level_height", "snowfall",
]

CURRENT_VARS = [
    "temperature_2m", "apparent_temperature", "precipitation", "rain",
    "showers", "weather_code", "cloud_cover", "pressure_msl",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "wind_gusts_10m", "relative_humidity_2m", "is_day",
]

AIR_QUALITY_VARS = ["pm10", "pm2_5", "uv_index", "european_aqi", "ozone", "nitrogen_dioxide"]

WEATHER_CODE_IT = {
    0: "cielo sereno", 1: "prevalentemente sereno", 2: "parzialmente nuvoloso",
    3: "coperto", 45: "nebbia", 48: "nebbia con brina",
    51: "pioviggine debole", 53: "pioviggine moderata", 55: "pioviggine intensa",
    56: "pioviggine gelata debole", 57: "pioviggine gelata intensa",
    61: "pioggia debole", 63: "pioggia moderata", 65: "pioggia forte",
    66: "pioggia gelata debole", 67: "pioggia gelata forte",
    71: "neve debole", 73: "neve moderata", 75: "neve forte", 77: "granelli di neve",
    80: "rovesci deboli", 81: "rovesci moderati", 82: "rovesci violenti",
    85: "rovesci di neve deboli", 86: "rovesci di neve forti",
    95: "temporale", 96: "temporale con grandine debole", 99: "temporale con grandine forte",
}


# --------------------------------------------------------------------------
# Strutture dati
# --------------------------------------------------------------------------

@dataclass
class ModelResult:
    model: str
    ok: bool
    current: dict[str, Any] = field(default_factory=dict)
    hourly: dict[str, list] = field(default_factory=dict)
    error: str | None = None


# --------------------------------------------------------------------------
# Fetch: Open-Meteo multi-modello
# --------------------------------------------------------------------------

async def fetch_open_meteo_model(client: httpx.AsyncClient, model: str) -> ModelResult:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT, "longitude": LON,
        "current": ",".join(CURRENT_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "forecast_days": 2, "timezone": "auto", "models": model,
    }
    try:
        r = await client.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return ModelResult(model=model, ok=True, current=data.get("current", {}), hourly=data.get("hourly", {}))
    except Exception as e:  # noqa: BLE001
        log.warning("Modello %s fallito: %s", model, e)
        return ModelResult(model=model, ok=False, error=str(e))


async def fetch_all_models(client: httpx.AsyncClient) -> list[ModelResult]:
    return await asyncio.gather(*[fetch_open_meteo_model(client, m) for m in OPEN_METEO_MODELS])


async def fetch_air_quality(client: httpx.AsyncClient) -> dict[str, Any] | None:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {"latitude": LAT, "longitude": LON, "current": ",".join(AIR_QUALITY_VARS), "timezone": "auto"}
    try:
        r = await client.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json().get("current", {})
    except Exception as e:  # noqa: BLE001
        log.warning("Air quality fallita: %s", e)
        return None


async def fetch_station_placeholder(client: httpx.AsyncClient) -> dict[str, Any] | None:
    station_url = os.environ.get("CUSTOM_STATION_URL", "").strip()
    if not station_url:
        return None
    try:
        r = await client.get(station_url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        log.warning("Stazione custom fallita: %s", e)
        return None


# --------------------------------------------------------------------------
# Cross-correlazione tra modelli numerici
# --------------------------------------------------------------------------

def cross_correlate_models(results: list[ModelResult]) -> dict[str, Any]:
    ok_results = [r for r in results if r.ok and r.hourly.get("time")]
    if not ok_results:
        return {"agreement": "n/d", "models_ok": 0, "models_total": len(results)}

    ref_time = ok_results[0].hourly["time"]
    horizon_idx = min(3, len(ref_time) - 1)

    precs, temps, winds, wind_dirs, cloud = [], [], [], [], []
    for r in ok_results:
        try:
            precs.append(sum(r.hourly["precipitation"][:horizon_idx + 1]))
            temps.append(r.hourly["temperature_2m"][horizon_idx])
            winds.append(r.hourly["wind_speed_10m"][horizon_idx])
            wind_dirs.append(r.hourly["wind_direction_10m"][horizon_idx])
            cloud.append(r.hourly.get("cloud_cover", [None] * (horizon_idx + 1))[horizon_idx])
        except (KeyError, IndexError, TypeError):
            continue

    def _spread(vals):
        vals = [v for v in vals if v is not None]
        return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

    prec_spread = _spread(precs)
    any_rain = any(p > 0.1 for p in precs) if precs else False
    all_rain = all(p > 0.1 for p in precs) if precs else False

    if all_rain:
        agreement = "alto accordo: tutti i modelli vedono pioggia"
    elif any_rain and prec_spread > 1.5:
        agreement = "disaccordo: solo alcuni modelli vedono pioggia (bassa confidenza)"
    elif not any_rain:
        agreement = "alto accordo: nessun modello vede pioggia a breve"
    else:
        agreement = "moderato accordo"

    return {
        "agreement": agreement,
        "models_ok": len(ok_results),
        "models_total": len(results),
        "precipitation_mm_3h_range": (round(min(precs), 1), round(max(precs), 1)) if precs else None,
        "temperature_spread_c": round(_spread(temps), 1) if temps else None,
        "wind_spread_kmh": round(_spread(winds), 1) if winds else None,
        "avg_wind_direction_deg": round(sum(wind_dirs) / len(wind_dirs), 0) if wind_dirs else None,
        "avg_cloud_cover_pct": round(sum(c for c in cloud if c is not None) / max(1, len([c for c in cloud if c is not None])), 0) if cloud else None,
    }


# --------------------------------------------------------------------------
# Pressione: trend dalle ultime ore (best_match)
# --------------------------------------------------------------------------

def compute_pressure_trend(best: ModelResult | None) -> wc.PressureTrend | None:
    if not best or not best.hourly.get("pressure_msl"):
        return None
    pressures = best.hourly["pressure_msl"]
    times = best.hourly.get("time", [])
    now_naive = datetime.now().replace(tzinfo=None, second=0, microsecond=0)
    idx_now = 0
    for i, t in enumerate(times):
        try:
            t_dt = datetime.fromisoformat(t).replace(tzinfo=None)
        except ValueError:
            continue
        if t_dt <= now_naive:
            idx_now = i
        else:
            break
    idx_3h_ago = max(0, idx_now - 3)
    if idx_now >= len(pressures) or idx_3h_ago >= len(pressures):
        return None
    delta = pressures[idx_now] - pressures[idx_3h_ago]
    return wc.classify_pressure_trend(delta)


# --------------------------------------------------------------------------
# Costruzione report
# --------------------------------------------------------------------------

def _fmt(val, unit="", nd=1, dash="?"):
    if val is None:
        return dash
    try:
        return f"{round(float(val), nd)}{unit}"
    except (TypeError, ValueError):
        return f"{val}{unit}"


COMPASS_IT_MAP = {
    "N": "Nord", "NNE": "Nord-Nordest", "NE": "Nordest", "ENE": "Est-Nordest",
    "E": "Est", "ESE": "Est-Sudest", "SE": "Sudest", "SSE": "Sud-Sudest",
    "S": "Sud", "SSW": "Sud-Sudovest", "SW": "Sudovest", "WSW": "Ovest-Sudovest",
    "W": "Ovest", "WNW": "Ovest-Nordovest", "NW": "Nordovest", "NNW": "Nord-Nordovest",
}


def build_report(
    models: list[ModelResult],
    nowcast: radar_nowcast.NowcastResult,
    correlation: dict[str, Any],
    air_quality: dict | None,
    station: dict | None,
) -> str:
    now = datetime.now(timezone.utc).astimezone().strftime("%d/%m/%Y %H:%M")
    best = next((m for m in models if m.model == "best_match" and m.ok), None)

    L: list[str] = []
    L.append(f"🌦️ <b>METEO {LOCATION_NAME.upper()}</b>")
    L.append(f"<i>Aggiornato: {now}</i>")
    L.append("━━━━━━━━━━━━━━━━━━━")

    # ============================= SITUAZIONE ATTUALE =============================
    if best and best.current:
        c = best.current
        temp = c.get("temperature_2m")
        rh = c.get("relative_humidity_2m")
        wind = c.get("wind_speed_10m")
        wind_dir = c.get("wind_direction_10m")
        gusts = c.get("wind_gusts_10m")
        pressure = c.get("pressure_msl")
        cloud = c.get("cloud_cover")
        prec = c.get("precipitation", 0) or 0
        wcode = c.get("weather_code")

        L.append("")
        L.append("<b>📍 SITUAZIONE ATTUALE</b>")
        if wcode is not None:
            L.append(f"{WEATHER_CODE_IT.get(int(wcode), '—').capitalize()}")

        L.append(f"🌡️ Temperatura: <b>{_fmt(temp, '°C')}</b>")

        if temp is not None and rh is not None:
            dew = wc.dew_point_c(temp, rh)
            feels, feels_method = wc.feels_like_c(temp, rh, wind or 0)
            L.append(f"🥵 Percepita: <b>{_fmt(feels, '°C')}</b> <i>({feels_method})</i>")
            L.append(f"💧 Umidità relativa: {_fmt(rh, '%')} · Punto di rugiada: {_fmt(dew, '°C')}")
            humidex_val = wc.humidex(temp, dew)
            L.append(f"😓 Humidex: {_fmt(humidex_val, '°C')}")
            fog_risk = wc.fog_frost_risk(temp, dew)
            if fog_risk != "nessun rischio particolare":
                L.append(f"🌫️ {fog_risk.capitalize()}")

        if pressure is not None:
            L.append(f"📊 Pressione (livello mare): {_fmt(pressure, ' hPa', 0)}")
            trend = compute_pressure_trend(best)
            if trend:
                L.append(
                    f"   Trend 3h: {trend.delta_3h_hpa:+.1f} hPa — <b>{trend.label}</b> "
                    f"({trend.forecast_hint})"
                )

        if wind is not None:
            beaufort_n, beaufort_lbl = wc.beaufort_scale(wind)
            dir_txt = f" da {wc.degrees_to_compass_it(wind_dir)}" if wind_dir is not None else ""
            L.append(f"💨 Vento: {_fmt(wind, ' km/h')}{dir_txt} — Beaufort {beaufort_n} ({beaufort_lbl})")
            if gusts is not None:
                L.append(f"   Raffiche: {_fmt(gusts, ' km/h')}")

        if cloud is not None:
            L.append(f"☁️ Copertura nuvolosa: {_fmt(cloud, '%', 0)}")

        if prec and prec > 0:
            L.append(f"🌧️ Precipitazione in corso: <b>{_fmt(prec, ' mm/h')}</b>")

    # ============================= RADAR / NOWCASTING =============================
    L.append("")
    L.append("━━━━━━━━━━━━━━━━━━━")
    L.append("<b>📡 RADAR — NOWCASTING PROFESSIONALE</b>")
    L.append(f"<i>Metodo: {nowcast.method}</i>")

    if nowcast.ok:
        status_icon = "🔴" if nowcast.rain_now else "🟢"
        L.append(
            f"{status_icon} Copertura pioggia sull'area ora: "
            f"<b>{_fmt(nowcast.rain_now_pct_area, '%')}</b>"
        )
        if nowcast.motion_direction_compass and nowcast.motion_speed_kmh:
            L.append(
                f"➡️ Moto celle precipitative: verso "
                f"{COMPASS_IT_MAP.get(nowcast.motion_direction_compass, nowcast.motion_direction_compass)} "
                f"a {_fmt(nowcast.motion_speed_kmh, ' km/h')} "
                f"(confidenza: {nowcast.motion_confidence})"
            )
        L.append(f"   Frame radar analizzati: {nowcast.frames_used}")

        if nowcast.forecast_15min_pct_area is not None:
            L.append("")
            L.append("<b>Proiezione copertura pioggia (estrapolazione):</b>")
            L.append(f"   +15 min: {_fmt(nowcast.forecast_15min_pct_area, '%')}")
            L.append(f"   +30 min: {_fmt(nowcast.forecast_30min_pct_area, '%')}")
            L.append(f"   +60 min: {_fmt(nowcast.forecast_60min_pct_area, '%')}")

        if nowcast.eta_rain_minutes:
            L.append(f"⏱️ Possibile arrivo pioggia entro ~<b>{int(nowcast.eta_rain_minutes)} min</b> (stima)")
    else:
        L.append(f"⚠️ Radar non disponibile in questo ciclo ({nowcast.error}).")

    # ============================= MODELLI NUMERICI =============================
    L.append("")
    L.append("━━━━━━━━━━━━━━━━━━━")
    L.append("<b>🧮 CROSS-CORRELAZIONE MODELLI NUMERICI</b>")
    L.append("<i>ICON (DWD) · GFS (NOAA) · ECMWF-IFS · AROME/ARPEGE (Météo-France)</i>")
    L.append(f"Modelli rispondenti: {correlation.get('models_ok')}/{correlation.get('models_total')}")
    L.append(f"Verdetto: <b>{correlation.get('agreement')}</b>")
    if correlation.get("precipitation_mm_3h_range"):
        lo, hi = correlation["precipitation_mm_3h_range"]
        L.append(f"Pioggia attesa prossime 3h: {lo}–{hi} mm (range tra modelli)")
    if correlation.get("temperature_spread_c") is not None:
        L.append(f"Divergenza temperatura tra modelli: ±{correlation['temperature_spread_c']}°C")
    if correlation.get("wind_spread_kmh") is not None:
        L.append(f"Divergenza vento tra modelli: ±{correlation['wind_spread_kmh']} km/h")
    if correlation.get("avg_wind_direction_deg") is not None:
        L.append(f"Direzione vento media prevista: {wc.degrees_to_compass_it(correlation['avg_wind_direction_deg'])}")
    if correlation.get("avg_cloud_cover_pct") is not None:
        L.append(f"Copertura nuvolosa media prevista: {_fmt(correlation['avg_cloud_cover_pct'], '%', 0)}")

    # ============================= PROSSIME ORE =============================
    if best and best.hourly.get("time"):
        L.append("")
        L.append("━━━━━━━━━━━━━━━━━━━")
        L.append("<b>🕐 PROSSIME ORE (best match)</b>")
        times = best.hourly["time"]
        h = best.hourly
        for i in range(0, min(6, len(times))):
            try:
                t_dt = datetime.fromisoformat(times[i])
                t = t_dt.strftime("%H:%M")
            except (ValueError, IndexError):
                t = times[i]
            temp_i = h.get("temperature_2m", [None])[i]
            prec_p = h.get("precipitation_probability", [None])[i]
            prec_mm = h.get("precipitation", [None])[i]
            wind_i = h.get("wind_speed_10m", [None])[i]
            wind_dir_i = h.get("wind_direction_10m", [None])[i]
            uv_i = h.get("uv_index", [None])[i]
            dir_txt = f" {wc.degrees_to_compass_it(wind_dir_i)}" if wind_dir_i is not None else ""
            uv_txt = f" · UV {_fmt(uv_i, '', 0)} ({wc.uv_risk_label(uv_i)})" if uv_i is not None else ""
            L.append(
                f"<b>{t}</b> — {_fmt(temp_i, '°C')} · 🌧️{_fmt(prec_p, '%', 0)} ({_fmt(prec_mm, 'mm')}) "
                f"· 💨{_fmt(wind_i, 'km/h', 0)}{dir_txt}{uv_txt}"
            )

        cape_vals = h.get("cape", [])
        if cape_vals:
            max_cape = max((v for v in cape_vals[:6] if v is not None), default=None)
            if max_cape is not None:
                L.append(f"⚡ Potenziale temporalesco (CAPE max 6h): {_fmt(max_cape, ' J/kg', 0)} — {wc.cape_risk_label(max_cape)}")

        freeze_vals = h.get("freezing_level_height", [])
        if freeze_vals and freeze_vals[0] is not None:
            L.append(f"❄️ Quota zero termico: {_fmt(freeze_vals[0], ' m', 0)}")

    # ============================= QUALITÀ ARIA =============================
    if air_quality:
        L.append("")
        L.append("━━━━━━━━━━━━━━━━━━━")
        L.append("<b>🏭 QUALITÀ DELL'ARIA</b>")
        aqi = air_quality.get("european_aqi")
        pm10 = air_quality.get("pm10")
        pm25 = air_quality.get("pm2_5")
        uv_now = air_quality.get("uv_index")
        if aqi is not None:
            L.append(f"Indice qualità aria europeo: {_fmt(aqi, '', 0)}")
        if pm10 is not None or pm25 is not None:
            L.append(f"PM10: {_fmt(pm10, ' µg/m³', 0)} · PM2.5: {_fmt(pm25, ' µg/m³', 0)}")
        if uv_now is not None:
            L.append(f"☀️ Indice UV attuale: {_fmt(uv_now, '', 1)} ({wc.uv_risk_label(uv_now)})")

    # ============================= STAZIONE / NOTE =============================
    L.append("")
    L.append("━━━━━━━━━━━━━━━━━━━")
    if station:
        L.append("<b>📟 STAZIONE LOCALE</b>")
        L.append(str(station))
    else:
        L.append(
            "<i>ℹ️ Nessuna stazione fisica locale configurata: dati basati su "
            "modelli numerici multipli + radar osservato in tempo reale.</i>"
        )

    return "\n".join(L)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

async def run() -> None:
    async with httpx.AsyncClient(headers={"User-Agent": "montesilvano-meteo-watch/2.0"}) as client:
        models, nowcast, air_quality, station = await asyncio.gather(
            fetch_all_models(client),
            radar_nowcast.run_nowcast(client, LAT, LON),
            fetch_air_quality(client),
            fetch_station_placeholder(client),
        )

        correlation = cross_correlate_models(models)
        report = build_report(models, nowcast, correlation, air_quality, station)

        log.info("Report generato (%d caratteri)", len(report))

        radar_caption = None
        if nowcast.ok and nowcast.image_png:
            status_icon = "🔴" if nowcast.rain_now else "🟢"
            cap_lines = [
                f"📡 <b>Radar {LOCATION_NAME}</b> — {status_icon} copertura pioggia locale: {nowcast.rain_now_pct_area:.1f}%",
            ]
            if nowcast.motion_direction_compass and nowcast.motion_speed_kmh:
                cap_lines.append(
                    f"➡️ Celle in moto verso {nowcast.motion_direction_compass} "
                    f"a {nowcast.motion_speed_kmh:.0f} km/h"
                )
            cap_lines.append("<i>Fonte: RainViewer (osservato, non previsione)</i>")
            radar_caption = "\n".join(cap_lines)

        await telegram_client.send_or_edit_report(
            client, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, report,
            radar_image_bytes=nowcast.image_png if nowcast.ok else None,
            radar_caption=radar_caption or "",
        )


def main() -> None:
    try:
        asyncio.run(run())
    except Exception as e:  # noqa: BLE001
        log.exception("Errore fatale nel job")

        async def _alert():
            async with httpx.AsyncClient() as client:
                await telegram_client.send_alert(client, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"{type(e).__name__}: {e}")

        try:
            asyncio.run(_alert())
        except Exception:  # noqa: BLE001
            log.exception("Impossibile inviare alert di errore")
        sys.exit(1)


if __name__ == "__main__":
    main()
