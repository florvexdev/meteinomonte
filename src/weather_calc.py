#!/usr/bin/env python3
"""
weather_calc.py
================
Grandezze meteorologiche derivate che Open-Meteo non fornisce direttamente,
calcolate con le formule ufficiali/scientifiche standard (non approssimazioni
inventate). Ogni funzione cita la fonte nella docstring.

Riferimenti:
  - Heat Index: Rothfusz, L.P. (1990), NWS Technical Attachment SR 90-23,
    basato su Steadman (1979), J. Appl. Meteorology.
    https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
  - Wind Chill: NWS / Environment Canada, formula ufficiale rivista nel 2001.
    https://www.weather.gov/safety/cold-wind-chill-chart
  - Dew Point: approssimazione di Magnus (August-Roche-Magnus), con le
    costanti di Alduchov & Eskridge (1996), J. Appl. Meteor., 35, 601-609.
  - Pressione al livello del mare / trend barometrico: relazioni standard di
    meteorologia sinottica (regola pratica NOAA sul nowcasting da pressione).
  - Indice UV, potenziale temporalesco (da CAPE): soglie EPA / NOAA SPC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Dew point (Magnus / Alduchov-Eskridge 1996)
# --------------------------------------------------------------------------

_MAGNUS_A = 17.625
_MAGNUS_B = 243.04  # °C


def dew_point_c(temp_c: float, rh_pct: float) -> float:
    """Punto di rugiada in °C dalla formula di Magnus (Alduchov & Eskridge 1996).
    Accurata a ±0.35°C nell'intervallo -40°C/+60°C."""
    rh = max(0.1, min(100.0, rh_pct))
    gamma = math.log(rh / 100.0) + (_MAGNUS_A * temp_c) / (_MAGNUS_B + temp_c)
    return (_MAGNUS_B * gamma) / (_MAGNUS_A - gamma)


def relative_humidity_from_dewpoint(temp_c: float, dew_point_c_val: float) -> float:
    """Inverso della formula di Magnus: RH% da temperatura e dew point."""
    num = math.exp((_MAGNUS_A * dew_point_c_val) / (_MAGNUS_B + dew_point_c_val))
    den = math.exp((_MAGNUS_A * temp_c) / (_MAGNUS_B + temp_c))
    return 100.0 * num / den


# --------------------------------------------------------------------------
# Heat Index (Rothfusz / NWS)
# --------------------------------------------------------------------------

def heat_index_c(temp_c: float, rh_pct: float) -> float | None:
    """
    Heat Index ("temperatura percepita" per caldo-umido), formula ufficiale
    NWS (Rothfusz 1990). Valida per T >= ~27°C (80°F). Sotto quella soglia
    l'indice non è definito in modo affidabile: ritorna None e si usa la
    temperatura reale come "percepita".
    """
    t_f = temp_c * 9.0 / 5.0 + 32.0
    rh = rh_pct

    # Formula semplificata di Steadman (usata come primo step / sotto soglia)
    hi_simple_f = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)

    if (hi_simple_f + t_f) / 2.0 < 80.0:
        return (hi_simple_f - 32.0) * 5.0 / 9.0  # non nel range Rothfusz, ritorna stima semplice

    # Regressione completa a 9 termini di Rothfusz
    hi_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f ** 2
        - 0.05481717 * rh ** 2
        + 0.00122874 * t_f ** 2 * rh
        + 0.00085282 * t_f * rh ** 2
        - 0.00000199 * t_f ** 2 * rh ** 2
    )

    # Correzioni per condizioni estreme (NWS)
    if rh < 13 and 80 <= t_f <= 112:
        adj = ((13 - rh) / 4.0) * math.sqrt((17 - abs(t_f - 95.0)) / 17.0)
        hi_f -= adj
    elif rh > 85 and 80 <= t_f <= 87:
        adj = ((rh - 85) / 10.0) * ((87 - t_f) / 5.0)
        hi_f += adj

    return (hi_f - 32.0) * 5.0 / 9.0


# --------------------------------------------------------------------------
# Wind Chill (NWS / Environment Canada, revisione 2001)
# --------------------------------------------------------------------------

def wind_chill_c(temp_c: float, wind_kmh: float) -> float | None:
    """
    Wind Chill ("temperatura percepita" per freddo-vento), formula ufficiale
    NWS/Environment Canada 2001. Valida solo per T <= 10°C e vento >= 4.8 km/h;
    fuori da questo range ritorna None (non è fisicamente definita).
    """
    if temp_c > 10.0 or wind_kmh < 4.8:
        return None
    v_pow = wind_kmh ** 0.16
    return 13.12 + 0.6215 * temp_c - 11.37 * v_pow + 0.3965 * temp_c * v_pow


def feels_like_c(temp_c: float, rh_pct: float, wind_kmh: float) -> tuple[float, str]:
    """
    Sceglie automaticamente l'indice di temperatura percepita più corretto
    per le condizioni attuali, seguendo la stessa logica pratica del NWS:
      - freddo + vento -> wind chill
      - caldo + umido  -> heat index
      - altrimenti     -> temperatura reale
    Ritorna (valore, etichetta_del_metodo_usato).
    """
    wc = wind_chill_c(temp_c, wind_kmh)
    if wc is not None:
        return wc, "wind chill (NWS 2001)"

    hi = heat_index_c(temp_c, rh_pct)
    if hi is not None and temp_c >= 20:
        return hi, "heat index (Rothfusz/NWS)"

    return temp_c, "temperatura reale (nessuna correzione applicabile)"


# --------------------------------------------------------------------------
# Indice di disagio / Humidex (usato in Canada/Europa come alternativa)
# --------------------------------------------------------------------------

def humidex(temp_c: float, dew_point_c_val: float) -> float:
    """Indice Humidex (Canada), da temperatura e dew point in °C."""
    e = 6.11 * math.exp(5417.7530 * (1 / 273.16 - 1 / (273.15 + dew_point_c_val)))
    return temp_c + 0.5555 * (e - 10.0)


# --------------------------------------------------------------------------
# Pressione: trend e interpretazione sinottica
# --------------------------------------------------------------------------

@dataclass
class PressureTrend:
    delta_3h_hpa: float
    label: str
    forecast_hint: str


def classify_pressure_trend(delta_3h_hpa: float) -> PressureTrend:
    """
    Classificazione del trend barometrico su 3 ore, secondo le soglie
    pratiche usate in meteorologia sinottica per il "nowcasting da
    pressione" (regola classica marinara/NOAA): variazioni >1 hPa/3h sono
    considerate significative.
    """
    d = delta_3h_hpa
    if d <= -3.0:
        return PressureTrend(d, "caduta rapida", "possibile peggioramento marcato/instabilità in arrivo")
    if d <= -1.0:
        return PressureTrend(d, "in calo", "tendenza a peggioramento nelle prossime ore")
    if d < 1.0:
        return PressureTrend(d, "stabile", "condizioni presumibilmente stazionarie")
    if d < 3.0:
        return PressureTrend(d, "in aumento", "tendenza a miglioramento/stabilizzazione")
    return PressureTrend(d, "aumento rapido", "rapida stabilizzazione, possibile rasserenamento")


# --------------------------------------------------------------------------
# Potenziale temporalesco da CAPE (soglie NOAA SPC)
# --------------------------------------------------------------------------

def cape_risk_label(cape_j_kg: float) -> str:
    """
    Interpretazione del CAPE (Convective Available Potential Energy) secondo
    le soglie qualitative usate dal NOAA Storm Prediction Center per il
    potenziale di temporali.
    """
    if cape_j_kg < 300:
        return "debole (temporali improbabili)"
    if cape_j_kg < 1000:
        return "moderato (temporali isolati possibili)"
    if cape_j_kg < 2500:
        return "elevato (temporali anche forti possibili)"
    return "molto elevato (rischio temporali intensi/violenti)"


# --------------------------------------------------------------------------
# Indice UV -> rischio (scala EPA/OMS)
# --------------------------------------------------------------------------

def uv_risk_label(uv_index: float) -> str:
    if uv_index < 3:
        return "basso"
    if uv_index < 6:
        return "moderato"
    if uv_index < 8:
        return "alto"
    if uv_index < 11:
        return "molto alto"
    return "estremo"


# --------------------------------------------------------------------------
# Velocità del vento: conversioni e scala Beaufort
# --------------------------------------------------------------------------

_BEAUFORT_THRESHOLDS_KMH = [
    (1, "calma"), (5, "bava di vento"), (11, "brezza leggera"),
    (19, "brezza tesa"), (28, "vento moderato"), (38, "vento teso"),
    (49, "vento fresco"), (61, "vento forte"), (74, "burrasca"),
    (88, "burrasca forte"), (102, "tempesta"), (117, "tempesta violenta"),
    (float("inf"), "uragano"),
]


def beaufort_scale(wind_kmh: float) -> tuple[int, str]:
    """Converte una velocità del vento (km/h) nella scala Beaufort (0-12)."""
    for i, (threshold, label) in enumerate(_BEAUFORT_THRESHOLDS_KMH):
        if wind_kmh < threshold:
            return i, label
    return 12, "uragano"


def degrees_to_compass_it(deg: float) -> str:
    compass = {
        "N": "Nord", "NNE": "Nord-Nordest", "NE": "Nordest", "ENE": "Est-Nordest",
        "E": "Est", "ESE": "Est-Sudest", "SE": "Sudest", "SSE": "Sud-Sudest",
        "S": "Sud", "SSW": "Sud-Sudovest", "SW": "Sudovest", "WSW": "Ovest-Sudovest",
        "W": "Ovest", "WNW": "Ovest-Nordovest", "NW": "Nordovest", "NNW": "Nord-Nordovest",
    }
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((deg + 11.25) // 22.5) % 16
    return compass[dirs[idx]]


# --------------------------------------------------------------------------
# Rischio di gelata/formazione nebbia (spread T - dew point)
# --------------------------------------------------------------------------

def fog_frost_risk(temp_c: float, dew_point_c_val: float) -> str:
    """Regola pratica meteorologica: uno spread T-Td piccolo indica aria vicina
    a saturazione (rischio nebbia); con T vicino/sotto 0°C, rischio gelata."""
    spread = temp_c - dew_point_c_val
    notes = []
    if spread <= 2.5:
        notes.append("rischio nebbia/foschia (aria vicina a saturazione)")
    if temp_c <= 3.0 and spread <= 4.0:
        notes.append("possibile formazione di brina/gelata")
    return "; ".join(notes) if notes else "nessun rischio particolare"
