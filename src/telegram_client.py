#!/usr/bin/env python3
"""
telegram_client.py
====================
Client minimale per Telegram Bot API con supporto "edit-in-place": invece di
spammare un messaggio nuovo ogni ciclo, il bot MODIFICA lo stesso messaggio
via editMessageText, così la chat resta pulita con un solo messaggio che si
aggiorna. Se il messaggio non esiste ancora (primo avvio) o Telegram rifiuta
la modifica (es. messaggio troppo vecchio, >48h, o cancellato manualmente),
il bot ne invia uno nuovo e ne salva l'id per i cicli successivi.

Persistenza dell'id messaggio:
  GitHub Actions non mantiene stato tra un run e l'altro di default. Per
  avere edit-in-place reale servono due componenti:
    1. Questo modulo, che sa fare editMessageText dato un message_id.
    2. Il workflow YAML, che salva `state/message_id.txt` nel repo con un
       commit automatico a fine job (vedi .github/workflows/meteo.yml) così
       il prossimo run parte già sapendo quale messaggio modificare.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

log = logging.getLogger("montesilvano-meteo.telegram")

TELEGRAM_MAX_LEN = 4096
STATE_FILE = Path(__file__).resolve().parent.parent / "state" / "message_id.txt"


def read_saved_message_id() -> int | None:
    try:
        if STATE_FILE.exists():
            content = STATE_FILE.read_text().strip()
            return int(content) if content else None
    except (ValueError, OSError) as e:  # noqa: BLE001
        log.warning("Impossibile leggere message_id salvato: %s", e)
    return None


def save_message_id(message_id: int) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(str(message_id))
    except OSError as e:  # noqa: BLE001
        log.warning("Impossibile salvare message_id: %s", e)


def clear_saved_message_id() -> None:
    try:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
    except OSError:  # noqa: BLE001
        pass


def _truncate_for_telegram(text: str) -> str:
    if len(text) <= TELEGRAM_MAX_LEN:
        return text
    cut = text[: TELEGRAM_MAX_LEN - 40]
    last_newline = cut.rfind("\n")
    if last_newline > TELEGRAM_MAX_LEN // 2:
        cut = cut[:last_newline]
    return cut + "\n\n[…troncato per limite Telegram…]"


async def send_new_message(client: httpx.AsyncClient, token: str, chat_id: str, text: str) -> int | None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": _truncate_for_telegram(text),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = await client.post(url, json=payload, timeout=20.0)
    r.raise_for_status()
    data = r.json()
    return data.get("result", {}).get("message_id")


async def edit_message(client: httpx.AsyncClient, token: str, chat_id: str, message_id: int, text: str) -> bool:
    """Ritorna True se la modifica è riuscita, False se va rifatto un invio nuovo."""
    url = f"https://api.telegram.org/bot{token}/editMessageText"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": _truncate_for_telegram(text),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = await client.post(url, json=payload, timeout=20.0)
    if r.status_code == 200:
        return True

    body = r.text
    # "message is not modified" non è un vero errore: il contenuto è identico
    if "message is not modified" in body:
        return True

    log.warning("editMessageText fallito (%s): %s — invio un nuovo messaggio.", r.status_code, body)
    return False


async def send_photo(
    client: httpx.AsyncClient,
    token: str,
    chat_id: str,
    photo_bytes: bytes,
    caption: str = "",
    filename: str = "radar.png",
) -> int | None:
    """Invia una foto (bytes PNG) come nuovo messaggio, con didascalia opzionale
    (max 1024 caratteri per i limiti di Telegram su sendPhoto)."""
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    cap = caption[:1024] if caption else ""
    files = {"photo": (filename, photo_bytes, "image/png")}
    data = {"chat_id": chat_id, "caption": cap, "parse_mode": "HTML"}
    r = await client.post(url, data=data, files=files, timeout=30.0)
    if r.status_code != 200:
        log.warning("sendPhoto fallito (%s): %s", r.status_code, r.text)
        return None
    result = r.json().get("result", {})
    return result.get("message_id")


async def send_or_edit_report(
    client: httpx.AsyncClient,
    token: str,
    chat_id: str,
    text: str,
    radar_image_bytes: bytes | None = None,
    radar_caption: str = "",
) -> None:
    """
    Punto di ingresso principale: prova a modificare il messaggio testuale
    precedente; se non esiste o l'edit fallisce, invia un nuovo messaggio e
    ne salva l'id. Se è disponibile un'immagine radar, la invia SEMPRE come
    messaggio foto separato (editMessageText non può aggiungere/sostituire
    allegati su un messaggio di solo testo già esistente), così ogni ciclo
    mostra un'istantanea aggiornata della situazione radar.
    """
    if not token or not chat_id:
        log.error("Credenziali Telegram mancanti: report stampato solo su stdout.")
        print(text)
        return

    existing_id = read_saved_message_id()

    if existing_id is not None:
        ok = await edit_message(client, token, chat_id, existing_id, text)
        if not ok:
            clear_saved_message_id()
            new_id = await send_new_message(client, token, chat_id, text)
            if new_id is not None:
                save_message_id(new_id)
                log.info("Nuovo messaggio Telegram inviato (id=%s).", new_id)
        else:
            log.info("Messaggio Telegram %s aggiornato in-place.", existing_id)
    else:
        new_id = await send_new_message(client, token, chat_id, text)
        if new_id is not None:
            save_message_id(new_id)
            log.info("Nuovo messaggio Telegram inviato (id=%s).", new_id)

    if radar_image_bytes:
        photo_id = await send_photo(client, token, chat_id, radar_image_bytes, radar_caption)
        if photo_id is not None:
            log.info("Immagine radar inviata (message_id=%s).", photo_id)
        else:
            log.warning("Invio immagine radar fallito: proseguo comunque (report testuale già inviato).")


async def send_alert(client: httpx.AsyncClient, token: str, chat_id: str, error_text: str) -> None:
    """Alert di errore: sempre come messaggio nuovo separato (non va a
    sovrascrivere il report principale, altrimenti si perderebbe l'id buono)."""
    if not token or not chat_id:
        log.error("Impossibile inviare alert Telegram: credenziali mancanti.")
        return
    text = f"⚠️ <b>Montesilvano Meteo Watch — errore nel job</b>\n<code>{error_text[:3500]}</code>"
    try:
        await send_new_message(client, token, chat_id, text)
    except Exception:  # noqa: BLE001
        log.exception("Anche l'invio dell'alert di errore è fallito.")
