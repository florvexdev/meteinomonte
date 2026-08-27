# Montesilvano Meteo Watch

Bot che invia (e poi **aggiorna in-place**, senza spam) su Telegram un
resoconto meteo molto esteso di Montesilvano (PE), ogni ~10 minuti,
incrociando più fonti gratuite e calcolando localmente grandezze derivate
con formule scientifiche verificate.

## Cosa contiene il resoconto

**Situazione attuale**
- Temperatura reale e **percepita** (sceglie automaticamente Heat Index o
  Wind Chill a seconda delle condizioni, come fa il NWS)
- Umidità relativa, punto di rugiada, Humidex
- Rischio nebbia/gelata (calcolato dallo spread temperatura–rugiada)
- Pressione al livello del mare **e trend delle ultime 3 ore** con
  interpretazione sinottica (in calo/stabile/in aumento)
- Vento: velocità, direzione (in italiano), raffiche, **scala Beaufort**
- Copertura nuvolosa, precipitazione in corso

**Radar — nowcasting professionale**
- Algoritmo **Lucas-Kanade dense optical flow** (via [pysteps](https://pysteps.readthedocs.io),
  la libreria open-source usata da MeteoSwiss/KNMI e nella letteratura
  scientifica sul nowcasting), non una cross-correlazione FFT fatta in casa
- Direzione e velocità di spostamento delle celle di pioggia, con un indice
  di **confidenza** sulla coerenza del campo di moto stimato
- Estrapolazione Lagrangiana della copertura pioggia a **+15/+30/+60 minuti**
- ETA stimato di arrivo pioggia se non sta già piovendo

**Cross-correlazione tra modelli numerici**
- 5 modelli indipendenti (Open-Meteo `best_match`, ICON/DWD, GFS/NOAA,
  ECMWF-IFS, AROME-ARPEGE/Météo-France) confrontati tra loro
- Indice di accordo/disaccordo, range di pioggia attesa, divergenza
  temperatura/vento tra modelli

**Prossime 6 ore**, con indice UV e classificazione di rischio, **CAPE**
(potenziale temporalesco, soglie NOAA SPC), quota dello zero termico.

**Qualità dell'aria**: PM10, PM2.5, indice europeo AQI, indice UV attuale.

## Le formule usate (non improvvisate)

| Grandezza | Formula | Fonte |
|---|---|---|
| Heat Index | Regressione Rothfusz a 9 termini | NWS Technical Attachment SR 90-23 (1990) |
| Wind Chill | Formula NWS/Environment Canada | Revisione ufficiale 2001 |
| Dew Point | Approssimazione di Magnus | Alduchov & Eskridge (1996), *J. Appl. Meteor.* |
| Humidex | Formula standard canadese | Environment Canada |
| Nowcasting radar | Lucas-Kanade optical flow + advezione Lagrangiana | Lucas & Kanade (1981); Pulkkinen et al. (2019), pysteps, *Geosci. Model Dev.* |

Tutte le fonti sono citate anche nelle docstring di `weather_calc.py` e
`radar_nowcast.py`.

## ⚠️ Nota importante sui dati

Non esiste un'API REST pubblica, gratuita e stabile per la stazione meteo
fisica di Montesilvano. Il bot **non finge** di leggere una stazione locale:
usa modelli numerici (che assimilano dati stazioni/radar/satelliti a monte)
+ il radar osservato in tempo reale. È dichiarato esplicitamente in ogni
report. C'è un punto di innesto pronto (`fetch_station_placeholder` in
`main.py`) se in futuro configuri una tua stazione (Ecowitt, Netatmo...).

## Messaggio che si aggiorna (edit-in-place)

Invece di mandare un messaggio nuovo ogni 10 minuti, il bot:
1. Al primo avvio invia un messaggio e ne salva l'`message_id` in
   `state/message_id.txt`.
2. Nei cicli successivi chiama `editMessageText` su quello stesso messaggio.
3. Se l'edit fallisce (es. messaggio troppo vecchio, cancellato a mano),
   invia automaticamente un nuovo messaggio e aggiorna l'id salvato.
4. Il workflow GitHub Actions **committa automaticamente** `state/message_id.txt`
   a fine job, così lo stato persiste tra un'esecuzione e l'altra.

Se vuoi forzare un messaggio nuovo (es. dopo una lunga pausa), cancella
semplicemente `state/message_id.txt` dal repo.

## Setup

1. **Crea un bot Telegram**: parla con [@BotFather](https://t.me/BotFather),
   `/newbot`, copia il token.
2. **Trova il tuo `chat_id`**: scrivi un messaggio al bot, poi apri
   `https://api.telegram.org/bot<TOKEN>/getUpdates` e leggi `"chat":{"id": ...}`.
3. **Crea questa repo su GitHub come repo pubblica** (per i minuti Actions gratis illimitati).
4. **Settings → Secrets and variables → Actions**:
   - Tab *Secrets*: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
   - Tab *Variables* (opzionale, hanno default per Montesilvano): `TARGET_LAT`,
     `TARGET_LON`, `TARGET_NAME`, `CUSTOM_STATION_URL`
5. **Settings → Actions → General**: verifica che "Workflow permissions" sia
   su **"Read and write permissions"** (serve per il commit automatico dello stato).
6. Vai su **Actions**, lancia manualmente **"Montesilvano Meteo Watch"** per testare.
7. Da quel momento gira da solo ogni 10 minuti, aggiornando lo stesso messaggio.

## Struttura

```
.github/workflows/meteo.yml       # cron ogni 10 minuti + commit automatico stato
.github/workflows/keepalive.yml   # commit mensile per evitare la disattivazione a 60gg
src/main.py                       # orchestrazione, fetch, costruzione report
src/weather_calc.py               # formule meteorologiche derivate (con fonti)
src/radar_nowcast.py              # nowcasting radar con pysteps (Lucas-Kanade)
src/telegram_client.py            # invio/edit Telegram, gestione message_id
state/message_id.txt              # id del messaggio da modificare (auto-generato)
requirements.txt
```

## Limiti onesti da conoscere

- GitHub non garantisce l'orario esatto: sotto carico i job possono
  ritardare di alcuni minuti.
- Il nowcasting radar usa 4 frame RainViewer (~40 minuti di storico) e
  un'estrapolazione lineare del campo di moto: utile come tendenza a
  brevissimo termine, non un servizio meteorologico certificato.
- Se Telegram o le API esterne sono giù per un ciclo, il job logga l'errore
  e (se possibile) manda comunque un alert Telegram separato.
- `pysteps` richiede `opencv-python-headless` e (su Python 3.12)
  `setuptools<81` per il supporto legacy a `pkg_resources`: già gestito nel
  `requirements.txt`, non serve intervenire.

## Test locale

```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=xxx
export TELEGRAM_CHAT_ID=xxx
cd src && python main.py
```

Senza credenziali Telegram, lo script stampa il report su stdout invece di
inviarlo — utile per debug e per vedere subito la formattazione.
