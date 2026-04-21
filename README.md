# Reindeer Sentinel

Prototype for monitoring reindeer grazing conditions in Northern Norway with Sentinel-2, now with an interactive frontend.

## What it does

- fetches Sentinel-2 band data for a selected grazing area
- calculates `NDVI` for vegetation / lichen health
- calculates `NDWI` for moisture / possible ice crust conditions
- derives a relative grazing suitability layer
- estimates an approximate herd position, recent path, and predicted next move as heuristic signals
- shows the herd trace on top of a normal Sentinel-2 satellite image
- includes a rough herd-size assumption based on favorable grazing area
- provides both a script view and a small Streamlit app for area search

## Files

- `reindeer_sentinel.py` - core analysis functions and script entrypoint
- `streamlit_app.py` - interactive frontend for searching areas and running analysis
- `requirements.txt` - Python dependencies

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set credentials as environment variables:

```bash
export CLIENT_ID="your-client-id"
export CLIENT_SECRET="your-client-secret"
```

On PowerShell:

```powershell
$env:CLIENT_ID="your-client-id"
$env:CLIENT_SECRET="your-client-secret"
```

You can also place them in a local `.env` file:

```env
CLIENT_ID=your-client-id
CLIENT_SECRET=your-client-secret
```

## Run

Script version:

```bash
python reindeer_sentinel.py
```

Interactive frontend:

```bash
streamlit run streamlit_app.py
```

## Notes

- The app uses `SentinelHubRequest` with the Copernicus Data Space Ecosystem Process API.
- You do not need `INSTANCE_ID` for this version.
- You do need a free CDSE account and OAuth client credentials.
- Area search uses OpenStreetMap Nominatim geocoding.
- Herd position, path, and herd-size estimates are heuristics based on vegetation and moisture signals, not direct tracking or counted animals.
