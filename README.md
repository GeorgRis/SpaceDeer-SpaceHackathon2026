# Reindeer Sentinel

Prototype for monitoring reindeer grazing conditions in Northern Norway with Sentinel-2, now with an interactive frontend.

## What it does

- fetches Sentinel-2 band data for a selected grazing area
- calculates `NDVI` for vegetation / lichen health
- calculates `NDWI` for moisture / possible ice crust conditions
- derives a relative grazing suitability layer
- estimates likely reindeer movement direction as a heuristic signal
- shows the prediction on top of a normal Sentinel-2 satellite image
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
- Movement prediction is a heuristic based on vegetation and moisture signals, not a trained ecological migration model.
