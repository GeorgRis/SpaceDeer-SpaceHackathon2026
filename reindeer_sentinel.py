from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SHConfig,
    SentinelHubRequest,
    bbox_to_dimensions,
)

CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DEFAULT_AOI_COORDS = [23.50, 69.00, 24.00, 69.20]
DEFAULT_TIME_RANGE = ("2023-09-01", "2023-10-31")
DEFAULT_RESOLUTION_METERS = 60


@dataclass(frozen=True)
class AreaAnalysis:
    bbox_coords: list[float]
    time_range: tuple[str, str]
    image: np.ndarray
    true_color: np.ndarray
    ndvi: np.ndarray
    ndwi: np.ndarray
    suitability: np.ndarray
    summary: dict[str, float]
    movement: dict[str, Any]
    herd: dict[str, Any]


def load_dotenv(dotenv_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file into the environment."""
    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def build_config() -> SHConfig:
    """Build a Copernicus Data Space Ecosystem configuration."""
    config = SHConfig()
    config.sh_client_id = os.getenv("CLIENT_ID") or os.getenv("SH_CLIENT_ID", "")
    config.sh_client_secret = os.getenv("CLIENT_SECRET") or os.getenv("SH_CLIENT_SECRET", "")
    config.sh_base_url = CDSE_BASE_URL
    config.sh_token_url = CDSE_TOKEN_URL

    missing = []
    if not config.sh_client_id:
        missing.append("CLIENT_ID or SH_CLIENT_ID")
    if not config.sh_client_secret:
        missing.append("CLIENT_SECRET or SH_CLIENT_SECRET")

    if missing:
        raise ValueError(
            "Missing Copernicus Data Space credentials. "
            f"Set these environment variables before running: {', '.join(missing)}"
        )

    return config


def build_evalscript() -> str:
    """Fetch true-color bands plus NIR as float32 values for mapping and indices."""
    return """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08"],
      units: "REFLECTANCE"
    }],
    output: {
      bands: 4,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08];
}
"""


def build_bbox_from_center(center_lat: float, center_lon: float, radius_km: float) -> list[float]:
    """Approximate a WGS84 bounding box around a center point."""
    lat_offset = radius_km / 111.0
    lon_scale = max(np.cos(np.radians(center_lat)), 0.1)
    lon_offset = radius_km / (111.0 * lon_scale)
    return [
        round(center_lon - lon_offset, 5),
        round(center_lat - lat_offset, 5),
        round(center_lon + lon_offset, 5),
        round(center_lat + lat_offset, 5),
    ]


def _normalize_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def shift_time_range(time_range: tuple[str, str], days: int) -> tuple[str, str]:
    start = datetime.fromisoformat(_normalize_date(time_range[0])).date()
    end = datetime.fromisoformat(_normalize_date(time_range[1])).date()
    return ((start + timedelta(days=days)).isoformat(), (end + timedelta(days=days)).isoformat())


def build_request(
    config: SHConfig,
    bbox_coords: list[float],
    time_range: tuple[str, str],
    resolution_meters: int = DEFAULT_RESOLUTION_METERS,
) -> SentinelHubRequest:
    """Create a Process API request over the selected area."""
    bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution_meters)
    data_collection = DataCollection.SENTINEL2_L2A.define_from("s2l2a-cdse", service_url=config.sh_base_url)

    return SentinelHubRequest(
        evalscript=build_evalscript(),
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=(_normalize_date(time_range[0]), _normalize_date(time_range[1])),
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )


def fetch_image(request: SentinelHubRequest) -> np.ndarray:
    """Download the processed image as a numpy array."""
    images = request.get_data()
    if not images:
        raise RuntimeError("No Sentinel-2 scenes were returned for the requested area and time range.")
    return images[0]


def calculate_indices(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate NDVI and NDWI from [B02, B03, B04, B08] image bands."""
    b03 = image[:, :, 1].astype(np.float32)
    b04 = image[:, :, 2].astype(np.float32)
    b08 = image[:, :, 3].astype(np.float32)

    eps = 1e-6
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndwi = (b03 - b08) / (b03 + b08 + eps)
    return ndvi, ndwi


def make_true_color(image: np.ndarray) -> np.ndarray:
    """Create a display-friendly RGB image from [B02, B03, B04, B08] reflectance."""
    rgb = image[:, :, [2, 1, 0]].astype(np.float32)
    stretched_channels: list[np.ndarray] = []
    for channel_index in range(3):
        channel = rgb[:, :, channel_index]
        low = float(np.percentile(channel, 2))
        high = float(np.percentile(channel, 98))
        if high - low < 1e-6:
            stretched = np.clip(channel, 0, 1)
        else:
            stretched = np.clip((channel - low) / (high - low), 0, 1)
        stretched_channels.append(stretched)

    stretched_rgb = np.stack(stretched_channels, axis=2)
    gamma = 1 / 1.15
    return np.power(stretched_rgb, gamma)


def calculate_suitability(ndvi: np.ndarray, ndwi: np.ndarray) -> np.ndarray:
    """
    Estimate relative grazing suitability.

    Higher NDVI is generally better, while NDWI is best around mild moisture
    and becomes less favorable when it gets too dry or too wet.
    """
    ndvi_score = np.clip((ndvi + 0.15) / 0.85, 0, 1)
    moisture_penalty = np.clip(np.abs(ndwi - 0.05) / 0.45, 0, 1)
    suitability = 0.7 * ndvi_score + 0.3 * (1 - moisture_penalty)
    return np.clip(suitability, 0, 1)


def summarize_indices(ndvi: np.ndarray, ndwi: np.ndarray, suitability: np.ndarray) -> dict[str, float]:
    favorable_threshold = 0.62
    risky_threshold = 0.35
    favorable_share = float(np.mean(suitability >= favorable_threshold) * 100)
    risky_share = float(np.mean(suitability <= risky_threshold) * 100)

    return {
        "mean_ndvi": float(np.mean(ndvi)),
        "mean_ndwi": float(np.mean(ndwi)),
        "mean_suitability": float(np.mean(suitability)),
        "favorable_share_percent": favorable_share,
        "risky_share_percent": risky_share,
    }


def _weighted_centroid(weights: np.ndarray) -> tuple[float, float]:
    rows, cols = weights.shape
    grid_y, grid_x = np.indices((rows, cols))
    total = float(np.sum(weights))
    if total <= 0:
        return rows / 2.0, cols / 2.0
    return float(np.sum(grid_y * weights) / total), float(np.sum(grid_x * weights) / total)


def _dominant_zone_centroid(weights: np.ndarray) -> tuple[tuple[float, float], float]:
    """
    Find the centroid of the strongest suitability cells instead of the full map average.

    This avoids collapsing the estimated herd position into the center of the image
    when the area is broadly uniform.
    """
    rows, cols = weights.shape
    threshold = float(np.percentile(weights, 92))
    hotspot = np.where(weights >= threshold, weights, 0.0)
    hotspot_total = float(np.sum(hotspot))
    total = float(np.sum(weights))
    if hotspot_total <= 0 or total <= 0:
        return (rows / 2.0, cols / 2.0), 0.0

    centroid = _weighted_centroid(hotspot)
    concentration = float(np.clip(hotspot_total / total, 0, 1))
    return centroid, concentration


def _clip_point(x: float, y: float, cols: int, rows: int) -> tuple[float, float]:
    clipped_x = float(np.clip(x, 0, max(cols - 1, 0)))
    clipped_y = float(np.clip(y, 0, max(rows - 1, 0)))
    return clipped_x, clipped_y


def _movement_direction(dx: float, dy: float) -> str:
    horizontal = "east" if dx > 0 else "west"
    vertical = "south" if dy > 0 else "north"
    if abs(dx) < 0.08 and abs(dy) < 0.08:
        return "stable"
    if abs(dx) > abs(dy) * 1.3:
        return horizontal
    if abs(dy) > abs(dx) * 1.3:
        return vertical
    return f"{vertical}-{horizontal}"


def predict_movement(
    current_suitability: np.ndarray,
    previous_suitability: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Predict relative movement direction based on where conditions improve.

    This is a heuristic indicator, not a biologically validated migration model.
    """
    current_centroid, current_concentration = _dominant_zone_centroid(current_suitability)
    rows, cols = current_suitability.shape
    center_y = rows / 2.0
    center_x = cols / 2.0

    if previous_suitability is not None:
        previous_centroid, previous_concentration = _dominant_zone_centroid(previous_suitability)
        dy = current_centroid[0] - previous_centroid[0]
        dx = current_centroid[1] - previous_centroid[1]
    else:
        previous_centroid = (center_y, center_x)
        previous_concentration = 0.0
        dy = current_centroid[0] - center_y
        dx = current_centroid[1] - center_x

    normalized_dx = dx / max(cols, 1)
    normalized_dy = dy / max(rows, 1)
    vector_strength = float(np.hypot(normalized_dx, normalized_dy))
    direction = _movement_direction(normalized_dx, normalized_dy)

    confidence = float(np.clip(vector_strength * 9.0 + current_concentration * 0.6, 0, 1))
    if current_concentration < 0.12:
        direction = "diffuse"
        confidence = min(confidence, 0.3)
    elif direction == "stable":
        confidence = min(confidence, 0.35)

    future_x, future_y = _clip_point(
        current_centroid[1] + dx * 1.35,
        current_centroid[0] + dy * 1.35,
        cols,
        rows,
    )

    return {
        "direction": direction,
        "confidence": confidence,
        "vector": {"dx": float(normalized_dx), "dy": float(normalized_dy)},
        "previous_centroid": {"x": float(previous_centroid[1]), "y": float(previous_centroid[0])},
        "centroid": {"x": float(current_centroid[1]), "y": float(current_centroid[0])},
        "future_centroid": {"x": future_x, "y": future_y},
        "concentration": current_concentration,
        "previous_concentration": previous_concentration,
    }


def pixel_to_geo(bbox_coords: list[float], cols: int, rows: int, x: float, y: float) -> dict[str, float]:
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    lon = min_lon + (x / max(cols - 1, 1)) * (max_lon - min_lon)
    lat = max_lat - (y / max(rows - 1, 1)) * (max_lat - min_lat)
    return {"lat": float(lat), "lon": float(lon)}


def estimate_herd_size(summary: dict[str, float], bbox_coords: list[float]) -> dict[str, Any]:
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    center_lat = (min_lat + max_lat) / 2
    width_km = (max_lon - min_lon) * 111.0 * max(np.cos(np.radians(center_lat)), 0.1)
    height_km = (max_lat - min_lat) * 111.0
    area_km2 = max(width_km * height_km, 1.0)
    suitable_area_km2 = area_km2 * (summary["favorable_share_percent"] / 100.0)
    estimated_count = int(np.clip(round(suitable_area_km2 * 1.6), 25, 1800))

    if estimated_count < 120:
        band = "small herd"
    elif estimated_count < 400:
        band = "medium herd"
    else:
        band = "large herd"

    return {
        "estimated_count": estimated_count,
        "band": band,
        "assumption": "Estimated from favorable grazing area and a simple density assumption, not direct observation.",
        "suitable_area_km2": float(suitable_area_km2),
        "area_km2": float(area_km2),
    }


def build_herd_trace(
    movement: dict[str, Any],
    bbox_coords: list[float],
    rows: int,
    cols: int,
) -> dict[str, Any]:
    previous_pixel = movement["previous_centroid"]
    current_pixel = movement["centroid"]
    future_pixel = movement["future_centroid"]

    return {
        "previous": {
            "pixel": previous_pixel,
            "geo": pixel_to_geo(bbox_coords, cols, rows, previous_pixel["x"], previous_pixel["y"]),
        },
        "current": {
            "pixel": current_pixel,
            "geo": pixel_to_geo(bbox_coords, cols, rows, current_pixel["x"], current_pixel["y"]),
        },
        "future": {
            "pixel": future_pixel,
            "geo": pixel_to_geo(bbox_coords, cols, rows, future_pixel["x"], future_pixel["y"]),
        },
    }


def build_recommendation(summary: dict[str, float], movement: dict[str, Any]) -> str:
    direction = movement["direction"].replace("-", " ")
    if direction == "diffuse":
        return "The signal is too spread out to place the herd confidently in one core zone. Try a smaller area or a shorter time window."
    if direction == "stable":
        return "Keep monitoring the current area. The data does not show a strong directional pull toward a new sector right now."
    if summary["favorable_share_percent"] >= 45:
        return f"Prioritize patrols and planning toward the {direction} part of the selected area where grazing conditions look strongest."
    if summary["risky_share_percent"] >= 40:
        return f"Expect animals to avoid the weakest terrain and drift toward the {direction} side where conditions look relatively better."
    return f"Conditions are mixed, but the best short-term signal still points toward the {direction} sector."


def analyze_area(
    config: SHConfig,
    bbox_coords: list[float],
    time_range: tuple[str, str],
    resolution_meters: int = DEFAULT_RESOLUTION_METERS,
    previous_time_range: tuple[str, str] | None = None,
) -> AreaAnalysis:
    request = build_request(config, bbox_coords, time_range, resolution_meters)
    image = fetch_image(request)
    ndvi, ndwi = calculate_indices(image)
    suitability = calculate_suitability(ndvi, ndwi)

    previous_suitability = None
    if previous_time_range is not None:
        previous_request = build_request(config, bbox_coords, previous_time_range, resolution_meters)
        previous_image = fetch_image(previous_request)
        previous_ndvi, previous_ndwi = calculate_indices(previous_image)
        previous_suitability = calculate_suitability(previous_ndvi, previous_ndwi)

    true_color = make_true_color(image)
    summary = summarize_indices(ndvi, ndwi, suitability)
    movement = predict_movement(suitability, previous_suitability)
    rows, cols = suitability.shape
    herd = estimate_herd_size(summary, bbox_coords)
    movement["target_geo"] = pixel_to_geo(
        bbox_coords,
        cols,
        rows,
        movement["future_centroid"]["x"],
        movement["future_centroid"]["y"],
    )
    movement["recommendation"] = build_recommendation(summary, movement)
    herd["trace"] = build_herd_trace(movement, bbox_coords, rows, cols)

    return AreaAnalysis(
        bbox_coords=bbox_coords,
        time_range=(_normalize_date(time_range[0]), _normalize_date(time_range[1])),
        image=image,
        true_color=true_color,
        ndvi=ndvi,
        ndwi=ndwi,
        suitability=suitability,
        summary=summary,
        movement=movement,
        herd=herd,
    )


def movement_label(movement: dict[str, Any]) -> str:
    direction = movement["direction"]
    confidence = movement["confidence"]

    direction_labels = {
        "diffuse": "Conditions are too evenly spread to identify one clear herd core in the selected area.",
        "stable": "Conditions look relatively stable within the selected area.",
        "north": "Conditions appear to improve toward the north.",
        "south": "Conditions appear to improve toward the south.",
        "east": "Conditions appear to improve toward the east.",
        "west": "Conditions appear to improve toward the west.",
        "north-east": "Conditions appear to improve toward the north-east.",
        "north-west": "Conditions appear to improve toward the north-west.",
        "south-east": "Conditions appear to improve toward the south-east.",
        "south-west": "Conditions appear to improve toward the south-west.",
    }
    confidence_label = "low" if confidence < 0.35 else "medium" if confidence < 0.7 else "high"
    return f"{direction_labels.get(direction, 'Conditions show a weak directional signal.')} Heuristic confidence: {confidence_label}."


def plot_prediction_map(analysis: AreaAnalysis) -> plt.Figure:
    rows, cols = analysis.suitability.shape
    previous = analysis.herd["trace"]["previous"]["pixel"]
    current = analysis.herd["trace"]["current"]["pixel"]
    future = analysis.herd["trace"]["future"]["pixel"]
    direction = analysis.movement["direction"]
    confidence = analysis.movement["confidence"]

    hotspot_threshold = float(np.percentile(analysis.suitability, 92))
    hotspot_mask = np.where(analysis.suitability >= hotspot_threshold, analysis.suitability, np.nan)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(analysis.true_color)
    ax.imshow(hotspot_mask, cmap="autumn_r", alpha=0.38, vmin=hotspot_threshold, vmax=1)
    ax.plot(
        [previous["x"], current["x"]],
        [previous["y"], current["y"]],
        color="#264653",
        linewidth=2.6,
        linestyle="--",
        alpha=0.9,
    )
    ax.scatter(previous["x"], previous["y"], s=85, c="#2a9d8f", edgecolors="white", linewidths=1.2, label="Past position")
    ax.scatter(current["x"], current["y"], s=115, c="#16324f", edgecolors="white", linewidths=1.4, label="Current position")

    if direction in {"stable", "diffuse"} or confidence < 0.18:
        stable_radius = max(min(rows, cols) * 0.06, 4)
        stable_ring = plt.Circle(
            (current["x"], current["y"]),
            stable_radius,
            color="#16324f",
            fill=False,
            linewidth=2.5,
            alpha=0.95,
        )
        ax.add_patch(stable_ring)
    else:
        ax.scatter(future["x"], future["y"], s=135, c="#c0392b", edgecolors="white", linewidths=1.2, label="Predicted next position")
        ax.annotate(
            "",
            xy=(future["x"], future["y"]),
            xytext=(current["x"], current["y"]),
            arrowprops={"arrowstyle": "->", "lw": 3, "color": "#101820"},
        )
        ax.plot(
            [current["x"], future["x"]],
            [current["y"], future["y"]],
            color="#c0392b",
            linewidth=2.8,
            alpha=0.9,
        )

    ax.text(
        0.02,
        0.02,
        "Diffuse signal" if direction == "diffuse" else "No clear movement" if direction == "stable" else analysis.movement["direction"].replace("-", " ").title(),
        transform=ax.transAxes,
        fontsize=12,
        color="white",
        bbox={"facecolor": "#101820", "alpha": 0.8, "boxstyle": "round,pad=0.35"},
    )
    ax.text(
        0.02,
        0.93,
        f"Estimated herd: ~{analysis.herd['estimated_count']} ({analysis.herd['band']})",
        transform=ax.transAxes,
        fontsize=11,
        color="white",
        bbox={"facecolor": "#7f5539", "alpha": 0.84, "boxstyle": "round,pad=0.3"},
    )
    ax.set_title("Satellite view with predicted movement signal")
    ax.axis("off")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def plot_analysis(analysis: AreaAnalysis) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    rgb_plot = axes[0].imshow(analysis.true_color)
    axes[0].set_title("Satellite")
    axes[0].axis("off")

    ndvi_plot = axes[1].imshow(analysis.ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("NDVI")
    axes[1].axis("off")
    plt.colorbar(ndvi_plot, ax=axes[1], fraction=0.046, pad=0.04)

    ndwi_plot = axes[2].imshow(analysis.ndwi, cmap="Blues", vmin=-1, vmax=1)
    axes[2].set_title("NDWI")
    axes[2].axis("off")
    plt.colorbar(ndwi_plot, ax=axes[2], fraction=0.046, pad=0.04)

    suitability_plot = axes[3].imshow(analysis.suitability, cmap="YlGn", vmin=0, vmax=1)
    axes[3].set_title("Relative Grazing Suitability")
    axes[3].axis("off")
    plt.colorbar(suitability_plot, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Area {analysis.bbox_coords} | {analysis.time_range[0]} to {analysis.time_range[1]}",
        fontsize=12,
    )
    plt.tight_layout()
    return fig


def main() -> None:
    load_dotenv()
    config = build_config()
    analysis = analyze_area(
        config=config,
        bbox_coords=DEFAULT_AOI_COORDS,
        time_range=DEFAULT_TIME_RANGE,
        resolution_meters=DEFAULT_RESOLUTION_METERS,
        previous_time_range=shift_time_range(DEFAULT_TIME_RANGE, -60),
    )
    fig = plot_analysis(analysis)
    print(movement_label(analysis.movement))
    plt.show(block=True)


if __name__ == "__main__":
    main()
