from __future__ import annotations

from datetime import timedelta

import requests
import streamlit as st

from reindeer_sentinel import (
    DEFAULT_RESOLUTION_METERS,
    analyze_area,
    build_bbox_from_center,
    build_config,
    load_dotenv,
    movement_label,
    plot_analysis,
    plot_prediction_map,
)

DEFAULT_PLACE = {
    "name": "Kautokeino, Finnmark, Norway",
    "lat": 69.0119,
    "lon": 23.0412,
}


@st.cache_data(show_spinner=False)
def geocode_place(query: str) -> dict[str, float | str]:
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": "ReindeerSentinel/0.1"},
        timeout=20,
    )
    response.raise_for_status()
    matches = response.json()
    if not matches:
        raise ValueError(f"No locations matched '{query}'.")

    match = matches[0]
    return {
        "name": str(match["display_name"]),
        "lat": float(match["lat"]),
        "lon": float(match["lon"]),
    }


def initialize_state() -> None:
    st.session_state.setdefault("place_name", DEFAULT_PLACE["name"])
    st.session_state.setdefault("center_lat", DEFAULT_PLACE["lat"])
    st.session_state.setdefault("center_lon", DEFAULT_PLACE["lon"])
    st.session_state.setdefault("search_query", "Kautokeino")


def main() -> None:
    st.set_page_config(page_title="Reindeer Sentinel", page_icon="R", layout="wide")
    load_dotenv()
    initialize_state()

    st.title("Reindeer Sentinel")
    st.caption("Search an area, view the satellite scene, and see where grazing conditions suggest reindeer may drift next.")

    with st.sidebar:
        st.subheader("Area Search")
        search_query = st.text_input("Place or area", key="search_query")
        if st.button("Find area", use_container_width=True):
            with st.spinner("Searching for area..."):
                location = geocode_place(search_query)
            st.session_state["place_name"] = location["name"]
            st.session_state["center_lat"] = location["lat"]
            st.session_state["center_lon"] = location["lon"]

        st.caption(st.session_state["place_name"])
        center_lat = st.number_input("Center latitude", value=float(st.session_state["center_lat"]), format="%.5f")
        center_lon = st.number_input("Center longitude", value=float(st.session_state["center_lon"]), format="%.5f")
        radius_km = st.slider("Search radius (km)", min_value=5, max_value=80, value=20, step=5)
        resolution = st.slider("Resolution (m)", min_value=20, max_value=120, value=DEFAULT_RESOLUTION_METERS, step=10)

        st.subheader("Time Window")
        default_end = st.date_input("End date")
        default_start = default_end - timedelta(days=30)
        start_date = st.date_input("Start date", value=default_start)
        compare_previous = st.checkbox("Compare with previous period", value=True)

        run_analysis = st.button("Analyze area", type="primary", use_container_width=True)

    bbox_coords = build_bbox_from_center(center_lat, center_lon, radius_km)

    intro_cols = st.columns((1.4, 1, 1))
    intro_cols[0].markdown(
        f"""
        **Selected area:** {st.session_state["place_name"]}  
        **BBox:** `{bbox_coords}`  
        **Center:** `{center_lat:.5f}, {center_lon:.5f}`
        """
    )
    intro_cols[1].info("You get a satellite view, a predicted direction, and a plain-language recommendation.")
    intro_cols[2].warning("Prediction is heuristic. It shows the strongest environmental signal, not guaranteed herd behavior.")

    if run_analysis:
        if start_date >= default_end:
            st.error("Start date must be earlier than end date.")
            st.stop()

        try:
            config = build_config()
        except ValueError as exc:
            st.error(str(exc))
            st.info("Add credentials in `.env` or environment variables before running the app.")
            st.stop()

        current_range = (start_date.isoformat(), default_end.isoformat())
        previous_range = None
        if compare_previous:
            days = (default_end - start_date).days + 1
            previous_range = (
                (start_date - timedelta(days=days)).isoformat(),
                (default_end - timedelta(days=days)).isoformat(),
            )

        try:
            with st.spinner("Fetching Sentinel-2 data and calculating grazing indicators..."):
                analysis = analyze_area(
                    config=config,
                    bbox_coords=bbox_coords,
                    time_range=current_range,
                    resolution_meters=resolution,
                    previous_time_range=previous_range,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis failed: {exc}")
            st.stop()

        movement = analysis.movement
        target_label = (
            "Current area"
            if movement["direction"] == "stable"
            else f"{movement['target_geo']['lat']:.4f}, {movement['target_geo']['lon']:.4f}"
        )
        summary_cols = st.columns(3)
        summary_cols[0].metric("Likely direction", movement["direction"].replace("-", " ").title())
        summary_cols[1].metric("Confidence", f"{movement['confidence']:.0%}")
        summary_cols[2].metric("Target point", target_label)

        st.subheader("What this means")
        st.success(movement["recommendation"])
        st.write(movement_label(movement))
        st.progress(int(movement["confidence"] * 100), text=f"Confidence in this signal: {movement['confidence']:.0%}")

        prediction_cols = st.columns((1.5, 1))
        with prediction_cols[0]:
            prediction_figure = plot_prediction_map(analysis)
            st.pyplot(prediction_figure, clear_figure=True, use_container_width=True)
        with prediction_cols[1]:
            st.markdown("**How to read the map**")
            st.write("The background is a normal Sentinel-2 satellite image.")
            st.write("The warm overlay highlights only the strongest grazing-condition pockets, not the whole map.")
            if movement["direction"] == "stable":
                st.write("The ring around the center means the model does not see a strong directional shift right now.")
                st.write("Suggested next focus: keep monitoring the current area rather than moving to a new sector.")
            else:
                st.write("The arrow points from the selected area center toward the strongest current signal.")
                st.write(
                    f"Suggested next focus: **{movement['direction'].replace('-', ' ').title()}** sector around "
                    f"`{movement['target_geo']['lat']:.4f}, {movement['target_geo']['lon']:.4f}`."
                )

            metric_cols = st.columns(2)
            metric_cols[0].metric("Favorable terrain", f"{analysis.summary['favorable_share_percent']:.1f}%")
            metric_cols[1].metric("Risk terrain", f"{analysis.summary['risky_share_percent']:.1f}%")
            metric_cols = st.columns(2)
            metric_cols[0].metric("Mean NDVI", f"{analysis.summary['mean_ndvi']:.2f}")
            metric_cols[1].metric("Mean NDWI", f"{analysis.summary['mean_ndwi']:.2f}")

        with st.expander("See technical layers"):
            figure = plot_analysis(analysis)
            st.pyplot(figure, clear_figure=True, use_container_width=True)

        st.subheader("Operational interpretation")
        if analysis.summary["favorable_share_percent"] >= 45:
            st.success("This area has a solid share of potentially good grazing terrain, so the prediction is worth checking in the field.")
        elif analysis.summary["risky_share_percent"] >= 40:
            st.warning("Much of the area looks weaker, so animals may shift away from poor terrain and concentrate in a smaller zone.")
        else:
            st.info("Conditions are mixed. Try a tighter radius or another time window to sharpen the signal.")
    else:
        st.info("Pick an area, adjust the time window, and click Analyze area.")


if __name__ == "__main__":
    main()
