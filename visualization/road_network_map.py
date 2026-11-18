
import pandas as pd
import osmnx as ox
import folium


def create_road_network_map(csv_file):

    # 1. Load GPS dataset
    gps_data = pd.read_csv(csv_file)

    print("GPS dataset loaded successfully!")

    # 2. Validate GPS columns
    required_columns = ["latitude", "longitude"]

    for column in required_columns:
        if column not in gps_data.columns:
            raise ValueError(f"Missing column: {column}")

    # 3. Clean GPS data
    gps_data = gps_data.dropna(
        subset=["latitude", "longitude"]
    )

    gps_data = gps_data[
        gps_data["latitude"].between(-90, 90)
        & gps_data["longitude"].between(-180, 180)
    ]

    if gps_data.empty:
        raise ValueError("No valid GPS coordinates found.")

    # 4. Calculate geographical center
    center_lat = gps_data["latitude"].mean()
    center_lon = gps_data["longitude"].mean()

    # 5. Download actual road network
    print("Downloading road network from OpenStreetMap...")

    graph = ox.graph_from_point(
        (center_lat, center_lon),
        dist=1000,
        network_type="drive",
        simplify=True
    )

    nodes, edges = ox.graph_to_gdfs(graph)

    print("Road network downloaded successfully!")
    print("Total road segments:", len(edges))

    # 6. Create interactive map
    road_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=16,
        tiles=None
    )

    # 7. Add Esri street map
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Street Map",
        overlay=False,
        control=True
    ).add_to(road_map)

    # 8. Create separate interactive layers
    road_layer = folium.FeatureGroup(
        name="Road Network",
        show=True
    )

    gps_layer = folium.FeatureGroup(
        name="GPS Points",
        show=True
    )

    trajectory_layer = folium.FeatureGroup(
        name="GPS Trajectory",
        show=True
    )

    endpoint_layer = folium.FeatureGroup(
        name="Start and End Locations",
        show=True
    )

    # 9. Display actual road geometries
    folium.GeoJson(
        edges.to_json(),
        name="Road Segments",
        style_function=lambda feature: {
            "color": "blue",
            "weight": 2,
            "opacity": 0.8
        }
    ).add_to(road_layer)

    road_layer.add_to(road_map)

    # 10. Extract GPS coordinates
    coordinates = gps_data[
        ["latitude", "longitude"]
    ].values.tolist()

    # 11. Display GPS points
    for index, point in enumerate(coordinates):

        folium.CircleMarker(
            location=point,
            radius=5,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.9,
            popup=f"GPS Point {index + 1}"
        ).add_to(gps_layer)

    gps_layer.add_to(road_map)

    # 12. Display GPS trajectory
    folium.PolyLine(
        coordinates,
        color="red",
        weight=4,
        opacity=0.9,
        tooltip="Original GPS Trajectory"
    ).add_to(trajectory_layer)

    trajectory_layer.add_to(road_map)

    # 13. Display start and end locations
    folium.Marker(
        coordinates[0],
        popup="Starting Location",
        tooltip="Start",
        icon=folium.Icon(color="green")
    ).add_to(endpoint_layer)

    folium.Marker(
        coordinates[-1],
        popup="Ending Location",
        tooltip="End",
        icon=folium.Icon(color="red")
    ).add_to(endpoint_layer)

    endpoint_layer.add_to(road_map)

    # 14. Add interactive layer selector
    folium.LayerControl(
        collapsed=False
    ).add_to(road_map)

    # 15. Add map legend
    legend_html = """
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        width: 190px;
        background-color: white;
        border: 2px solid grey;
        border-radius: 8px;
        z-index: 9999;
        padding: 12px;
        font-size: 13px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        <b>PathSnap Map Legend</b>
        <hr>
        <span style="color: blue;">━━</span>
        Road Network<br><br>

        <span style="color: red;">━━</span>
        GPS Trajectory<br><br>

        <span style="color: red;">●</span>
        GPS Points<br><br>

        <span style="color: green;">●</span>
        Start Location<br><br>

        <span style="color: red;">●</span>
        End Location
    </div>
    """

    road_map.get_root().html.add_child(
        folium.Element(legend_html)
    )

    # 16. Fit map to GPS coordinates
    road_map.fit_bounds(coordinates)

    # 17. Save interactive map
    output_file = "road_network_visualization.html"

    road_map.save(output_file)

    print("\nPATHSNAP MAP CREATED SUCCESSFULLY!")
    print("Road segments:", len(edges))
    print("GPS points:", len(coordinates))
    print("Interactive layers: 4")
    print("Saved as:", output_file)


if __name__ == "__main__":

    create_road_network_map(
        "visualization/sample_gps.csv"
    )
