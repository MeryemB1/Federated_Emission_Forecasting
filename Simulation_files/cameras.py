import sumolib
import sys
import csv

try:
    import pyproj
except ImportError:
    print("Please install pyproj using 'pip install pyproj'")
    sys.exit(1)

# Load the SUMO network
net = sumolib.net.readNet("manhattan.net.xml")

# Read the edge_centers.csv file
edge_center_file = "edge_centers.csv"
edge_centers = []
with open(edge_center_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')  # Assuming comma-separated
    for row in reader:
        edge_centers.append({
            "edge": row["edge"],
            "lon": float(row["longitude"]),
            "lat": float(row["latitude"])
        })

# Start building the XML string for the additional file
camera_definitions = '<additional>\n'
poi_definitions = ''

for item in edge_centers:
    edge_id = item["edge"]
    lat = item["lat"]
    lon = item["lon"]

    # Convert lat/lon to SUMO (x, y)
    x, y = net.convertLonLat2XY(lon, lat)

    # Find all lanes within a 100 m radius of the location
    lanes = net.getNeighboringLanes(x, y, 100)
    if not lanes:
        print(f"⚠️ No nearby lanes for edge {edge_id} at ({lat}, {lon})")
        continue

    # Get the closest lane (by distance) and retrieve its corresponding edge
    closest_lane = sorted(lanes, key=lambda l: l[1])[0][0]
    edge = closest_lane.getEdge()

    # Skip internal edges
    if edge.getID().startswith(":"):
        print(f"⚠️ Skipping internal edge {edge.getID()} for location {edge_id}")
        continue

    # ✅ Place laneAreaDetector on every lane of the selected edge
    for lane in edge.getLanes():
        lane_id = lane.getID()
        lane_length = lane.getLength()
        detector_id = f"{edge_id}_{lane_id}"
        camera_definitions += (
            f'    <laneAreaDetector id="{detector_id}" lane="{lane_id}" '
            f'pos="0.00" endPos="{lane_length:.2f}" freq="7" '
            f'file="camera_{edge_id}_output.xml"/>\n'
        )

    # ✅ Try to find reverse edge (opposite direction)
    from_node = edge.getFromNode()
    to_node = edge.getToNode()
    reverse_edge = None

    for candidate_edge in to_node.getOutgoing():
        if candidate_edge.getToNode() == from_node and not candidate_edge.getID().startswith(":"):
            reverse_edge = candidate_edge
            break

    # Add detectors to reverse edge (if it exists)
    if reverse_edge:
        for lane in reverse_edge.getLanes():
            lane_id = lane.getID()
            lane_length = lane.getLength()
            reverse_detector_id = f"{edge_id}_reverse_{lane_id}"
            camera_definitions += (
                f'    <laneAreaDetector id="{reverse_detector_id}" lane="{lane_id}" '
                f'pos="0.00" endPos="{lane_length:.2f}" freq="7" '
                f'file="camera_{edge_id}_reverse_output.xml"/>\n'
            )

    # POI marker at the center
    poi_definitions += (
        f'    <poi id="poi_{edge_id}" color="255,0,0" layer="202" '
        f'x="{x:.2f}" y="{y:.2f}" width="2" height="2" type="camera"/>\n'
    )

# Finalize and write the XML file
camera_definitions += poi_definitions
camera_definitions += '</additional>\n'

with open("cameras.add.xml", "w") as f:
    f.write(camera_definitions)

print("✅ Generated cameras.add.xml with detectors on both directions (if available)")
