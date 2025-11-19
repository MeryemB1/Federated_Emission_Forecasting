import pandas as pd
import sumolib
import networkx as nx
import math
import random
from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET

# -------------------------------
# 1. Load Data
# -------------------------------
centroids_df = pd.read_csv('region_centroids.csv')  
trip_df = pd.read_csv("./group6.csv", delimiter=",")  # Adjust delimiter if needed
net = sumolib.net.readNet("manhattan.net.xml")

# -------------------------------
# 2. Parse Connections
# -------------------------------
connections = {}
tree = ET.parse("manhattan.net.xml")
root = tree.getroot()
for connection in root.findall(".//connection"):
    from_edge = connection.attrib['from']
    to_edge = connection.attrib['to']
    from_lane = connection.attrib['fromLane']
    to_lane = connection.attrib['toLane']
    # Include all valid connections regardless of 'state'
    if from_edge not in connections:
        connections[from_edge] = []
    connections[from_edge].append((to_edge, from_lane, to_lane))

# -------------------------------
# 3. Helpers
# -------------------------------
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def is_vehicle_road(edge):
    edge_type = edge.getType()
    return edge_type is None or edge_type not in ['railway.subway', 'highway.footway', 'railway.rail']

def is_turn_allowed(from_edge, to_edge):
    if from_edge in connections:
        for valid_connection in connections[from_edge]:
            if valid_connection[0] == to_edge:
                return True
    return False

def find_candidate_edges(x, y, net, max_candidates=1):
    candidates = []
    for edge in net.getEdges():
        if not is_vehicle_road(edge):
            continue
        edge_geom = edge.getShape()
        min_dist = float('inf')
        for i in range(len(edge_geom) - 1):
            x1, y1 = edge_geom[i]
            x2, y2 = edge_geom[i + 1]
            dx, dy = x2 - x1, y2 - y1
            line_length = math.sqrt(dx**2 + dy**2)
            if line_length == 0:
                continue
            t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (line_length**2)))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            d = calculate_distance(x, y, closest_x, closest_y)
            if d < min_dist:
                min_dist = d
        candidates.append((edge, min_dist))
    candidates.sort(key=lambda item: item[1])
    return [edge for edge, dist in candidates[:max_candidates]]

def time_range_to_seconds(time_range_label):
    start_str, end_str = time_range_label.split(" - ")
    start_hour = int(start_str.split(":")[0])
    end_hour = int(end_str.split(":")[0])
    return start_hour * 3600, end_hour * 3600

# -------------------------------
# 4. Map Regions to Edges
# -------------------------------
region_to_edge_id = {}
for _, row in centroids_df.iterrows():
    region = row["region_name"]
    lon, lat = row["centroid_lon"], row["centroid_lat"]
    x, y = net.convertLonLat2XY(lon, lat)
    candidates = find_candidate_edges(x, y, net, max_candidates=1)
    if candidates:
        region_to_edge_id[region] = candidates[0].getID()
        print(f"edge for region {region}: {region_to_edge_id[region]}")
    else:
        print(f"❌ No edge found for region {region}")

# -------------------------------
# 5. Build SUMO Graph
# -------------------------------
G = nx.DiGraph()
for edge in net.getEdges():
    edge_id = edge.getID()
    G.add_node(edge_id)
    to_node = edge.getToNode()
    for out_edge in to_node.getOutgoing():
        out_id = out_edge.getID()
        G.add_edge(edge_id, out_id, weight=out_edge.getLength())

# Initialize trip counters
total_trip_count = 0
valid_trip_count = 0

# -------------------------------
# 6. Compute Routes (Multiple Date and Time Ranges)
# -------------------------------
routes = []

# Define time ranges (label, start_hour, end_hour)
time_ranges = [
    ("00:00 - 01:00", 0, 1),
    ("01:00 - 02:00", 1, 2),
    ("02:00 - 03:00", 2, 3),
    ("03:00 - 04:00", 3, 4),
    ("04:00 - 05:00", 4, 5),
    ("05:00 - 06:00", 5, 6),
    ("06:00 - 07:00", 6, 7),
    ("07:00 - 08:00", 7, 8),
    ("08:00 - 09:00", 8, 9),
    ("09:00 - 10:00", 9, 10),
    ("10:00 - 11:00", 10, 11),
    ("11:00 - 12:00", 11, 12),
    ("12:00 - 13:00", 12, 13),
    ("13:00 - 14:00", 13, 14),
    ("14:00 - 15:00", 14, 15),
    ("15:00 - 16:00", 15, 16),
    ("16:00 - 17:00", 16, 17),
    ("17:00 - 18:00", 17, 18),
    ("18:00 - 19:00", 18, 19),
    ("19:00 - 20:00", 19, 20),
    ("20:00 - 21:00", 20, 21),
    ("21:00 - 22:00", 21, 22),
    ("22:00 - 23:00", 22, 23),
    ("23:00 - 00:00", 23, 00),
]

# Define date ranges (as strings)
date_ranges = [
    "2024-08-21 - 2024-08-22",
    "2024-08-22 - 2024-08-23",
    "2024-08-23 - 2024-08-24",
    "2024-08-24 - 2024-08-25",
      
]

for date_range in date_ranges:
    for time_range_label, start_hour, end_hour in time_ranges:
        trips_col = f"Date range: {date_range} Time range: {time_range_label} Trips"
        for _, row in trip_df.iterrows():
            origin_region = row["Origin"]
            dest_region = row["Destination"]

            if trips_col not in row:
                print(f"⚠️ Column {trips_col} not found for OD pair {origin_region} → {dest_region}")
                continue

            trips = row[trips_col]
            if trips <= 0:
                continue  # Skip if no trips

            total_trip_count += trips

            if origin_region not in region_to_edge_id or dest_region not in region_to_edge_id:
                print(f"⚠️ Skipping OD pair {origin_region} → {dest_region} (missing edge mapping)")
                continue

            origin_edge = region_to_edge_id[origin_region]
            dest_edge = region_to_edge_id[dest_region]

            try:
                path = nx.shortest_path(G, source=origin_edge, target=dest_edge, weight="weight")
                turn_valid = True
                for i in range(len(path) - 1):
                    if not is_turn_allowed(path[i], path[i + 1]):
                        print(f"❌ Invalid turn from {path[i]} to {path[i + 1]}")
                        turn_valid = False
                        break

                if turn_valid:
                    total_len = 0
                    for u, v in zip(path, path[1:]):
                        try:
                            total_len += G[u][v]['weight']
                        except KeyError:
                            print(f"⚠️ Missing edge from {u} to {v} — skipping route.")
                            turn_valid = False
                            break

                if turn_valid:
                    routes.append({
                        "origin_region": origin_region,
                        "destination_region": dest_region,
                        "origin_edge": origin_edge,
                        "destination_edge": dest_edge,
                        "path": path,
                        "length_m": total_len,
                        "trip_count": trips,
                        "time_range": time_range_label,
                        "date_range": date_range
                    })
                    valid_trip_count += trips
                    print(f"✅ Route: {origin_region} → {dest_region} | Trips: {trips} | Len: {round(total_len)}m | Time: {time_range_label} | Date: {date_range}")
                else:
                    print(f"⚠️ Skipped route due to invalid path: {origin_region} → {dest_region}")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(f"❌ No route found: {origin_region} → {dest_region}")

if total_trip_count > 0:
    proportion = valid_trip_count / total_trip_count
    print(f"\n📊 Proportion of valid trips: {valid_trip_count}/{total_trip_count} = {proportion:.2%}")
else:
    print("⚠️ No trips to process.")

# -------------------------------
# 7. Function to modify edges for opposition at the start or end
# -------------------------------
def modify_edges_for_opposition(route_edges):
    edges = route_edges.split()
    if len(edges) > 1 and edges[1] == '-' + edges[0]:
        print(f"🔄 Removing first edge {edges[0]} at the start due to opposition.")
        edges = edges[1:]
    if len(edges) > 1 and edges[-1] == '-' + edges[-2]:
        print(f"🔄 Removing last edge {edges[-1]} at the end due to opposition.")
        edges = edges[:-1]
    return " ".join(edges)

# -------------------------------
# 8. Define Vehicle Types (vType definitions)
# -------------------------------
# We define a set of vType elements for each vehicle class and behavior combination.
# Base vehicle classes:
base_vehicle_types = {
    "car": {"vClass": "passenger", "accel": 2.6, "decel": 4.5, "length": 4.5, "maxSpeed": 30, "color": "1,0,0", "guiShape": "passenger"},
    "bus": {"vClass": "bus", "accel": 1.5, "decel": 3.0, "length": 12.0, "maxSpeed": 20, "color": "0,0.5,1", "guiShape": "bus"},
    "truck": {"vClass": "truck", "accel": 1.2, "decel": 2.5, "length": 10.0, "maxSpeed": 25, "color": "0.5,0.5,0.5", "guiShape": "delivery"},
    "motorcycle": {"vClass": "motorcycle", "accel": 2.5, "decel": 4.2, "length": 2.2, "maxSpeed": 35, "color": "1,1,0", "guiShape": "motorcycle"}
}

# Behavior modifiers for different driving styles:
behavior_modifiers = {
    "aggressive": {"accel": 1.2, "decel": 1.1, "maxSpeed": 1.1},
    "normal": {"accel": 1.0, "decel": 1.0, "maxSpeed": 1.0},
    "cautious": {"accel": 0.8, "decel": 0.9, "maxSpeed": 0.9}
}

# Define overall distribution for vehicle classes (this can be adjusted with real data)
vehicle_class_distribution = {
    "car": 0.7,
    "bus": 0.1,
    "truck": 0.15,
    "motorcycle": 0.05
}

# Function to assign a vehicle class based on distribution
def assign_vehicle_class():
    classes = list(vehicle_class_distribution.keys())
    probabilities = list(vehicle_class_distribution.values())
    return random.choices(classes, probabilities)[0]

# -------------------------------
# 9. Generate XML vType Definitions
# -------------------------------
routes_root = Element("routes")

# Create a dictionary to hold vType definitions
vtype_defs = {}

# We'll generate a vType for every combination of vehicle class and behavior.
for veh_class in base_vehicle_types:
    base = base_vehicle_types[veh_class]
    for behavior in behavior_modifiers:
        mod = behavior_modifiers[behavior]
        # Calculate modified parameters
        accel = base["accel"] * mod["accel"]
        decel = base["decel"] * mod["decel"]
        maxSpeed = base["maxSpeed"] * mod["maxSpeed"]
        vtype_id = f"{veh_class}_{behavior}"
        vtype_defs[vtype_id] = {
            "id": vtype_id,
            "vClass": base["vClass"],
            "accel": str(round(accel, 2)),
            "decel": str(round(decel, 2)),
            "length": str(base["length"]),
            "maxSpeed": str(maxSpeed),
            "color": base["color"],
            "guiShape": base["guiShape"]
        }
        # Create the vType XML element
        SubElement(
            routes_root,
            "vType",
            vtype_defs[vtype_id]
        )

# -------------------------------
# 10. Generate Routes (XML Routes Section)
# -------------------------------
route_id_map = {}
for route in routes:
    # Incorporate both date and time ranges into the route id for uniqueness
    route_id = f"route_{route['origin_region']}_{route['destination_region']}_{route['date_range']}_{route['time_range']}".replace(" ", "_")
    if route_id not in route_id_map:
        path = route["path"]
        modified_edges = modify_edges_for_opposition(" ".join(path))
        SubElement(
            routes_root,
            "route",
            {
                "id": route_id,
                "edges": modified_edges
            }
        )
        route_id_map[route_id] = modified_edges

# -------------------------------
# 11. Define behavior proportions for each time range (for additional selection if needed)
# -------------------------------
time_periods_behavior = {
    "00:00 - 01:00": {"cautious": 60, "normal": 30, "aggressive": 10},
    "01:00 - 02:00": {"cautious": 60, "normal": 30, "aggressive": 10},
    "02:00 - 03:00": {"cautious": 60, "normal": 30, "aggressive": 10},
    "03:00 - 04:00": {"cautious": 60, "normal": 30, "aggressive": 10},
    "04:00 - 05:00": {"cautious": 60, "normal": 30, "aggressive": 10},
    "05:00 - 06:00": {"cautious": 60, "normal": 30, "aggressive": 10},
    "06:00 - 07:00": {"cautious": 20, "normal": 40, "aggressive": 40},
    "07:00 - 08:00": {"cautious": 20, "normal": 40, "aggressive": 40},
    "08:00 - 09:00": {"cautious": 20, "normal": 40, "aggressive": 40},
    "09:00 - 10:00": {"cautious": 20, "normal": 40, "aggressive": 40},
    "10:00 - 11:00": {"cautious": 15, "normal": 60, "aggressive": 25},
    "11:00 - 12:00": {"cautious": 15, "normal": 60, "aggressive": 25},
    "12:00 - 13:00": {"cautious": 15, "normal": 70, "aggressive": 15},
    "13:00 - 14:00": {"cautious": 15, "normal": 70, "aggressive": 15},
    "14:00 - 15:00": {"cautious": 20, "normal": 65, "aggressive": 15},
    "15:00 - 16:00": {"cautious": 20, "normal": 65, "aggressive": 15},
    "16:00 - 17:00": {"cautious": 10, "normal": 40, "aggressive": 50},
    "17:00 - 18:00": {"cautious": 10, "normal": 40, "aggressive": 50},
    "18:00 - 19:00": {"cautious": 25, "normal": 60, "aggressive": 15},
    "19:00 - 20:00": {"cautious": 25, "normal": 60, "aggressive": 15},
    "20:00 - 21:00": {"cautious": 40, "normal": 50, "aggressive": 10},
    "21:00 - 22:00": {"cautious": 40, "normal": 50, "aggressive": 10},
    "22:00 - 23:00": {"cautious": 40, "normal": 50, "aggressive": 10},
    "23:00 - 00:00": {"cautious": 40, "normal": 50, "aggressive": 10}
}

def assign_driver_behavior(time_range_label):
    # For our purpose, we select behavior based on the time range proportions.
    behavior_dict = time_periods_behavior.get(time_range_label, {"normal": 100})
    behaviors = list(behavior_dict.keys())
    weights = list(behavior_dict.values())
    return random.choices(behaviors, weights=weights, k=1)[0]

# -------------------------------
# 12. Generate Vehicles with Assigned Vehicle Type and Behavior
# -------------------------------
vehicles = []
trip_id_counter = 0
mult=0
for route in routes:
    # Create a unique route id (must match the one used above)
    route_id = f"route_{route['origin_region']}_{route['destination_region']}_{route['date_range']}_{route['time_range']}".replace(" ", "_")
    start_sec, end_sec = time_range_to_seconds(route["time_range"])
    mult=0
    if(route["date_range"]=="2024-08-21 - 2024-08-22"):
        mult=0
    if(route["date_range"]=="2024-08-22 - 2024-08-23"):
        mult=3600*24
    if(route["date_range"]=="2024-08-23 - 2024-08-24"):
        mult=3600*24*2
    if(route["date_range"]=="2024-08-24 - 2024-08-25"):
        mult=3600*24*3
    start_sec=start_sec+mult
    end_sec=end_sec+mult

    for _ in range(int(route["trip_count"])):
        depart_time = round(random.uniform(start_sec, end_sec), 2)
        # First, assign a vehicle class based on overall distribution.
        veh_class = assign_vehicle_class()
        # Then, assign a behavior for this time range.
        behavior = assign_driver_behavior(route["time_range"])
        # Build the combined vType id.
        vtype_id = f"{veh_class}_{behavior}"
        vehicle = {
            "id": f"veh_{trip_id_counter}",
            "depart": depart_time,
            "route": route_id,
            "type": vtype_id
        }
        vehicles.append(vehicle)
        trip_id_counter += 1

vehicles.sort(key=lambda v: v["depart"])

# -------------------------------
# 13. Write Vehicles to XML
# -------------------------------
for v in vehicles:
    SubElement(
        routes_root,
        "vehicle",
        {
            "id": v["id"],
            "depart": str(v["depart"]),
            "route": v["route"],
            "type": v["type"]
        }
    )

# -------------------------------
# 14. Save to XML file
# -------------------------------
tree = ElementTree(routes_root)
tree.write("generated_routes.xml", encoding="utf-8", xml_declaration=True)
print(f"✅ Sorted {trip_id_counter} vehicles by depart time and saved to generated_routes.xml")