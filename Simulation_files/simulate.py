import traci
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict

# Path to the detector file
detector_file = "cameras.add.xml"

# Extract detector IDs from the XML file
def get_detectors_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    detectors = defaultdict(list)

    # Process laneAreaDetectors
    for detector in root.findall("laneAreaDetector"):
        det_id = detector.get("id")
        output_file = detector.get("file")
        if det_id and output_file:
            root_id = det_id.rsplit("_", 1)[0]
            detectors[root_id].append((det_id, output_file))

    return detectors

# Get all detectors grouped by root ID
detectors = get_detectors_from_xml(detector_file)

# Track previous coordinates of vehicles
previous_positions = {}

# Open CSV file to save results
output_csv = "DaySix.csv"
with open(output_csv, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time (s)", "Root Detector ID", "Vehicle ID", "Vehicle Type", "Speed (m/s)", "Direction", "CO2 Emission (mg)"])

    # Start SUMO simulation
    traci.start(["sumo-gui", "-c", "config.sumo.cfg"])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        current_time = traci.simulation.getTime()

        for root_id, detector_list in detectors.items():
            vehicle_records = []

            for detector_id, _ in detector_list:
                detected_vehicles = traci.lanearea.getLastStepVehicleIDs(detector_id)

                for veh in detected_vehicles:
                    veh_type = traci.vehicle.getTypeID(veh)
                    speed = traci.vehicle.getSpeed(veh)
                    position = traci.vehicle.getPosition(veh)  # (x, y)
                    co2_emission = traci.vehicle.getCO2Emission(veh)

                    if veh in previous_positions:
                        prev_x, prev_y = previous_positions[veh]
                        curr_x, curr_y = position

                        dx = curr_x - prev_x
                        dy = curr_y - prev_y

                        # Determine dominant movement direction
                        if abs(dx) > abs(dy):
                            direction = "Right" if dx > 0 else "Left"
                        else:
                            direction = "Up" if dy > 0 else "Down"
                    else:
                        direction = "Unknown"

                    # Update previous position
                    previous_positions[veh] = position

                    vehicle_records.append([current_time, root_id, veh, veh_type, speed, direction, co2_emission])

            for record in vehicle_records:
                writer.writerow(record)

    traci.close()
