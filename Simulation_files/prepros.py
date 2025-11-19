import os
import pandas as pd

# ------------------------------------------------------------------
# 1.  LOAD THE DATA (path is now on Nancy’s front-end, in your home)
# ------------------------------------------------------------------
data_path = os.path.expanduser("~/data/Training.csv")
df = pd.read_csv(data_path)

# ------------------------------------------------------------------
# 2.  CLEAN THE ‘Vehicle Type’ COLUMN
# ------------------------------------------------------------------
def simplify_vehicle_type(vtype: str) -> str:          # fast, vector-friendly
    vtype_low = vtype.lower()
    if vtype_low.startswith("car_"):
        return "car"
    if "bus_" in vtype_low:
        return "bus"
    if "motorcycle_" in vtype_low:
        return "motorcycle"
    if "truck_" in vtype_low:
        return "truck"
    return vtype           # leave untouched if no match

df["Vehicle Type"] = df["Vehicle Type"].apply(simplify_vehicle_type)
print("done")
# ------------------------------------------------------------------
# 3.  KEEP ONLY THE FIRST TWO TOKENS OF ‘Root Detector ID’
#     (memory-safe: no Python lists are built)
# ------------------------------------------------------------------
parts = df["Root Detector ID"].str.split("_", n=2, expand=True)
df["Root Detector ID"] = parts[0] + "_" + parts[1]

# ------------------------------------------------------------------
# 4.  QUICK LOOK
# ------------------------------------------------------------------
print(df.head())
df["Speed (m/s)"] = df["Speed (m/s)"] * 3.6
df.rename(columns={"Speed (m/s)": "Avg Speed (km/h)"}, inplace=True)



# Convert time to minutes
df["Minute"] = (df["Time (s)"] // 30).astype(int)

# Keep the original order of detector IDs as they appear
detector_order = df["Root Detector ID"].drop_duplicates().tolist()

# Define a function to get the first known direction
def get_known_direction(directions):
    for direction in directions:
        if direction != "Unknown":
            return direction
    return "Unknown"

# Group by Minute, Detector ID, and Vehicle ID
aggregated = df.groupby(["Minute", "Root Detector ID", "Vehicle ID"]).agg({
    "Avg Speed (km/h)": "mean",
    "CO2 Emission (mg)": "mean",
    "Direction": get_known_direction,
    "Vehicle Type": "first"
}).reset_index()

# Rename columns for clarity
aggregated.rename(columns={
    "Avg Speed (km/h)": "Average Speed (km/h)",
    "CO2 Emission (mg)": "Average CO2 Emission (mg)",
    "Direction": "Dominant Direction"
}, inplace=True)

# Sort by the order of detector IDs
aggregated["Detector Order"] = aggregated["Root Detector ID"].apply(lambda x: detector_order.index(x))
aggregated.sort_values(by=["Detector Order", "Minute", "Vehicle ID"], inplace=True)
aggregated.drop(columns="Detector Order", inplace=True)
out_path = os.path.expanduser("~/data/clean.csv")   # same folder as the input
# Save the result
aggregated.to_csv(out_path, index=False)

print(f"Clean file written to {out_path}")
