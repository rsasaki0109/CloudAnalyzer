"""Convert recorded ROS 2 GNSS fixes to a single MGRS tile for map validation.

Requires rosbags and pyproj. This preserves measured GNSS altitude; the road
builder estimates ground elevation independently from the surveyed point cloud.
"""
import argparse
import math
import sqlite3
from pathlib import Path

from pyproj import Proj
from rosbags.typesys import Stores, get_typestore

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bag", type=Path, help="ROS 2 SQLite .db3 recording")
parser.add_argument("out", type=Path, help="CSV trajectory destination")
parser.add_argument("--zone", type=int, required=True, help="UTM zone of the map")
parser.add_argument("--south", action="store_true")
args = parser.parse_args()
project = Proj(proj="utm", zone=args.zone, south=args.south, datum="WGS84")
store = get_typestore(Stores.ROS2_HUMBLE)
rows = []
tile = None
with sqlite3.connect(f"file:{args.bag.resolve().as_posix()}?mode=ro", uri=True) as connection:
    topics = connection.execute("SELECT id, type FROM topics WHERE type = 'sensor_msgs/msg/NavSatFix'").fetchall()
    if len(topics) != 1:
        raise ValueError("expected exactly one NavSatFix topic")
    topic_id, message_type = topics[0]
    for timestamp, data in connection.execute("SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,)):
        fix = store.deserialize_cdr(data, message_type)
        if fix.status.status < 0:
            continue
        east, north = project(fix.longitude, fix.latitude)
        current = (math.floor(east / 100000), math.floor(north / 100000))
        if tile is not None and current != tile:
            raise ValueError("trajectory crosses a 100 km tile; split it before projection")
        tile = current
        rows.append((timestamp / 1e9, east % 100000, north % 100000, fix.altitude))
if len(rows) < 2:
    raise ValueError("fewer than two valid GNSS fixes")
with args.out.open("w", newline="\n") as file:
    file.write("timestamp,x,y,z\n")
    for row in rows:
        file.write(",".join(str(value) for value in row) + "\n")
print(f"Wrote {len(rows)} measured fixes to {args.out}")
