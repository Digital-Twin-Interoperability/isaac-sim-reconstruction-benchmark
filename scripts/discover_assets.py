#!/usr/bin/env python3
"""Auto-discover warehouse assets from Isaac Sim's S3 asset catalog.

Queries the public Omniverse S3 bucket to enumerate all available USD assets
under Isaac/Props/, Isaac/Environments/Simple_Warehouse/Props/, etc.
Classifies them into categories and produces data/asset_taxonomy.json.

Usage:
    uv run python scripts/discover_assets.py
    uv run python scripts/discover_assets.py --dry-run
    uv run python scripts/discover_assets.py --version 5.1
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

S3_BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# ---------------------------------------------------------------------------
# S3 listing
# ---------------------------------------------------------------------------


def list_s3_prefix(prefix: str, max_keys: int = 5000) -> list[str]:
    """List all keys under an S3 prefix, handling pagination."""
    all_keys: list[str] = []
    marker = ""
    while True:
        url = f"{S3_BUCKET}/?prefix={prefix}&max-keys={max_keys}"
        if marker:
            url += f"&marker={marker}"
        with urlopen(url) as resp:
            root = ET.parse(resp).getroot()

        for c in root.findall("s3:Contents", S3_NS):
            key = c.find("s3:Key", S3_NS).text
            all_keys.append(key)

        truncated = root.find("s3:IsTruncated", S3_NS)
        if truncated is not None and truncated.text == "true":
            marker = all_keys[-1]
        else:
            break
    return all_keys


def discover_usd_assets(version: str, prefixes: list[str]) -> list[dict]:
    """Discover .usd files from S3 under given prefixes."""
    assets = []
    for prefix_template in prefixes:
        prefix = f"Assets/Isaac/{version}/{prefix_template}"
        print(f"  Listing s3://.../{prefix} ...")
        keys = list_s3_prefix(prefix)
        for key in keys:
            if not (key.endswith(".usd") or key.endswith(".usda") or key.endswith(".usdc")):
                continue
            # Skip thumbnails, textures, materials (these are support files, not placeable assets)
            if "/.thumbs/" in key or "/Textures/" in key or "/Materials/" in key:
                continue
            # Strip the version prefix → Isaac/...
            rel = key.replace(f"Assets/Isaac/{version}/", "")
            assets.append({"usd_path": rel, "source": f"s3:{prefix_template}"})
    return assets


# ---------------------------------------------------------------------------
# Instance deduplication
# ---------------------------------------------------------------------------

# Instance suffix: _NNN (2+ digits at the end) — these are placed instances,
# not unique asset types. E.g., SM_CardBoxA_01_301 is an instance of SM_CardBoxA_01.
INSTANCE_SUFFIX_RE = re.compile(r"_\d{2,}$")


def is_instance(stem: str) -> bool:
    """Check if a stem looks like a placed instance rather than a base asset."""
    # Strip SM_ prefix for analysis
    clean = stem.replace("SM_", "").replace("S_", "")
    # Base assets have patterns like CardBoxA_01, BarelPlastic_A_01
    # Instances have patterns like CardBoxA_01_301, CardBoxA_01_1487
    # Heuristic: if removing the last _NNN suffix still matches a known pattern, it's an instance
    parts = clean.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) >= 2:
        # Check if the number is > typical variant numbering (01-09)
        num = int(parts[1])
        if num >= 100:
            return True
    return False


# ---------------------------------------------------------------------------
# Asset classification rules
# ---------------------------------------------------------------------------

# (regex_on_stem, category_id, family, semantic_class, display_name)
CLASSIFICATION_RULES: list[tuple[str, str, str, str, str]] = [
    # Racks & shelving
    (r"RackFrame", "rack_frame", "rack", "rack_frame", "Rack Frame"),
    (r"RackShelf", "rack_shelf", "rack", "rack_shelf", "Rack Shelf"),
    (r"RackPile", "rack_pile", "rack", "rack_pile", "Rack Pile"),
    (r"Rackshield", "rack_shield", "rack", "rack_shield", "Rack Shield"),
    (r"RackLongEmpty", "rack_long", "rack", "rack_long", "Rack Long"),
    (r"BracketSlot", "bracket_slot", "rack", "bracket", "Bracket Slot"),
    (r"BracketBeam", "bracket_beam", "rack", "bracket", "Bracket Beam"),
    (r"BeamA", "beam", "rack", "beam", "Beam"),
    # Boxes
    (r"CardBox", "cardboard_box", "box", "box", "Cardboard Box"),
    (r"FuseBox", "fuse_box", "box", "fuse_box", "Fuse Box"),
    # Crates
    (r"CratePlastic", "crate_plastic", "crate", "crate", "Plastic Crate"),
    # Barrels
    (r"Barel", "barrel_plastic", "barrel", "barrel", "Plastic Barrel"),
    # Bottles
    (r"BottlePlastic", "bottle_plastic", "bottle", "bottle", "Plastic Bottle"),
    # Buckets
    (r"BucketPlastic", "bucket_plastic", "container", "bucket", "Plastic Bucket"),
    # Pallets
    (r"Palette", "pallet", "pallet", "pallet", "Pallet"),
    (r"pallet_holder", "pallet_holder", "pallet", "pallet_holder", "Pallet Holder"),
    (r"^pallet$", "pallet_standard", "pallet", "pallet", "Standard Pallet"),
    # Conveyors — classify by number range
    (r"ConveyorBelt_A0[1-9]$", "conveyor_straight", "conveyor", "conveyor_straight", "Straight Conveyor"),
    (r"ConveyorBelt_A1[0-4]$", "conveyor_curve", "conveyor", "conveyor_curve", "Curve Conveyor"),
    (r"ConveyorBelt_A(1[5-9]|[2-4]\d)$", "conveyor_special", "conveyor", "conveyor_special", "Special Conveyor"),
    (r"ConveyorBelt", "conveyor_other", "conveyor", "conveyor", "Conveyor"),
    # Vehicles
    (r"[Ff]orklift", "forklift", "vehicle", "forklift", "Forklift"),
    (r"ForkliftFork", "forklift_fork", "vehicle", "forklift", "Forklift Fork"),
    (r"Pushcart", "pushcart", "vehicle", "pushcart", "Pushcart"),
    (r"^dolly", "dolly", "vehicle", "dolly", "Dolly"),
    # Safety & signage
    (r"TrafficCone", "traffic_cone", "safety", "traffic_cone", "Traffic Cone"),
    (r"WetFloorSign", "wet_floor_sign", "safety", "wet_floor_sign", "Wet Floor Sign"),
    (r"FireExtinguisher", "fire_extinguisher", "safety", "fire_extinguisher", "Fire Extinguisher"),
    # Bins & containers
    (r"KLT", "klt_bin", "container", "klt_bin", "KLT Bin"),
    (r"flip_stack", "flip_stack", "container", "flip_stack", "Flip Stack"),
    # Building structure
    (r"Warehous", "warehouse_tile", "building", "warehouse_tile", "Warehouse Tile"),
    (r"Wall", "wall", "building", "wall", "Wall"),
    (r"floor", "floor", "building", "floor", "Floor"),
    (r"Ceiling", "ceiling", "building", "ceiling", "Ceiling"),
    (r"Pillar", "pillar", "building", "pillar", "Pillar"),
    (r"Lamp", "lamp", "building", "light_fixture", "Lamp"),
    # Signage & labels
    (r"Sign", "sign", "signage", "sign", "Sign"),
    (r"FloorDecal", "floor_decal", "signage", "floor_decal", "Floor Decal"),
    (r"EmergencyBoard", "emergency_board", "signage", "emergency_board", "Emergency Board"),
    (r"Barcode", "barcode", "signage", "barcode", "Barcode"),
    # Paper / notes
    (r"Paper", "paper", "signage", "paper", "Paper Note"),
    # Sensors (matched by path, see also classify_by_path)
    # People
    # Environments / Samples — handled via path-based classification
]

# ---------------------------------------------------------------------------
# Path-based classification (for assets that don't match name patterns)
# ---------------------------------------------------------------------------

# (path_regex, category_id, family, semantic_class, display_name)
PATH_CLASSIFICATION_RULES: list[tuple[str, str, str, str, str]] = [
    # Sensors
    (r"Sensors/.*/.*Lidar", "sensor_lidar", "sensor", "sensor_lidar", "LiDAR Sensor"),
    (r"Sensors/HESAI", "sensor_lidar", "sensor", "sensor_lidar", "LiDAR Sensor"),
    (r"Sensors/Ouster", "sensor_lidar", "sensor", "sensor_lidar", "LiDAR Sensor"),
    (r"Sensors/SICK", "sensor_lidar", "sensor", "sensor_lidar", "LiDAR Sensor"),
    (r"Sensors/Slamtec", "sensor_lidar", "sensor", "sensor_lidar", "LiDAR Sensor"),
    (r"Sensors/ZVISION", "sensor_lidar", "sensor", "sensor_lidar", "LiDAR Sensor"),
    (r"Sensors/(Intel|LeopardImaging|Orbbec|Stereolabs|Sensing|Tashan)", "sensor_camera", "sensor", "sensor_camera", "Camera Sensor"),
    (r"Sensors/DepthSensor", "sensor_depth", "sensor", "sensor_depth", "Depth Sensor"),
    (r"Sensors/NVIDIA", "sensor_debug", "sensor", "sensor", "Debug Sensor"),
    # People
    (r"People/", "person", "person", "person", "Person"),
    # Environments
    (r"Environments/Simple_Warehouse/warehouse", "environment_warehouse", "environment", "warehouse", "Warehouse Environment"),
    (r"Environments/Simple_Room", "environment_room", "environment", "room", "Simple Room"),
    (r"Environments/Grid", "environment_grid", "environment", "grid", "Grid Environment"),
    (r"Environments/Hospital", "environment_hospital", "environment", "hospital", "Hospital Environment"),
    (r"Environments/Office", "environment_office", "environment", "office", "Office Environment"),
    (r"Environments/Outdoor", "environment_outdoor", "environment", "outdoor", "Outdoor Environment"),
    (r"Environments/Jetracer", "environment_jetracer", "environment", "jetracer", "Jetracer Environment"),
    (r"Environments/Terrains", "environment_terrain", "environment", "terrain", "Terrain"),
    # Props — generic fallbacks for categories not caught by name rules
    (r"Props/Blocks/", "block", "prop", "block", "Block"),
    (r"Props/Beaker/", "beaker", "prop", "beaker", "Beaker"),
    (r"Props/Food/", "food", "prop", "food", "Food"),
    (r"Props/Mugs/", "mug", "prop", "mug", "Mug"),
    (r"Props/Shapes/", "shape", "prop", "shape", "Shape"),
    (r"Props/Rubiks_Cube/", "rubiks_cube", "prop", "rubiks_cube", "Rubiks Cube"),
    (r"Props/Mounts/", "mount", "prop", "mount", "Mount/Table"),
    (r"Props/Camera/", "camera_prop", "prop", "camera_prop", "Camera Prop"),
    (r"Props/PackingTable/", "packing_table", "prop", "packing_table", "Packing Table"),
    (r"Props/Sektion_Cabinet/", "cabinet", "prop", "cabinet", "Cabinet"),
    (r"Props/Sortbot_Housing/", "sortbot_housing", "prop", "sortbot_housing", "Sortbot Housing"),
    (r"Props/DeformableTube/", "deformable_tube", "prop", "deformable_tube", "Deformable Tube"),
    (r"Props/Factory/", "factory_part", "prop", "factory_part", "Factory Part"),
    (r"Props/YCB/", "ycb_object", "prop", "ycb_object", "YCB Object"),
    (r"Props/isaac_ros_segment_anything/", "sam_prop", "prop", "sam_prop", "SAM Test Object"),
    (r"Props/NVIDIA/", "nvidia_prop", "prop", "nvidia_prop", "NVIDIA Prop"),
    (r"Props/UIElements/", "ui_element", "prop", "ui_element", "UI Element"),
    # Samples — pre-built scenes
    (r"Samples/Replicator/", "sample_replicator", "sample", "sample", "Replicator Sample"),
    (r"Samples/ROS2/", "sample_ros2", "sample", "sample", "ROS2 Sample"),
    (r"Samples/Leonardo/", "sample_leonardo", "sample", "sample", "Leonardo Sample"),
    (r"Samples/NvBlox/", "sample_nvblox", "sample", "sample", "NvBlox Sample"),
    (r"Samples/Scene_Blox/", "sample_scene_blox", "sample", "sample", "Scene Blox Sample"),
    (r"Samples/Rigging/", "sample_rigging", "sample", "sample", "Rigging Sample"),
    (r"Samples/Examples/", "sample_examples", "sample", "sample", "Example Sample"),
    (r"Samples/Cortex/", "sample_cortex", "sample", "sample", "Cortex Sample"),
    (r"Samples/Groot/", "sample_groot", "sample", "sample", "Groot Sample"),
    (r"Samples/AnimRobot/", "sample_anim_robot", "sample", "sample", "AnimRobot Sample"),
    (r"Samples/OmniGraph/", "sample_omnigraph", "sample", "sample", "OmniGraph Sample"),
    # Sensor fallback
    (r"Sensors/", "sensor_other", "sensor", "sensor", "Sensor"),
    # IsaacLab
    (r"IsaacLab/", "isaaclab_asset", "isaaclab", "isaaclab", "IsaacLab Asset"),
]


def classify_by_path(usd_path: str) -> tuple[str, str, str, str] | None:
    """Fallback classification based on full path."""
    for pattern, cat_id, family, sem_class, display in PATH_CLASSIFICATION_RULES:
        if re.search(pattern, usd_path):
            return cat_id, family, sem_class, display
    return None


def classify_asset(stem: str) -> tuple[str, str, str, str] | None:
    """Return (category_id, family, semantic_class, display_prefix) or None."""
    for pattern, cat_id, family, sem_class, display in CLASSIFICATION_RULES:
        if re.search(pattern, stem, re.IGNORECASE):
            return cat_id, family, sem_class, display
    return None


# ---------------------------------------------------------------------------
# Taxonomy builder
# ---------------------------------------------------------------------------


def build_taxonomy(all_assets: list[dict]) -> tuple[dict, list[str]]:
    """Group assets into categories and build the taxonomy JSON structure."""
    # Deduplicate by usd_path
    seen: dict[str, dict] = {}
    for asset in all_assets:
        p = asset["usd_path"]
        if p not in seen:
            seen[p] = asset

    # Filter out instances and non-asset files
    categories: dict[str, dict] = {}
    unclassified = []

    for usd_path, asset in sorted(seen.items()):
        stem = Path(usd_path).stem

        # Skip placed instances
        if is_instance(stem):
            continue

        # Skip physics variants, instanceable meshes, etc.
        if any(s in stem.lower() for s in ["_physics", "instanceable", "collision"]):
            continue

        # Robot assets get special handling
        is_robot = "/Robots/" in usd_path
        if is_robot:
            if not is_robot_main_file(usd_path):
                continue
            result = classify_robot(usd_path)
        else:
            result = classify_asset(stem)

        # Fallback to path-based classification
        if result is None:
            result = classify_by_path(usd_path)

        if result is None:
            unclassified.append(usd_path)
            continue

        cat_id, family, sem_class, display_prefix = result

        if cat_id not in categories:
            categories[cat_id] = {
                "category_id": cat_id,
                "name": display_prefix,
                "family": family,
                "variants": [],
            }

        # Variant ID from filename stem (robots include manufacturer for clarity)
        if is_robot:
            parts = usd_path.split("/")
            manufacturer = parts[2] if len(parts) >= 3 else ""
            variant_id = f"{manufacturer}_{stem}"
            display_name = f"{manufacturer} {stem.replace('_', ' ')}"
        else:
            variant_id = stem.replace("SM_", "").replace("S_", "")
            display_name = variant_id.replace("_", " ")

        categories[cat_id]["variants"].append({
            "variant_id": variant_id,
            "name": display_name,
            "usd_path": usd_path,
            "semantic_class": sem_class,
        })

    sorted_cats = sorted(categories.values(), key=lambda c: (c["family"], c["category_id"]))
    taxonomy = {"version": "v2", "categories": sorted_cats}
    return taxonomy, unclassified


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

S3_PREFIXES = [
    # Everything
    "Isaac/Environments/",
    "Isaac/Props/",
    "Isaac/Robots/",
    "Isaac/Sensors/",
    "Isaac/People/",
    "Isaac/Samples/",
    "Isaac/IsaacLab/",
    # Skip Isaac/Materials/ — those are MDL textures, not scene assets
]


# ---------------------------------------------------------------------------
# Robot classification
# ---------------------------------------------------------------------------

# Known robot types by manufacturer/model directory
ROBOT_TYPE_MAP: dict[str, dict[str, str]] = {
    # Quadrupeds
    "ANYbotics": {"_default": "quadruped"},
    "BostonDynamics": {"_default": "quadruped"},
    "Unitree": {"Go1": "quadruped", "Go2": "quadruped", "B1": "quadruped", "B2": "quadruped",
                "A1": "quadruped", "Z1": "manipulator",
                "G1": "humanoid", "H1": "humanoid", "H1_2": "humanoid"},
    # Humanoids
    "1X": {"_default": "humanoid"},
    "Agibot": {"_default": "humanoid"},
    "Agility": {"_default": "humanoid"},
    "BoosterRobotics": {"_default": "humanoid"},
    "FourierIntelligence": {"_default": "humanoid"},
    "Ihmcrobotics": {"_default": "humanoid"},
    "RobotEra": {"_default": "humanoid"},
    "SanctuaryAI": {"_default": "humanoid"},
    "XHumanoid": {"_default": "humanoid"},
    "XiaoPeng": {"_default": "humanoid"},
    "IsaacSim": {"Humanoid": "humanoid", "Ant": "quadruped", "Cartpole": "other",
                 "ForkliftC": "mobile", "Quadcopter": "drone",
                 "SimpleArticulation": "other"},
    # Mobile robots
    "AgilexRobotics": {"_default": "mobile"},
    "Clearpath": {"_default": "mobile"},
    "Fraunhofer": {"_default": "mobile"},
    "Idealworks": {"_default": "mobile"},
    "iRobot": {"_default": "mobile"},
    "NVIDIA": {"_default": "mobile"},
    "Turtlebot": {"_default": "mobile"},
    "Yahboom": {"_default": "mobile"},
    # Manipulator arms
    "Denso": {"_default": "manipulator"},
    "Fanuc": {"_default": "manipulator"},
    "Festo": {"_default": "manipulator"},
    "Flexiv": {"_default": "manipulator"},
    "FrankaRobotics": {"_default": "manipulator"},
    "Kawasaki": {"_default": "manipulator"},
    "Kinova": {"_default": "manipulator"},
    "Kuka": {"_default": "manipulator"},
    "NASA": {"_default": "manipulator"},
    "NTNU": {"_default": "manipulator"},
    "OpenArm": {"_default": "manipulator"},
    "RethinkRobotics": {"_default": "manipulator"},
    "RobotStudio": {"_default": "manipulator"},
    "Techman": {"_default": "manipulator"},
    "Ufactory": {"_default": "manipulator"},
    "UniversalRobots": {"_default": "manipulator"},
    "Yaskawa": {"_default": "manipulator"},
    # Grippers / hands
    "Robotiq": {"_default": "gripper"},
    "Schunk": {"_default": "gripper"},
    "ShadowRobot": {"_default": "gripper"},
    "WonikRobotics": {"_default": "gripper"},
    # Drone
    "Bitcraze": {"_default": "drone"},
}

ROBOT_TYPE_TO_CATEGORY = {
    "quadruped": ("robot_quadruped", "robot", "robot_quadruped", "Quadruped Robot"),
    "humanoid": ("robot_humanoid", "robot", "robot_humanoid", "Humanoid Robot"),
    "mobile": ("robot_mobile", "robot", "robot_mobile", "Mobile Robot"),
    "manipulator": ("robot_manipulator", "robot", "robot_manipulator", "Manipulator Arm"),
    "gripper": ("robot_gripper", "robot", "robot_gripper", "Robot Gripper"),
    "drone": ("robot_drone", "robot", "robot_drone", "Drone"),
    "other": ("robot_other", "robot", "robot", "Other Robot"),
}


def is_robot_main_file(usd_path: str) -> bool:
    """Filter to just top-level robot model files, not sub-parts."""
    lower = usd_path.lower()
    # Skip sub-component directories
    skip_patterns = [
        "/parts/", "/detailedprops/", "/highresprops/", "/props/",
        "/configuration/", "/legacy/", "instanceable", "_collision",
    ]
    if any(p in lower for p in skip_patterns):
        return False
    return True


def classify_robot(usd_path: str) -> tuple[str, str, str, str] | None:
    """Classify a robot USD path into category/family/class/display."""
    parts = usd_path.split("/")
    if len(parts) < 4 or parts[1] != "Robots":
        return None
    manufacturer = parts[2]
    model_dir = parts[3] if len(parts) >= 4 else ""

    mfr_map = ROBOT_TYPE_MAP.get(manufacturer)
    if mfr_map is None:
        rtype = "other"
    else:
        rtype = mfr_map.get(model_dir, mfr_map.get("_default", "other"))

    return ROBOT_TYPE_TO_CATEGORY.get(rtype, ROBOT_TYPE_TO_CATEGORY["other"])


def main():
    parser = argparse.ArgumentParser(description="Discover Isaac Sim assets from S3 catalog")
    parser.add_argument(
        "--version", default="5.1", help="Isaac Sim asset version (default: 5.1)"
    )
    parser.add_argument(
        "--output", default="data/asset_taxonomy.json", help="Output taxonomy JSON path"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    print(f"Discovering assets from S3 (Isaac Sim {args.version}) ...\n")

    all_assets = discover_usd_assets(args.version, S3_PREFIXES)
    print(f"\nTotal .usd files found: {len(all_assets)}")

    taxonomy, unclassified = build_taxonomy(all_assets)

    # Summary
    total_variants = sum(len(c["variants"]) for c in taxonomy["categories"])
    print(f"\nTaxonomy v2:")
    print(f"  Categories: {len(taxonomy['categories'])}")
    print(f"  Total variants: {total_variants}")
    print()
    for cat in taxonomy["categories"]:
        print(f"  {cat['category_id']} ({cat['family']}): {len(cat['variants'])} variants")
        for v in cat["variants"]:
            print(f"    - {v['variant_id']}: {v['usd_path']}")

    if unclassified:
        print(f"\n  Unclassified ({len(unclassified)}):")
        for p in unclassified:
            print(f"    - {p}")

    if not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(taxonomy, indent=2) + "\n")
        print(f"\nWritten to {output_path}")
    else:
        print("\n(dry-run — not written)")


if __name__ == "__main__":
    main()
