from copy import deepcopy
from pathlib import Path

# Configuration file for DVRP simulation (3D)

RUN_VISUALIZER = True
SIMULATION_DELTA_TIME = 0.05 # s
if not RUN_VISUALIZER:
    SIMULATION_TIME = 3600 # s

# Map configuration
MAP_SEED = 123
ORDER_SEED = 456
NODE_OFFSET = 5.0
MAP_SOURCE = "real"  # "real" to use GeoJSON footprints, "random" for synthetic test map
MAP_GEOJSON_PATH = "src/map/buildings.geojson"  # Override if your data lives elsewhere
MAP_BUILDING_LIMIT = None  # Set to int to cap imported buildings for testing

# 3D Map dimensions
MAP_WIDTH = 2000  # X-axis (horizontal)
MAP_HEIGHT = 1000  # Z-axis (horizontal, called HEIGHT for backward compatibility)
MAP_DEPTH = 2000  # Z-axis (depth, same as MAP_HEIGHT)
MAX_MAP_HEIGHT = 500  # Y-axis (maximum altitude for drones)

# Building configuration
TOTAL_BUILDINGS = 100
BUILDING_MIN_SIZE = 20  # Minimum width/depth
BUILDING_MAX_SIZE = 80  # Maximum width/depth
BUILDING_MIN_HEIGHT = 50  # Minimum building height (Y-axis)
BUILDING_MAX_HEIGHT = 200  # Maximum building height (Y-axis)
FLOOR_HEIGHT = 3.0  # Height of each floor in meters
BUILDING_MERGING_MARGIN = 0.5
BUILDING_INNER_POLY_MARGIN = 0.1
BUILDING_OUTER_POLY_MARGIN = 0.4
BUILDING_HEIGHT_SCALE = 1  # Multiplier to globally scale real-world building heights
VISUALIZATION_HEIGHT_SCALE = 1.0  # Multiplier applied only when rendering in Ursina
CAMERA_MOVE_SPEED = 500.0  # EditorCamera move speed for visualization (units/sec)
FOOTPRINT_SIMPLIFY_TOLERANCE = 3.0  # >0 to simplify real-map footprints for collision (meters)

# Building type ratios (should add up to <= 1.0)
STORE_RATIO = 0.3  # 30% of buildings are stores
CUSTOMER_RATIO = 0.5  # 50% of buildings are customers
# Remaining buildings (20%) will be empty buildings

# Depot configuration
TOTAL_DEPOTS = 5
DRONES_PER_DEPOT = 5
DEPOT_SIZE = 20
DEPOT_SAFETY_MARGIN = 30.0  # Safety distance from buildings (in meters)

# Simulation configuration
SIMULATION_SPEED = 1.0  # Real-time multiplier
ORDER_GENERATION_RATE = 0.0003  # Orders per second (lower to avoid overload)
MAX_ORDER_DELAY = 300  # Maximum seconds to wait for order
ROUTE_RETRY_INTERVAL = 60.0  # Seconds to wait before retrying failed routes
ROUTE_RETRY_MAX_ATTEMPTS = 3  # How many times to retry routing an order
ROUTE_CONNECTIVITY_CACHE_TTL = 300.0  # seconds to keep failed depot-route connectivity info

# Drone configuration
DRONE_SPEED = 30  # m/s
DRONE_CAPACITY = 1  # Number of orders per drone
DRONE_BATTERY_LIFE = 25000 # m
DRONE_CHARGING_SPEED = 0.00455 # /s
DRONE_BATTERY_CAPACITY = 2 # kWh

# Routing visualization
VISUALIZE_FIRST_ROUTE = False  # Visualize the first delivery route
VISUALIZE_ALL_ROUTES = False  # Visualize all delivery routes (warning: can be slow)

# Colors for visualization
COLORS = {
    'building': (255, 255, 255),  # Empty buildings: white
    'store': (0, 255, 0),
    'customer': (255, 0, 0),
    'depot': (0, 0, 255),
    'drone': (255, 255, 0),
    'route': (255, 165, 0),
    'background': (240, 240, 240)
}

# Fixed Cost
DEPOT_COST = 3_000_000 # won
DRONE_COST = 1_000_000 # won

# Variable Cost
CHARGING_COST = 300 # won / kWh
TIME_PENALTY_CRITERIA = 120 # s
TIME_PENALTY = 5 # won / s
FAILED_ORDER_PENALTY = TIME_PENALTY * TIME_PENALTY_CRITERIA  # won per missed order (matches previous per-second scale)

# Fixed Cost Weight (set to 1.0 to report annualized fixed cost directly)
FIXED_COST_WEIGHT = 1.0

# Building preprocessing configuration
PROJECT_ROOT = Path(__file__).resolve().parent
MAP_DATA_DIR = PROJECT_ROOT / "src" / "map"

BUILDINGS_DATA_CONFIG = {
    "terrain_contour_paths": [
        MAP_DATA_DIR / "국가기본공간정보_포스텍_이동_효자_SK뷰" / "NF_L_F01000_L_F01000_000000.shp",
    ],
    "spot_elevation_paths": [
        MAP_DATA_DIR / "국가기본공간정보_포스텍_이동_효자_SK뷰" / "NF_P_F02000_P_F02000_000000.shp",
    ],
    "building_paths": [
        MAP_DATA_DIR / "F_FAC_BUILDING_경북_포항시_남구_북구" / "F_FAC_BUILDING_47111_202507.shp",
        MAP_DATA_DIR / "F_FAC_BUILDING_경북_포항시_남구_북구" / "F_FAC_BUILDING_47113_202507.shp",
    ],
    "output_csv_filename": MAP_DATA_DIR / "postech_building_list.csv",
    "output_2d_filename": MAP_DATA_DIR / "postech_2d_map.png",
    "output_3d_filename": MAP_DATA_DIR / "postech_3d_map.png",
    "output_geojson_filename": MAP_DATA_DIR / "buildings.geojson",
    "dpi_2d": 300,
    "dpi_3d": 300,
}


def get_buildings_data_config(overrides=None):
    """Return a copy of the building preprocessing config with optional overrides."""
    config = deepcopy(BUILDINGS_DATA_CONFIG)
    if overrides:
        config.update(overrides)
    return config
