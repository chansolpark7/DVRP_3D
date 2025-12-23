from copy import deepcopy
from pathlib import Path

# Configuration file for DVRP simulation (3D)

# ============================================
# SIMULATION MODE: "drone" or "motorbike"
# ============================================
SIMULATION_MODE = "motorbike"  # Change to "motorbike" for baseline comparison

RUN_VISUALIZER = True
SIMULATION_DELTA_TIME = 0.05 # s
if not RUN_VISUALIZER:
    SIMULATION_TIME = 200 # s

# Map configuration
MAP_SEED = 123
ORDER_SEED = 456
NODE_OFFSET = 5.0
MAP_SOURCE = "random"  # "real" to use GeoJSON footprints, "random" for synthetic test map
MAP_GEOJSON_PATH = "src/map/buildings.geojson"  # Override if your data lives elsewhere
MAP_BUILDING_LIMIT = None  # Set to int to cap imported buildings for testing

# 3D Map dimensions
MAP_WIDTH = 2000  # X-axis (horizontal)
MAP_HEIGHT = 1000  # Z-axis (horizontal, called HEIGHT for backward compatibility)
MAP_DEPTH = 2000  # Z-axis (depth, same as MAP_HEIGHT)
MAX_MAP_HEIGHT = 500  # Y-axis (maximum altitude for drones)

# Building configuration (in meters for realistic scale)
TOTAL_BUILDINGS = 200
BUILDING_MIN_SIZE = 10   # Minimum width/depth (meters) - small building
BUILDING_MAX_SIZE = 40   # Maximum width/depth (meters) - medium building
BUILDING_MIN_HEIGHT = 10  # Minimum building height (meters) - ~3 floors
BUILDING_MAX_HEIGHT = 60  # Maximum building height (meters) - ~20 floors
FLOOR_HEIGHT = 3.0  # Height of each floor in meters
BUILDING_MERGING_MARGIN = 0.5
BUILDING_INNER_POLY_MARGIN = 0.1
BUILDING_OUTER_POLY_MARGIN = 0.4
BUILDING_HEIGHT_SCALE = 1  # Multiplier to globally scale real-world building heights
VISUALIZATION_HEIGHT_SCALE = 1.0  # Multiplier applied only when rendering in Panda3D
VISUALIZATION_SCALE = 1.0  # Global scale multiplier for the entire scene (now using real meters, 1.0 recommended)
CAMERA_MOVE_SPEED = 500.0  # Camera move speed for visualization (units/sec)
FOOTPRINT_SIMPLIFY_TOLERANCE = 3.0  # >0 to simplify real-map footprints for collision (meters)

# Building type ratios (should add up to <= 1.0)
STORE_RATIO = 0.3  # 30% of buildings are stores
CUSTOMER_RATIO = 0.5  # 50% of buildings are customers
# Remaining buildings (20%) will be empty buildings

# Depot configuration
TOTAL_DEPOTS = 2
DRONES_PER_DEPOT = 2  # Also used for motorbikes when SIMULATION_MODE = "motorbike"
VEHICLES_PER_DEPOT = DRONES_PER_DEPOT  # Generic alias
DEPOT_SIZE = 20
DEPOT_SAFETY_MARGIN = 30.0  # Safety distance from buildings (in meters)

# Simulation configuration
SIMULATION_SPEED = 5  # Real-time multiplier
ORDER_GENERATION_RATE = 0.0002  # Orders per second (reduced for motorbike baseline)
MAX_ORDER_DELAY = 7200  # Maximum seconds to wait for order (2 hours)
ROUTE_RETRY_INTERVAL = 30.0  # Seconds to wait before retrying failed routes (reduced)
ROUTE_RETRY_MAX_ATTEMPTS = 5  # How many times to retry routing an order (increased)
ROUTE_CONNECTIVITY_CACHE_TTL = 60.0  # seconds to keep failed depot-route connectivity info (reduced)

# Drone configuration (realistic delivery drone parameters)
# Reference: DJI FlyCart 30, Wing drones, typical commercial delivery drones
DRONE_SPEED = 15  # m/s (~54 km/h, typical cruising speed for delivery drones with payload)
DRONE_VERTICAL_SPEED = 5  # m/s (vertical ascent/descent speed)
DRONE_CAPACITY = 3  # Number of orders per drone (multi-delivery)
DRONE_MAX_PAYLOAD = 5  # kg (maximum payload weight)
DRONE_BATTERY_LIFE = 15000  # m (15km range, conservative with payload, ~25min flight time)
DRONE_CHARGING_TIME = 3600  # seconds (1 hour for full charge, fast charging)
DRONE_CHARGING_SPEED = 1.0 / DRONE_CHARGING_TIME  # Battery fraction per second
DRONE_BATTERY_CAPACITY = 4  # kWh (typical lithium battery pack for delivery drones)

# Service time configuration (pickup and delivery operations) - for DRONE
PICKUP_SERVICE_TIME = 60.0  # Time to pick up food at store (seconds) - 1 minute
DELIVERY_SERVICE_TIME = 60.0  # Time to deliver food to customer (seconds) - 1 minute
SERVICE_TIME_PER_STOP = 60.0  # Default service time per stop (seconds) for route planning
CUSTOMER_MAX_WAIT_TIME = 1800.0  # Maximum customer wait time (30 minutes, relaxed for motorbike mode)
BATTERY_SAFETY_MARGIN = 1.2  # Battery safety margin (20% reserve)

# ============================================
# MOTORBIKE CONFIGURATION (Baseline Comparison)
# ============================================
MOTORBIKE_SPEED = 8  # m/s (~29 km/h, urban average with traffic)
MOTORBIKE_CAPACITY = 3  # Number of orders per motorbike (same as drone for fair comparison)

# Motorbike service time (height-dependent: rider must climb/use elevator)
# Formula: base_time + (floor_number * time_per_floor)
MOTORBIKE_BASE_SERVICE_TIME = 60.0  # Base service time (seconds)
MOTORBIKE_TIME_PER_FLOOR = 15.0  # Additional time per floor (seconds) - elevator wait + travel

# Motorbike has no range limitation (fuel assumed sufficient)
MOTORBIKE_RANGE_LIMIT = None  # Set to float (meters) if range limitation needed

# Motorbike costs
MOTORBIKE_COST = 3_000_000  # won per motorbike (purchase cost)
MOTORBIKE_FUEL_COST_PER_KM = 150  # won/km (fuel consumption)
MOTORBIKE_LABOR_COST_PER_HOUR = 15_000  # won/hour (rider wage)

# Insertion heuristic optimization parameters
INSERTION_MAX_DEPOT_DISTANCE = 2000  # Max distance from depot to store for drone filtering (meters)
INSERTION_GOOD_ENOUGH_THRESHOLD = 50  # Early termination threshold (cost delta)
INSERTION_TOP_K_CANDIDATES = 5  # Number of top candidates to fully validate

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
        MAP_DATA_DIR / "국가기본공간정보_서울_영등포구" / "NF_L_F01000_L_F01000_000000.shp",
        #MAP_DATA_DIR / "국가기본공간정보_포스텍_이동_효자_SK뷰" / "NF_L_F01000_L_F01000_000000.shp",
    ],
    "spot_elevation_paths": [
        MAP_DATA_DIR / "국가기본공간정보_서울_영등포구" / "NF_P_F02000_P_F02000_000000.shp",
        #MAP_DATA_DIR / "국가기본공간정보_포스텍_이동_효자_SK뷰" / "NF_P_F02000_P_F02000_000000.shp",
    ],
    "building_paths": [
        MAP_DATA_DIR / "F_FAC_BUILDING_서울_영등포구" / "F_FAC_BUILDING_11560_202512.shp",
        #MAP_DATA_DIR / "F_FAC_BUILDING_경북_포항시_남구_북구" / "F_FAC_BUILDING_47111_202507.shp",
        #MAP_DATA_DIR / "F_FAC_BUILDING_경북_포항시_남구_북구" / "F_FAC_BUILDING_47113_202507.shp",
    ],
    "output_csv_filename": MAP_DATA_DIR / "seoul_building_list.csv",
    "output_2d_filename": MAP_DATA_DIR / "seoul_2d_map.png",
    "output_3d_filename": MAP_DATA_DIR / "seoul_3d_map.png",
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
