"""
Map generation algorithms for creating urban environments (3D)
"""

import random
import math
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, box
from ..models.entities import Building, Position, EntityType, Map, Depot, Drone, DroneStatus, Store, Customer
import config

# Import floor height from config
FLOOR_HEIGHT = config.FLOOR_HEIGHT  # Height of each floor in meters
INNER_MARGIN = getattr(config, 'BUILDING_INNER_POLY_MARGIN', 0.1)
OUTER_MARGIN = getattr(config, 'BUILDING_OUTER_POLY_MARGIN', 0.4)


class MapGenerator:
    
    def __init__(self, map_width: float = config.MAP_WIDTH, map_depth: float = config.MAP_DEPTH, 
                 max_height: float = config.MAX_MAP_HEIGHT, seed: Optional[int] = None):
        """Initialize 3D Map Generator
        
        Args:
            map_width: Width of map (x-axis)
            map_depth: Depth of map (z-axis)
            max_height: Maximum height for the map (y-axis)
            seed: Random seed for reproducibility
        """
        self.map_width = map_width
        self.map_depth = map_depth
        self.max_height = max_height
        self.map = Map(map_width, map_depth, max_height)
        self.seed = seed

    
    def generate_buildings(self, num_buildings: int = config.TOTAL_BUILDINGS) -> List[Building]:
        """Generate random 3D buildings in the map with safety margins"""
        buildings = []
        
        for i in range(num_buildings):
            # Random building dimensions
            # width and depth: horizontal dimensions (x and z axes)
            width = random.uniform(config.BUILDING_MIN_SIZE, config.BUILDING_MAX_SIZE)
            depth = random.uniform(config.BUILDING_MIN_SIZE, config.BUILDING_MAX_SIZE)
            # height: vertical dimension (y-axis), buildings can be taller
            height = random.uniform(config.BUILDING_MIN_HEIGHT, config.BUILDING_MAX_HEIGHT)
            
            # Try to find a valid position on the ground with safety margin from other buildings
            # get_random_valid_position returns Position(x, height/2, z) - centered vertically
            position = self.map.get_random_valid_position(
                width, height, depth, 
                max_attempts=100, 
                safety_margin=config.BUILDING_MERGING_MARGIN
            )
            
            if position:
                # Create polygons for collision detection
                half_w = width / 2
                half_d = depth / 2
                cx, cz = position.x, position.z
                
                # Inner polygon (slightly smaller for collision detection)
                inner_poly = box(
                    cx - half_w + INNER_MARGIN,
                    cz - half_d + INNER_MARGIN,
                    cx + half_w - INNER_MARGIN,
                    cz + half_d - INNER_MARGIN
                )
                
                # Outer polygon (slightly larger for path planning nodes)
                outer_poly = box(
                    cx - half_w - OUTER_MARGIN,
                    cz - half_d - OUTER_MARGIN,
                    cx + half_w + OUTER_MARGIN,
                    cz + half_d + OUTER_MARGIN
                )
                
                building = Building(
                    id=i,
                    position=position,
                    width=width,
                    height=height,
                    depth=depth,
                    inner_poly=inner_poly,
                    outer_poly=outer_poly
                )
                buildings.append(building)
                self.map.add_building(building)
        
        return buildings
    
    def assign_entities_to_buildings(self, buildings: List[Building], 
                                   store_ratio: float = config.STORE_RATIO,
                                   customer_ratio: float = config.CUSTOMER_RATIO) -> Tuple[List[Store], List[Customer]]:
        """Assign stores and customers to buildings and create floor-by-floor entities
        
        For each building designated as store/customer building:
        - Creates Store/Customer objects for each floor
        - One building can only be either all stores OR all customers (not mixed)
        - Remaining buildings will be empty (entity_type = None)
        
        Returns:
            Tuple of (list of Store objects, list of Customer objects)
        """
        # Shuffle buildings to randomize assignment
        available_buildings = buildings.copy()
        random.shuffle(available_buildings)
        
        num_store_buildings = int(len(buildings) * store_ratio)
        num_customer_buildings = int(len(buildings) * customer_ratio)
        num_empty_buildings = len(buildings) - num_store_buildings - num_customer_buildings
        
        print(f"  Building distribution: {num_store_buildings} stores, {num_customer_buildings} customers, {num_empty_buildings} empty")
        
        all_stores = []
        all_customers = []
        
        store_id_counter = 0
        customer_id_counter = 0
        
        # Assign store buildings and create store entities per floor
        print(f"  Creating stores in {num_store_buildings} buildings...")
        for i in range(num_store_buildings):
            if available_buildings:
                building = available_buildings.pop()
                building.entity_type = EntityType.STORE
                
                # Calculate number of floors in this building
                num_floors = max(1, int(building.height / FLOOR_HEIGHT))
                
                # Create a Store entity for each floor
                for floor in range(num_floors):
                    floor_y = floor * FLOOR_HEIGHT + FLOOR_HEIGHT / 2  # Center of floor
                    
                    store = Store(
                        id=store_id_counter,
                        position=Position(building.position.x, floor_y, building.position.z),
                        building_id=building.id,
                        floor_number=floor
                    )
                    
                    all_stores.append(store)
                    self.map.add_store(store)
                    store_id_counter += 1
        
        print(f"    - Created {len(all_stores)} store entities across floors")
        
        # Assign customer buildings and create customer entities per floor
        print(f"  Creating customers in {num_customer_buildings} buildings...")
        for i in range(num_customer_buildings):
            if available_buildings:
                building = available_buildings.pop()
                building.entity_type = EntityType.CUSTOMER
                
                # Calculate number of floors in this building
                num_floors = max(1, int(building.height / FLOOR_HEIGHT))
                
                # Create a Customer entity for each floor
                for floor in range(num_floors):
                    floor_y = floor * FLOOR_HEIGHT + FLOOR_HEIGHT / 2  # Center of floor
                    
                    customer = Customer(
                        id=customer_id_counter,
                        position=Position(building.position.x, floor_y, building.position.z),
                        building_id=building.id,
                        floor_number=floor
                    )
                    
                    all_customers.append(customer)
                    self.map.add_customer(customer)
                    customer_id_counter += 1
        
        print(f"    - Created {len(all_customers)} customer entities across floors")
        
        # Remaining buildings stay as empty buildings (entity_type = None)
        if available_buildings:
            print(f"  {len(available_buildings)} buildings remain empty (no entity type assigned)")
        
        return all_stores, all_customers
    
    def generate_map(self, num_buildings: int = config.TOTAL_BUILDINGS, 
                     store_ratio: float = config.STORE_RATIO,
                     customer_ratio: float = config.CUSTOMER_RATIO) -> Map:
        """Generate a complete 3D map with buildings, stores, and customers
        
        Creates:
        - 3D building structures
        - Store entities on each floor of store buildings
        - Customer entities on each floor of customer buildings
        - Empty buildings (no entity type)
        """
        if self.seed is not None:
            print(f"Generating map with fixed seed: {self.seed}")
            random.seed(self.seed)

        print("Generating 3D buildings...")
        buildings = self.generate_buildings(num_buildings)
        print(f"Generated {len(buildings)} 3D buildings")
        
        print("Assigning stores and customers to building floors...")
        stores, customers = self.assign_entities_to_buildings(buildings, store_ratio, customer_ratio)
        print(f"Total entities: {len(stores)} stores + {len(customers)} customers")
        
        return self.map
    
    def get_building_positions_for_clustering(self) -> List[Tuple[float, float, float, EntityType]]:
        """Get 3D positions of buildings with their entity types for clustering"""
        positions = []
        for building in self.map.buildings:
            if building.entity_type:
                center = building.get_center()
                positions.append((center.x, center.y, center.z, building.entity_type))
        return positions


class DepotPlacer:
    """Places depots in optimal locations based on clustering"""
    
    def __init__(self, map_obj: Map):
        self.map = map_obj
    
    def find_optimal_depot_positions(self, num_depots: int = config.TOTAL_DEPOTS) -> List[Position]:
        """Find optimal ground-level positions for depots based on building distribution
        
        Depots are always placed at ground level (y=0) and must not overlap with building footprints.
        """
        depot_positions = []
        
        # Get all building floor centers (on the ground)
        building_centers = []
        for building in self.map.buildings:
            if building.entity_type:  # Only consider buildings with entities
                building_centers.append(building.get_floor_center())
        
        if not building_centers:
            return depot_positions
        
        # Simple approach: divide map into grid (on x-z plane) and place depots in areas with most buildings
        grid_size = 3  # 3x3 grid
        cell_width = self.map.width / grid_size
        cell_depth = self.map.depth / grid_size
        
        # Count buildings in each grid cell (using x and z coordinates)
        cell_counts = {}
        for center in building_centers:
            grid_x = int(center.x // cell_width)
            grid_z = int(center.z // cell_depth)
            grid_x = min(grid_x, grid_size - 1)
            grid_z = min(grid_z, grid_size - 1)
            
            cell_key = (grid_x, grid_z)
            cell_counts[cell_key] = cell_counts.get(cell_key, 0) + 1
        
        # Sort cells by building count and select top cells for depot placement
        sorted_cells = sorted(cell_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Depot dimensions (ground-level platform)
        depot_width = config.DEPOT_SIZE
        depot_height = 2  # Small height for depot platform
        depot_depth = config.DEPOT_SIZE
        
        print(f"  Placing {num_depots} depots at ground level...")
        for i in range(min(num_depots, len(sorted_cells))):
            grid_x, grid_z = sorted_cells[i][0]
            
            # Calculate center of the grid cell on ground plane (y=0)
            cell_center_x = grid_x * cell_width + cell_width / 2
            cell_center_z = grid_z * cell_depth + cell_depth / 2
            
            # IMPORTANT: Depot is always at ground level (y = height/2 for centered position)
            cell_center_y = depot_height / 2  # Centered at ground level
            
            # Find a valid position that doesn't overlap with building footprints or existing depots
            depot_pos = self._find_valid_ground_depot_position(
                Position(cell_center_x, cell_center_y, cell_center_z), 
                depot_width, depot_depth, depot_positions
            )
            
            if depot_pos:
                depot_positions.append(depot_pos)
                print(f"    - Depot {i} placed at ({depot_pos.x:.1f}, 0, {depot_pos.z:.1f})")
        
        # If we need more depots, place them randomly (still at ground level)
        max_random_attempts = 50  # Prevent infinite loop
        random_attempt_count = 0
        
        while len(depot_positions) < num_depots and random_attempt_count < max_random_attempts:
            depot_pos = self._find_random_ground_depot_position(depot_width, depot_depth, depot_positions)
            if depot_pos:
                depot_positions.append(depot_pos)
                print(f"    - Depot {len(depot_positions)-1} placed at ({depot_pos.x:.1f}, 0, {depot_pos.z:.1f})")
                random_attempt_count = 0  # Reset counter on success
            else:
                random_attempt_count += 1
        
        if len(depot_positions) < num_depots:
            print(f"    WARNING: Could only place {len(depot_positions)}/{num_depots} depots. Map may be too crowded.")
        
        return depot_positions
    
    def _check_ground_footprint_overlap(self, pos: Position, width: float, depth: float, 
                                        safety_margin: float = config.DEPOT_SAFETY_MARGIN) -> bool:
        """Check if a ground-level footprint overlaps with any building footprint
        
        Args:
            pos: Center position on ground (x, y, z)
            width: Width along x-axis
            depth: Depth along z-axis
            safety_margin: Safety distance to maintain from buildings (default from config)
            
        Returns:
            True if overlaps with any building (including safety margin), False otherwise
        """
        half_width = width / 2
        half_depth = depth / 2
        
        for building in self.map.buildings:
            # Check 2D footprint overlap on x-z plane
            building_half_width = building.width / 2
            building_half_depth = building.depth / 2
            
            # Check if rectangles overlap in 2D (including safety margin)
            x_overlap = abs(pos.x - building.position.x) < (half_width + building_half_width + safety_margin)
            z_overlap = abs(pos.z - building.position.z) < (half_depth + building_half_depth + safety_margin)
            
            if x_overlap and z_overlap:
                return True  # Overlaps with this building (or too close)
        
        return False  # No overlap
    
    def _check_depot_overlap(self, pos: Position, width: float, depth: float, 
                             existing_depots: List[Position]) -> bool:
        """Check if depot overlaps with existing depots
        
        Args:
            pos: Position to check
            width: Depot width
            depth: Depot depth
            existing_depots: List of existing depot positions
            
        Returns:
            True if overlaps with any existing depot, False otherwise
        """
        if not existing_depots:
            return False
        
        half_width = width / 2
        half_depth = depth / 2
        safety_margin = config.DEPOT_SAFETY_MARGIN
        
        for depot_pos in existing_depots:
            # Check 2D distance on x-z plane
            x_dist = abs(pos.x - depot_pos.x)
            z_dist = abs(pos.z - depot_pos.z)
            
            # Check if too close (including safety margin)
            if x_dist < (width + safety_margin) and z_dist < (depth + safety_margin):
                return True
        
        return False
    
    def _find_valid_ground_depot_position(self, preferred_pos: Position, width: float, depth: float,
                                         existing_depots: List[Position] = None) -> Optional[Position]:
        """Find a valid ground-level position for depot near preferred position
        
        Ensures depot footprint doesn't overlap with building footprints or existing depots.
        Uses expanding circle search to find the nearest valid position.
        
        Args:
            preferred_pos: Preferred position to place depot (from clustering)
            width: Depot width (x-axis)
            depth: Depot depth (z-axis)
            existing_depots: List of existing depot positions to avoid
            
        Returns:
            Valid Position object if found, None otherwise
        """
        if existing_depots is None:
            existing_depots = []
        
        depot_height = 2  # Fixed height for depot
        half_width = width / 2
        half_depth = depth / 2
        
        # Try the preferred position first
        if (preferred_pos.x - half_width >= 0 and preferred_pos.x + half_width <= self.map.width and
            preferred_pos.z - half_depth >= 0 and preferred_pos.z + half_depth <= self.map.depth):
            
            if (not self._check_ground_footprint_overlap(preferred_pos, width, depth) and
                not self._check_depot_overlap(preferred_pos, width, depth, existing_depots)):
                # Preferred position is valid - return immediately
                return Position(preferred_pos.x, depot_height / 2, preferred_pos.z)
        
        # Preferred position is not valid - search for nearest alternative
        # Expanding circle search around preferred position
        search_radius = 0
        radius_step = max(width, depth) * 1.5  # Search radius increment
        max_search_radius = min(self.map.width, self.map.depth) / 3  # Maximum search distance
        
        print(f"    Depot position ({preferred_pos.x:.1f}, {preferred_pos.z:.1f}) invalid, searching nearby...")
        
        while search_radius <= max_search_radius:
            search_radius += radius_step
            
            # Try positions at different angles on the circle perimeter
            # Use more angles for finer-grained search (every 45 degrees)
            for angle in range(0, 360, 45):
                angle_rad = math.radians(angle)
                offset_x = search_radius * math.cos(angle_rad)
                offset_z = search_radius * math.sin(angle_rad)
                
                test_pos = Position(
                    preferred_pos.x + offset_x,
                    depot_height / 2,  # Ground level (centered)
                    preferred_pos.z + offset_z
                )
                
                # Check if test position is within map bounds
                if (test_pos.x - half_width >= 0 and test_pos.x + half_width <= self.map.width and
                    test_pos.z - half_depth >= 0 and test_pos.z + half_depth <= self.map.depth):
                    
                    # Check if footprint overlaps with buildings or existing depots
                    if (not self._check_ground_footprint_overlap(test_pos, width, depth) and
                        not self._check_depot_overlap(test_pos, width, depth, existing_depots)):
                        # Found a valid position - return immediately
                        print(f"    ✓ Found alternative at ({test_pos.x:.1f}, {test_pos.z:.1f}), distance: {search_radius:.1f}m")
                        return test_pos
        
        # Could not find a valid position within max search radius
        print(f"    ✗ Warning: Could not find a valid spot for depot near ({preferred_pos.x:.1f}, {preferred_pos.z:.1f}). Skipping this depot.")
        return None
    
    def _find_random_ground_depot_position(self, width: float, depth: float, 
                                           existing_depots: List[Position] = None,
                                           max_attempts: int = 500) -> Optional[Position]:
        """Find a random valid ground-level position for depot
        
        Args:
            width: Depot width (x-axis)
            depth: Depot depth (z-axis)
            existing_depots: List of existing depot positions to avoid
            max_attempts: Maximum number of random attempts
        """
        if existing_depots is None:
            existing_depots = []
        
        depot_height = 2
        half_width = width / 2
        half_depth = depth / 2
        
        building_overlaps = 0
        depot_overlaps = 0
        
        for attempt in range(max_attempts):
            # Random position on ground plane
            x = random.uniform(half_width, self.map.width - half_width)
            z = random.uniform(half_depth, self.map.depth - half_depth)
            
            test_pos = Position(x, depot_height / 2, z)
            
            # Check if footprint overlaps with buildings or existing depots
            building_overlap = self._check_ground_footprint_overlap(test_pos, width, depth)
            depot_overlap = self._check_depot_overlap(test_pos, width, depth, existing_depots)
            
            if building_overlap:
                building_overlaps += 1
            if depot_overlap:
                depot_overlaps += 1
            
            if not building_overlap and not depot_overlap:
                return test_pos
        
        # Debug info if failed
        print(f"      Failed to place depot after {max_attempts} attempts")
        print(f"      - Building overlaps: {building_overlaps}/{max_attempts}")
        print(f"      - Depot overlaps: {depot_overlaps}/{max_attempts}")
        
        return None
    
    def create_depots_with_drones(self, depot_positions: List[Position]) -> List[Depot]:
        """Create depot objects with drones at given positions"""
        depots = []
        
        for i, pos in enumerate(depot_positions):
            
            # 1. 드론 리스트가 비어있는 Depot 객체를 먼저 생성합니다.
            depot = Depot(
                id=i,
                position=pos,
                drones=[], # 우선 빈 리스트로 생성
                capacity=config.DRONES_PER_DEPOT
            )
            
            # 2. 생성된 Depot의 중심점(get_center())을 이용해 드론들을 생성합니다.
            drones = []
            for j in range(config.DRONES_PER_DEPOT):
                drone = Drone(
                    id=f"drone_{i}_{j}",
                    # Depot의 중심점을 드론의 초기 위치로 설정
                    position=depot.get_center().copy(), 
                    depot=depot, # 생성 시점에 depot을 바로 할당
                    status=DroneStatus.IDLE
                )
                drones.append(drone)
            
            # 3. 생성된 드론 리스트를 Depot 객체에 다시 할당합니다.
            depot.drones = drones
            
            depots.append(depot)
            self.map.add_depot(depot)
        
        return depots
