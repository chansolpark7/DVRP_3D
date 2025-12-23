"""
Entity classes for the DVRP simulation
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from shapely.geometry import Polygon, Point, box
from shapely.strtree import STRtree
import config

class EntityType(Enum):
    STORE = "store"
    CUSTOMER = "customer"
    DEPOT = "depot"


class OrderStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DroneStatus(Enum):
    IDLE = "idle"
    LOADING = "loading"
    FLYING = "flying"
    PICKING_UP = "picking_up"  # Waiting at store for food pickup
    DELIVERING = "delivering"  # Flying to customer (legacy, also used during delivery flight)
    DROPPING_OFF = "dropping_off"  # Waiting at customer for delivery handoff
    RETURNING = "returning"


@dataclass
class Position:
    """Represents a 3D position"""
    x: float
    y: float
    z: float
    building_id: Optional[int] = None
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate 3D Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Position') -> 'Position':
        return Position(self.x - other.x, self.y - other.y, self.z - other.z)

    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        elif idx == 2:
            return self.z
        else:
            raise IndexError(f"Position index out of range: {idx}")

    def normalize(self) -> 'Position':
        """벡터를 정규화하여 단위 벡터(길이 1)를 반환합니다."""
        mag = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag == 0:
            return Position(0, 0, 0)
        return Position(self.x / mag, self.y / mag, self.z / mag)

    def __mul__(self, scalar: float) -> 'Position':
        """벡터에 스칼라 값을 곱합니다."""
        return Position(self.x * scalar, self.y * scalar, self.z * scalar)

    # distance_to_2d 함수도 routing.py에서 사용하므로 추가합니다.
    def distance_to_2d(self, other: 'Position') -> float:
        """Calculate Euclidean distance in 2D (x, z plane)"""
        return math.sqrt((self.x - other.x)**2 + (self.z - other.z)**2)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __lt__(self, other: 'Position') -> bool:
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def copy(self) -> 'Position':
        return Position(self.x, self.y, self.z)


@dataclass
class Building:
    """Represents a building in the 3D urban environment
    
    In Panda3D, y-axis is up (height). Building position represents the center point:
    - x, z: horizontal center of the building
    - y: vertical center (height/2 from ground)
    - width: size along x-axis
    - depth: size along z-axis
    - height: size along y-axis (vertical)
    - footprint: optional list of (x, z) tuples describing polygonal base in world coords
    """
    id: int
    position: Position  # Center position (x, height/2, z)
    width: float  # Size along x-axis
    height: float  # Size along y-axis (vertical)
    depth: float  # Size along z-axis
    entity_type: Optional[EntityType] = None
    footprints: Optional[List[List[Tuple[float, float]]]] = None
    inner_poly: Optional[Polygon] = None
    outer_poly: Optional[Polygon] = None

    def _vertical_bounds(self) -> Tuple[float, float]:
        """Return (min_y, max_y) of the building."""
        half_height = self.height / 2
        return self.position.y - half_height, self.position.y + half_height
    
    def contains_point(self, pos: Position) -> bool:
        """Check if a 3D point is inside this building using polygon footprint when available."""
        y_min, y_max = self._vertical_bounds()
        if not (y_min <= pos.y <= y_max):
            return False

        poly = self.inner_poly
        if poly is not None:
            return poly.covers(Point(pos.x, pos.z))

        # Fallback AABB check (should be rare)
        half_width = self.width / 2
        half_depth = self.depth / 2
        return (
            self.position.x - half_width <= pos.x <= self.position.x + half_width
            and self.position.z - half_depth <= pos.z <= self.position.z + half_depth
        )
    
    def get_center(self) -> Position:
        """Get the center position of the building"""
        return self.position.copy()
    
    def get_floor_center(self) -> Position:
        """Get the center position at ground level (y=0)"""
        return Position(self.position.x, 0, self.position.z)
    
    def collides_with(self, other: 'Building', safety_margin: float = 0.0) -> bool:
        """Check if buildings overlap using polygon footprints (with optional margin) and height."""
        y_min_self, y_max_self = self._vertical_bounds()
        y_min_other, y_max_other = other._vertical_bounds()
        y_overlap = not (y_max_self < y_min_other or y_max_other < y_min_self)

        poly_self = self.inner_poly
        poly_other = other.inner_poly

        if poly_self is not None and poly_other is not None:
            return y_overlap and poly_self.intersects(poly_other)

        # Fallback AABB overlap if polygons are missing
        half_width_self = self.width / 2
        half_depth_self = self.depth / 2
        half_width_other = other.width / 2
        half_depth_other = other.depth / 2

        x_overlap = abs(self.position.x - other.position.x) < (half_width_self + half_width_other + safety_margin)
        z_overlap = abs(self.position.z - other.position.z) < (half_depth_self + half_depth_other + safety_margin)
        
        return y_overlap and x_overlap and z_overlap


@dataclass
class Store:
    """Represents a store location (can be on any floor of a building)"""
    id: int
    position: Position  # 3D position (x, floor_height, z)
    building_id: int  # Reference to parent building
    floor_number: int  # Which floor (0-indexed)
    
    def get_center(self) -> Position:
        """Get the center position of the store"""
        return self.position.copy()


@dataclass
class Customer:
    """Represents a customer location (can be on any floor of a building)"""
    id: int
    position: Position  # 3D position (x, floor_height, z)
    building_id: int  # Reference to parent building
    floor_number: int  # Which floor (0-indexed)
    
    def get_center(self) -> Position:
        """Get the center position of the customer"""
        return self.position.copy()


@dataclass
class Depot:
    """Represents a drone depot"""
    id: int
    position: Position
    drones: List['Drone']
    capacity: int = 5
    
    def __post_init__(self):
        if not hasattr(self, 'drones') or self.drones is None:
            self.drones = []

    def get_center(self) -> Position:
        """Get the center position of the depot in 3D space."""
        # For 3D, depot position is already at center
        # If depot has a size, we can offset appropriately
        return self.position.copy()

    def get_available_drones(self) -> List['Drone']:
        """Get list of available (idle) drones"""
        return [drone for drone in self.drones if drone.status == DroneStatus.IDLE]
    
    def assign_drone(self, order: 'Order') -> Optional['Drone']:
        """Assign an available drone with sufficient battery to an order"""
        available_drones = self.get_available_drones()
        if not available_drones:
            return None

        # Prefer drones with higher battery levels
        sorted_drones = sorted(available_drones, key=lambda d: d.battery_level, reverse=True)
        for drone in sorted_drones:
            if not order or drone.can_complete_order(order):
                drone.assign_order(order)
                return drone
        return None


@dataclass
class Drone:
    """Represents a delivery drone in 3D space"""
    id: int
    position: Position  # 3D position (x, y, z)
    depot: Depot
    status: DroneStatus = DroneStatus.IDLE
    current_order: Optional['Order'] = None
    current_orders: List['Order'] = None  # Multi-delivery: list of assigned orders
    route: List[Position] = None  # List of 3D waypoints
    route_waypoint_order_map: dict = None  # Maps waypoint index to (Order, visit_type)
    battery_level: float = 1.0  # 0.0 to 1.0
    speed: float = config.DRONE_SPEED  # horizontal speed (units per second)
    vertical_speed: float = getattr(config, 'DRONE_VERTICAL_SPEED', config.DRONE_SPEED * 0.5)  # vertical speed (units per second)
    collision_status: str = 'none'  # 'none', 'accidental', 'destination_entry'
    _waypoint_index: int = 0  # Internal counter for tracking current waypoint
    _service_time_remaining: float = 0.0  # Remaining service time at current stop (seconds)
    _current_service_type: str = None  # "store" or "customer" - type of current service operation
    
    def __post_init__(self):
        if self.current_orders is None:
            self.current_orders = []
        if self.route_waypoint_order_map is None:
            self.route_waypoint_order_map = {}
    
    def assign_order(self, order: 'Order'):
        """Assign an order to this drone"""
        self.current_order = order
        self.status = DroneStatus.LOADING
        order.status = OrderStatus.ASSIGNED
    
    def start_delivery(self, route: List[Position]):
        """Start delivery with given route"""
        if route and len(route) > 1:
            self.route = route
            self.status = DroneStatus.FLYING
            print(f"🚁 Drone {self.id}: Starting delivery with {len(route)} waypoints")
        else:
            print(f"❌ ERROR: Drone {self.id} received invalid route")
            self.route = None
            self.status = DroneStatus.IDLE
    
    def update_position(self, dt: float):
        """
        경로에 따라 드론 위치를 업데이트하고, 각 경유지에 도달할 때마다
        상태를 올바르게 변경합니다. (3D 이동 지원, 다중 배송 지원)
        서비스 시간이 있으면 해당 시간 동안 대기합니다.
        """
        # Handle service time waiting (pickup at store or delivery to customer)
        if self._service_time_remaining > 0:
            self._service_time_remaining -= dt
            if self._service_time_remaining <= 0:
                self._service_time_remaining = 0
                self._complete_service()
            return
        
        # 경로가 없거나 비어있으면 아무것도 하지 않습니다.
        if not self.route:
            return
        
        target = self.route[0]
        direction = Position(
            target.x - self.position.x,
            target.y - self.position.y,
            target.z - self.position.z
        )
        distance = self.position.distance_to(target)

        # 이미 목표 지점에 있거나 매우 가까운 경우 즉시 다음 waypoint로
        if distance < 0.1:
            self._handle_waypoint_arrival()
            return
        
        # 수평 및 수직 이동 속도 계산
        effective_speed = self.speed
        if distance > 0:
            # 전체 이동 거리 기준으로 이동
            move_distance = effective_speed * dt
            self.battery_level -= move_distance / config.DRONE_BATTERY_LIFE
            
            if distance < move_distance:
                # 목표 지점에 도착
                self.position = target.copy()
                self._handle_waypoint_arrival()
            else:
                # 목표 지점을 향해 이동합니다.
                ratio = move_distance / distance
                self.position.x += direction.x * ratio
                self.position.y += direction.y * ratio
                self.position.z += direction.z * ratio

    def _handle_waypoint_arrival(self):
        """waypoint 도착 시 상태 처리 (다중 배송 지원, 서비스 시간 적용)"""
        if not self.route:
            return
            
        self.route.pop(0)
        current_waypoint_idx = self._waypoint_index
        self._waypoint_index += 1
        
        # Skip service time for first waypoint (depot start) to avoid waiting at depot
        if current_waypoint_idx == 0:
            # First waypoint is the start position - just continue flying
            if not self.route:
                self._complete_delivery_route()
            return
        
        # 다중 배송 모드: waypoint_order_map 기반 처리
        if self.route_waypoint_order_map and current_waypoint_idx in self.route_waypoint_order_map:
            order, visit_type = self.route_waypoint_order_map[current_waypoint_idx]
            
            if visit_type == "store":
                # Start pickup service time
                pickup_time = getattr(config, 'PICKUP_SERVICE_TIME', 60.0)
                self._service_time_remaining = pickup_time
                self._current_service_type = "store"
                self.status = DroneStatus.PICKING_UP
                print(f"✈️  Drone {self.id}: Arrived at STORE for Order {order.id} - picking up ({pickup_time:.0f}s)")
                    
            elif visit_type == "customer":
                # Start delivery service time
                delivery_time = getattr(config, 'DELIVERY_SERVICE_TIME', 60.0)
                self._service_time_remaining = delivery_time
                self._current_service_type = "customer"
                self.status = DroneStatus.DROPPING_OFF
                print(f"📦 Drone {self.id}: Arrived at CUSTOMER for Order {order.id} - delivering ({delivery_time:.0f}s)")
        else:
            # 단일 배송 모드 (기존 로직 with service time)
            if self.status == DroneStatus.FLYING:
                # Start pickup service time
                pickup_time = getattr(config, 'PICKUP_SERVICE_TIME', 60.0)
                self._service_time_remaining = pickup_time
                self._current_service_type = "store"
                self.status = DroneStatus.PICKING_UP
                print(f"✈️  Drone {self.id}: Arrived at STORE - picking up ({pickup_time:.0f}s)")
            elif self.status == DroneStatus.DELIVERING:
                # Start delivery service time
                delivery_time = getattr(config, 'DELIVERY_SERVICE_TIME', 60.0)
                self._service_time_remaining = delivery_time
                self._current_service_type = "customer"
                self.status = DroneStatus.DROPPING_OFF
                print(f"📦 Drone {self.id}: Arrived at CUSTOMER - delivering ({delivery_time:.0f}s)")
        
        # 경로의 마지막 목적지에 도착했는지 확인 (단, 서비스 중이 아닐 때만)
        if not self.route and self._service_time_remaining <= 0:
            self._complete_delivery_route()
    
    def _complete_service(self):
        """Complete the current service operation (pickup or delivery)"""
        current_waypoint_idx = self._waypoint_index - 1  # We already incremented in _handle_waypoint_arrival
        
        # 다중 배송 모드: waypoint_order_map 기반 처리
        if self.route_waypoint_order_map and current_waypoint_idx in self.route_waypoint_order_map:
            order, visit_type = self.route_waypoint_order_map[current_waypoint_idx]
            
            if visit_type == "store":
                print(f"✅ Drone {self.id}: Pickup complete for Order {order.id}")
                order.status = OrderStatus.IN_PROGRESS
                self.status = DroneStatus.DELIVERING  # Continue to next waypoint
                    
            elif visit_type == "customer":
                print(f"✅ Drone {self.id}: Delivery complete for Order {order.id}")
                order.status = OrderStatus.COMPLETED
                if order in self.current_orders:
                    self.current_orders.remove(order)
                self.status = DroneStatus.FLYING  # Continue to next waypoint or returning
        else:
            # 단일 배송 모드
            if self._current_service_type == "store":
                print(f"✅ Drone {self.id}: Pickup complete")
                self.status = DroneStatus.DELIVERING
            elif self._current_service_type == "customer":
                print(f"✅ Drone {self.id}: Delivery complete")
                if self.current_order:
                    self.current_order.status = OrderStatus.COMPLETED
                self.status = DroneStatus.RETURNING
        
        self._current_service_type = None
        
        # 경로의 마지막 목적지였는지 확인
        if not self.route:
            self._complete_delivery_route()

    def _complete_delivery_route(self):
        """배송 경로 완료 처리"""
        # 다중 배송 모드: 모든 주문 완료 확인
        if self.current_orders:
            for order in self.current_orders:
                if order.status != OrderStatus.COMPLETED:
                    order.status = OrderStatus.COMPLETED
            print(f"✅ Drone {self.id}: All {len(self.current_orders)} orders COMPLETED")
            self.current_orders = []
        # 단일 배송 모드
        elif self.current_order:
            if self.current_order.status != OrderStatus.COMPLETED:
                self.current_order.status = OrderStatus.COMPLETED
            print(f"✅ Drone {self.id}: Order {self.current_order.id} COMPLETED")
            self.current_order = None
        
        # 상태 초기화
        self.status = DroneStatus.IDLE
        self.route_waypoint_order_map = {}
        self._waypoint_index = 0

    def can_complete_order(self, order: 'Order') -> bool:
        """Return True if current battery can finish depot->store->customer->depot trip."""
        if not order:
            return True

        required_distance = (
            self.position.distance_to(order.store_position) +
            order.store_position.distance_to(order.customer_position) +
            order.customer_position.distance_to(self.depot.get_center())
        )
        max_distance = self.battery_level * config.DRONE_BATTERY_LIFE
        return required_distance <= max_distance


@dataclass
class Motorbike:
    """Represents a delivery motorbike (ground-level, 2D movement)
    
    Key differences from Drone:
    - Moves only on ground level (y=0)
    - Service time depends on building height (rider must climb/use elevator)
    - No battery limitation (fuel assumed sufficient)
    - Tracks total distance for fuel cost calculation
    """
    id: int
    position: Position  # 2D position on ground (y should always be 0)
    depot: Depot
    status: DroneStatus = DroneStatus.IDLE  # Reuse DroneStatus for simplicity
    current_order: Optional['Order'] = None
    current_orders: List['Order'] = None  # Multi-delivery: list of assigned orders
    route: List[Position] = None  # List of 2D waypoints (ground level)
    route_waypoint_order_map: dict = None  # Maps waypoint index to (Order, visit_type)
    speed: float = None  # Ground speed (units per second)
    collision_status: str = 'none'
    _waypoint_index: int = 0
    _service_time_remaining: float = 0.0
    _current_service_type: str = None
    _current_target_floor: int = 0  # Floor number for current delivery (for service time calculation)
    total_distance_traveled: float = 0.0  # Track total distance for fuel cost
    
    # For compatibility with Drone interface
    battery_level: float = 1.0  # Always 1.0 (no battery limitation)
    
    def __post_init__(self):
        if self.current_orders is None:
            self.current_orders = []
        if self.route_waypoint_order_map is None:
            self.route_waypoint_order_map = {}
        if self.speed is None:
            self.speed = getattr(config, 'MOTORBIKE_SPEED', 8.0)
        # Ensure position is at ground level
        self.position.y = 0
    
    def assign_order(self, order: 'Order'):
        """Assign an order to this motorbike"""
        self.current_order = order
        self.status = DroneStatus.LOADING
        order.status = OrderStatus.ASSIGNED
    
    def start_delivery(self, route: List[Position]):
        """Start delivery with given route (ground level)"""
        if route and len(route) > 1:
            # Ensure all waypoints are at ground level for motorbike
            self.route = []
            for pos in route:
                ground_pos = pos.copy()
                ground_pos.y = 0  # Force ground level
                self.route.append(ground_pos)
            self.status = DroneStatus.FLYING
            print(f"🏍️ Motorbike {self.id}: Starting delivery with {len(self.route)} waypoints")
        else:
            print(f"❌ ERROR: Motorbike {self.id} received invalid route")
            self.route = None
            self.status = DroneStatus.IDLE
    
    def _calculate_service_time(self, floor_number: int) -> float:
        """Calculate service time based on floor number.
        
        Formula: base_time + (floor_number * time_per_floor)
        Ground floor (floor 0) has no additional time.
        """
        base_time = getattr(config, 'MOTORBIKE_BASE_SERVICE_TIME', 60.0)
        time_per_floor = getattr(config, 'MOTORBIKE_TIME_PER_FLOOR', 15.0)
        return base_time + (floor_number * time_per_floor)
    
    def _get_floor_from_position(self, position: Position) -> int:
        """Get floor number from a position's height."""
        floor_height = getattr(config, 'FLOOR_HEIGHT', 3.0)
        return int(position.y / floor_height)
    
    def _clamp_to_map_bounds(self, x: float, z: float) -> tuple:
        """Clamp position to stay within map boundaries."""
        margin = 5.0
        map_width = getattr(config, 'MAP_WIDTH', 2000)
        map_depth = getattr(config, 'MAP_DEPTH', 2000)
        
        clamped_x = max(margin, min(map_width - margin, x))
        clamped_z = max(margin, min(map_depth - margin, z))
        return clamped_x, clamped_z
    
    def update_position(self, dt: float):
        """Update motorbike position along route (ground-level 2D movement)."""
        # Handle service time waiting
        if self._service_time_remaining > 0:
            self._service_time_remaining -= dt
            if self._service_time_remaining <= 0:
                self._service_time_remaining = 0
                self._complete_service()
            return
        
        if not self.route:
            return
        
        target = self.route[0]
        # 2D movement only (ground level)
        direction = Position(
            target.x - self.position.x,
            0,  # No vertical movement
            target.z - self.position.z
        )
        distance = self.position.distance_to_2d(target)
        
        if distance < 0.1:
            self._handle_waypoint_arrival()
            return
        
        if distance > 0:
            move_distance = self.speed * dt
            self.total_distance_traveled += min(move_distance, distance)
            
            if distance < move_distance:
                # Clamp target to map bounds
                clamped_x, clamped_z = self._clamp_to_map_bounds(target.x, target.z)
                self.position.x = clamped_x
                self.position.z = clamped_z
                self.position.y = 0  # Stay on ground
                self._handle_waypoint_arrival()
            else:
                ratio = move_distance / distance
                new_x = self.position.x + direction.x * ratio
                new_z = self.position.z + direction.z * ratio
                # Clamp to map bounds
                self.position.x, self.position.z = self._clamp_to_map_bounds(new_x, new_z)
                self.position.y = 0  # Stay on ground
    
    def _handle_waypoint_arrival(self):
        """Handle arrival at waypoint with height-dependent service time."""
        if not self.route:
            return
        
        self.route.pop(0)
        current_waypoint_idx = self._waypoint_index
        self._waypoint_index += 1
        
        # Skip service time for first waypoint (depot start)
        if current_waypoint_idx == 0:
            if not self.route:
                self._complete_delivery_route()
            return
        
        # Multi-delivery mode
        if self.route_waypoint_order_map and current_waypoint_idx in self.route_waypoint_order_map:
            order, visit_type = self.route_waypoint_order_map[current_waypoint_idx]
            
            if visit_type == "store":
                # Calculate service time based on store floor
                floor_num = self._get_floor_from_position(order.store_position)
                service_time = self._calculate_service_time(floor_num)
                self._service_time_remaining = service_time
                self._current_service_type = "store"
                self._current_target_floor = floor_num
                self.status = DroneStatus.PICKING_UP
                print(f"🏍️ Motorbike {self.id}: Arrived at STORE (floor {floor_num}) for Order {order.id} - picking up ({service_time:.0f}s)")
                
            elif visit_type == "customer":
                # Calculate service time based on customer floor
                floor_num = self._get_floor_from_position(order.customer_position)
                service_time = self._calculate_service_time(floor_num)
                self._service_time_remaining = service_time
                self._current_service_type = "customer"
                self._current_target_floor = floor_num
                self.status = DroneStatus.DROPPING_OFF
                print(f"🏍️ Motorbike {self.id}: Arrived at CUSTOMER (floor {floor_num}) for Order {order.id} - delivering ({service_time:.0f}s)")
        else:
            # Single delivery mode
            if self.status == DroneStatus.FLYING:
                floor_num = 0
                if self.current_order:
                    floor_num = self._get_floor_from_position(self.current_order.store_position)
                service_time = self._calculate_service_time(floor_num)
                self._service_time_remaining = service_time
                self._current_service_type = "store"
                self.status = DroneStatus.PICKING_UP
                print(f"🏍️ Motorbike {self.id}: Arrived at STORE (floor {floor_num}) - picking up ({service_time:.0f}s)")
            elif self.status == DroneStatus.DELIVERING:
                floor_num = 0
                if self.current_order:
                    floor_num = self._get_floor_from_position(self.current_order.customer_position)
                service_time = self._calculate_service_time(floor_num)
                self._service_time_remaining = service_time
                self._current_service_type = "customer"
                self.status = DroneStatus.DROPPING_OFF
                print(f"🏍️ Motorbike {self.id}: Arrived at CUSTOMER (floor {floor_num}) - delivering ({service_time:.0f}s)")
        
        if not self.route and self._service_time_remaining <= 0:
            self._complete_delivery_route()
    
    def _complete_service(self):
        """Complete the current service operation."""
        current_waypoint_idx = self._waypoint_index - 1
        
        if self.route_waypoint_order_map and current_waypoint_idx in self.route_waypoint_order_map:
            order, visit_type = self.route_waypoint_order_map[current_waypoint_idx]
            
            if visit_type == "store":
                print(f"✅ Motorbike {self.id}: Pickup complete for Order {order.id}")
                order.status = OrderStatus.IN_PROGRESS
                self.status = DroneStatus.DELIVERING
            elif visit_type == "customer":
                print(f"✅ Motorbike {self.id}: Delivery complete for Order {order.id}")
                order.status = OrderStatus.COMPLETED
                if order in self.current_orders:
                    self.current_orders.remove(order)
                self.status = DroneStatus.FLYING
        else:
            if self._current_service_type == "store":
                print(f"✅ Motorbike {self.id}: Pickup complete")
                self.status = DroneStatus.DELIVERING
            elif self._current_service_type == "customer":
                print(f"✅ Motorbike {self.id}: Delivery complete")
                if self.current_order:
                    self.current_order.status = OrderStatus.COMPLETED
                self.status = DroneStatus.RETURNING
        
        self._current_service_type = None
        self._current_target_floor = 0
        
        if not self.route:
            self._complete_delivery_route()
    
    def _complete_delivery_route(self):
        """Complete the delivery route."""
        if self.current_orders:
            for order in self.current_orders:
                if order.status != OrderStatus.COMPLETED:
                    order.status = OrderStatus.COMPLETED
            print(f"✅ Motorbike {self.id}: All {len(self.current_orders)} orders COMPLETED")
            self.current_orders = []
        elif self.current_order:
            if self.current_order.status != OrderStatus.COMPLETED:
                self.current_order.status = OrderStatus.COMPLETED
            print(f"✅ Motorbike {self.id}: Order {self.current_order.id} COMPLETED")
            self.current_order = None
        
        self.status = DroneStatus.IDLE
        self.route_waypoint_order_map = {}
        self._waypoint_index = 0
    
    def can_complete_order(self, order: 'Order') -> bool:
        """Motorbike always returns True (no range limitation by default)."""
        range_limit = getattr(config, 'MOTORBIKE_RANGE_LIMIT', None)
        if range_limit is None:
            return True
        
        required_distance = (
            self.position.distance_to_2d(order.store_position) +
            order.store_position.distance_to_2d(order.customer_position) +
            order.customer_position.distance_to_2d(self.depot.get_center())
        )
        return required_distance <= range_limit


# Type alias for vehicle (either Drone or Motorbike)
Vehicle = Drone  # Will be set dynamically based on config


@dataclass
class Order:
    """Represents a food delivery order"""
    id: int
    customer_id: int
    store_id: int
    customer_position: Position
    store_position: Position
    created_time: float
    status: OrderStatus = OrderStatus.PENDING
    assigned_drone: Optional[Drone] = None
    estimated_delivery_time: Optional[float] = None
    store_building_id: Optional[int] = None  # ID of building containing the store
    customer_building_id: Optional[int] = None  # ID of building containing the customer
    
    def get_distance(self) -> float:
        """Calculate distance between store and customer"""
        return self.store_position.distance_to(self.customer_position)
    
    def is_expired(self, current_time: float) -> bool:
        """Check if order has expired based on maximum wait time"""
        return current_time - self.created_time > 300  # 5 minutes max wait


class Map:
    """Represents the 3D urban environment map"""
    
    def __init__(self, width: float, depth: float, max_height: float = 100):
        """
        Initialize 3D map
        Args:
            width: Size along x-axis
            depth: Size along z-axis  
            max_height: Maximum height along y-axis
        """
        self.width = width
        self.depth = depth
        self.max_height = max_height
        self.buildings: List[Building] = []
        self.depots: List[Depot] = []
        self.stores: List['Store'] = []  # Store objects on various floors
        self.customers: List['Customer'] = []  # Customer objects on various floors
        self.tree = None
    
    def add_building(self, building: Building):
        """Add a building to the map"""
        self.buildings.append(building)
    
    def add_store(self, store: 'Store'):
        """Add a store to the map"""
        self.stores.append(store)
    
    def add_customer(self, customer: 'Customer'):
        """Add a customer to the map"""
        self.customers.append(customer)
    
    def add_depot(self, depot: Depot):
        """Add a depot to the map"""
        self.depots.append(depot)

    def build_tree(self):
        """Build spatial index tree and ID-to-index lookup for buildings."""
        polys = []
        self.building_id_to_index = {}  # Maps building.id to list index
        for idx, building in enumerate(self.buildings):
            polys.append(building.inner_poly)
            self.building_id_to_index[building.id] = idx
        self.tree = STRtree(polys)

    def get_building_containing_point(self, point: Position) -> Optional[Building]:
        """Return building containing the given 3D point (polygon-based when available)."""
        p = Point(point.x, point.z)
        candidate_polys = self.tree.query(p)
    
        for poly_index in candidate_polys:
            building = self.buildings[poly_index]

            y_min, y_max = building._vertical_bounds()
            if not (y_min <= point.y <= y_max):
                continue

            poly = building.inner_poly
            if poly and poly.covers(p):
                return building

        return None # 어떤 건물에도 포함되지 않음

    def get_building_at_position(self, pos: Position) -> Optional[Building]:
        """Get building at a specific 3D position"""
        for building in self.buildings:
            if building.contains_point(pos):
                return building
        return None
    
    def is_position_valid(self, pos: Position, width: float = 0, height: float = 0, depth: float = 0,
                          safety_margin: float = 0.0) -> bool:
        """Check if a 3D position is valid (not overlapping with buildings)
        
        Args:
            pos: Center position of the object
            width: Size along x-axis
            height: Size along y-axis (vertical)
            depth: Size along z-axis
            safety_margin: Additional safety distance to maintain from other buildings
        """
        # Check bounds (assuming pos is center)
        half_width = width / 2
        half_depth = depth / 2
        
        if (pos.x - half_width < 0 or pos.x + half_width > self.width or 
            pos.z - half_depth < 0 or pos.z + half_depth > self.depth or
            pos.y < 0 or pos.y > self.max_height):
            return False
        
        # Check building collisions (with safety margin)
        if width > 0 and height > 0 and depth > 0:
            test_building = Building(0, pos, width, height, depth)
            for building in self.buildings:
                if test_building.collides_with(building, safety_margin):
                    return False
        
        return True
    
    def get_random_valid_position(self, width: float, height: float, depth: float, 
                                  max_attempts: int = 100, safety_margin: float = 0.0) -> Optional[Position]:
        """Get a random valid 3D position for placing entities
        
        Args:
            width: Size along x-axis
            height: Size along y-axis (vertical)
            depth: Size along z-axis
            max_attempts: Maximum number of placement attempts
            safety_margin: Additional safety distance to maintain from other buildings
        """
        for _ in range(max_attempts):
            # Random position on ground plane (y = height/2 to center the building)
            x = random.uniform(width/2, self.width - width/2)
            z = random.uniform(depth/2, self.depth - depth/2)
            y = height / 2  # Center vertically, ground at y=0
            
            pos = Position(x, y, z)
            
            if self.is_position_valid(pos, width, height, depth, safety_margin):
                return pos
        
        return None
