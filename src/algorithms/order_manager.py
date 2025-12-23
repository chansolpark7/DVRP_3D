"""
Order management system for dynamic order generation and depot assignment (3D)
"""

import random
import time
from typing import List, Dict, Optional, Tuple, Callable
from ..models.entities import (
    Order, Depot, Drone, OrderStatus, DroneStatus, Position, Store, Customer
)
from .clustering import MixedClustering
from .routing import MultiLevelAStarRouting, MotorbikeRouting
from .insertion_strategy import InsertionStrategy
import config


class OrderGenerator:

    def __init__(self, map_obj, generation_rate: float = config.ORDER_GENERATION_RATE, seed: Optional[int] = None):
        self.map = map_obj
        self.generation_rate = generation_rate
        self.order_counter = 0
        self.last_generation_time = 0.0
        self.seed = seed
        if self.seed is not None:
            print(f"Initializing order generator with fixed seed: {self.seed}")
            random.seed(self.seed)
    
    def generate_random_order(self, current_time: float) -> Optional[Order]:
        time_since_last = current_time - self.last_generation_time
        
        probability = self.generation_rate * time_since_last
        
        if random.random() < probability:
            self.last_generation_time = current_time
            return self._create_random_order(current_time)
        
        return None
    
    def _create_random_order(self, current_time: float) -> Order:
        customers = self.map.customers
        stores = self.map.stores
        
        if not customers or not stores:
            return None
        
        customer_entity = random.choice(customers)
        store_entity = random.choice(stores)
        
        order = Order(
            id=self.order_counter,
            customer_id=customer_entity.id,
            store_id=store_entity.id,
            customer_position=customer_entity.get_center(),
            store_position=store_entity.get_center(),
            created_time=current_time,
            store_building_id=store_entity.building_id,
            customer_building_id=customer_entity.building_id
        )
        
        self.order_counter += 1
        return order
    
    def generate_batch_orders(self, current_time: float, batch_size: int = 5) -> List[Order]:
        orders = []
        for _ in range(batch_size):
            order = self._create_random_order(current_time)
            if order:
                orders.append(order)
        return orders


class OrderManager:
    """Unified order management with integrated insertion strategy.
    
    Handles all drone assignments through a single InsertionStrategy that
    fairly compares inserting into existing routes vs starting new routes.
    """
    
    def __init__(self, map_obj, seed: Optional[int] = None):
        self.map = map_obj
        self.orders: List[Order] = []
        self.completed_orders: List[Order] = []
        self.order_generator = OrderGenerator(map_obj, seed=seed)
        self.route_connectivity_cache: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        self.connectivity_cache_ttl = getattr(config, "ROUTE_CONNECTIVITY_CACHE_TTL", 300.0)
        
        # Check simulation mode
        self.simulation_mode = getattr(config, 'SIMULATION_MODE', 'drone')
        
        # Initialize routing algorithm based on simulation mode
        if self.simulation_mode == "motorbike":
            print("  Initializing 2D ground-level routing for motorbikes...")
            self.routing_algorithm = MotorbikeRouting(map_obj)
            vehicle_capacity = getattr(config, 'MOTORBIKE_CAPACITY', 3)
            print(f"  Unified insertion strategy enabled (motorbike capacity: {vehicle_capacity})")
        else:
            print("  Initializing 3D routing for drones...")
            self.routing_algorithm = MultiLevelAStarRouting(map_obj, k_levels=3)
            print(f"  Unified insertion strategy enabled (drone capacity: {config.DRONE_CAPACITY})")
        
        # Initialize unified insertion strategy
        self.insertion_strategy = InsertionStrategy(self.routing_algorithm, map_obj)
        
        self.route_failure_handlers: List[Callable[[Order, str], None]] = []

    def register_route_failure_handler(self, handler: Callable[[Order, str], None]):
        """Allow external systems (simulation engine) to handle route failures."""
        if handler not in self.route_failure_handlers:
            self.route_failure_handlers.append(handler)
    
    def process_orders(self, current_time: float) -> List[Order]:
        """Process orders: generate new ones and assign pending ones."""
        new_order = self.order_generator.generate_random_order(current_time)
        if new_order:
            self.orders.append(new_order)
            print(f"📝 New order #{new_order.id}: Customer {new_order.customer_id} -> Store {new_order.store_id}")
        
        new_completed = []
        for order in self.orders[:]:
            if order.status == OrderStatus.PENDING:
                self._assign_order(order, current_time)
            elif order.status == OrderStatus.COMPLETED:
                self.orders.remove(order)
                self.completed_orders.append(order)
                new_completed.append(order)
            elif order.status == OrderStatus.CANCELLED:
                self.orders.remove(order)
            elif order.is_expired(current_time):
                order.status = OrderStatus.CANCELLED
                self.orders.remove(order)
                print(f"Order {order.id} expired and cancelled")
        
        return new_completed
    
    def retry_order_assignment(self, order: Order, current_time: Optional[float] = None):
        """Expose manual retry hook for failed orders."""
        if order.status == OrderStatus.CANCELLED:
            return
        if current_time is None:
            current_time = time.time()
        self._assign_order(order, current_time)

    def _notify_route_failure(self, order: Order, reason: str):
        for handler in self.route_failure_handlers:
            try:
                handler(order, reason)
            except Exception as exc:
                print(f"Route failure handler error: {exc}")
    
    def _assign_order(self, order: Order, current_time: float):
        """Assign order using unified insertion strategy.
        
        The strategy automatically compares:
        - Inserting into existing drone routes (cost delta)
        - Starting new routes with idle drones (absolute cost)
        
        And selects the best option.
        """
        success = self.insertion_strategy.assign_order(order, current_time)
        
        if success:
            print(f"✓ Order {order.id} assigned successfully")
        else:
            # Assignment failed - order remains PENDING for retry
            order.status = OrderStatus.PENDING
            order.assigned_drone = None
            self._notify_route_failure(order, "no_valid_assignment")
    
    def get_order_statistics(self) -> Dict:
        total_orders = len(self.orders) + len(self.completed_orders)
        pending_orders = len([o for o in self.orders if o.status == OrderStatus.PENDING])
        assigned_orders = len([o for o in self.orders if o.status == OrderStatus.ASSIGNED])
        in_progress_orders = len([o for o in self.orders if o.status == OrderStatus.IN_PROGRESS])
        completed_orders = len(self.completed_orders)
        
        avg_delivery_distance = 0
        if completed_orders > 0:
            total_distance = sum(o.get_distance() for o in self.completed_orders)
            avg_delivery_distance = total_distance / completed_orders
        
        return {
            'total_orders': total_orders,
            'pending_orders': pending_orders,
            'assigned_orders': assigned_orders,
            'in_progress_orders': in_progress_orders,
            'completed_orders': completed_orders,
            'avg_delivery_distance': avg_delivery_distance
        }
    
    def generate_test_orders(self, num_orders: int = 10, current_time: float = None) -> List[Order]:
        if current_time is None:
            current_time = time.time()
        
        test_orders = self.order_generator.generate_batch_orders(current_time, num_orders)
        self.orders.extend(test_orders)
        
        print(f"Generated {len(test_orders)} test orders")
        return test_orders
    
    def assign_all_pending_orders(self, current_time: Optional[float] = None):
        if current_time is None:
            current_time = time.time()
        pending_orders = [o for o in self.orders if o.status == OrderStatus.PENDING]
        
        for order in pending_orders:
            self._assign_order(order, current_time)
        
        print(f"Assigned {len(pending_orders)} pending orders with routes")
    
    def get_depot_load_balancing_info(self) -> Dict:
        depot_info = {}
        
        for depot in self.map.depots:
            total_drones = len(depot.drones)
            available_drones = len(depot.get_available_drones())
            busy_drones = total_drones - available_drones
            
            utilization_rate = 0.0
            if total_drones > 0:
                utilization_rate = (busy_drones / total_drones) * 100
            
            depot_info[depot.id] = {
                'total_drones': total_drones,
                'available_drones': available_drones,
                'busy_drones': busy_drones,
                'utilization_rate': utilization_rate
            }
        
        return depot_info


class OrderValidator:
    
    @staticmethod
    def validate_order_feasibility(order: Order, drone: Drone) -> bool:
        total_distance = (drone.position.distance_to(order.store_position) +
                          order.store_position.distance_to(order.customer_position) +
                          order.customer_position.distance_to(drone.depot.get_center()))
        
        max_distance = drone.battery_level * config.DRONE_BATTERY_LIFE
        
        return total_distance <= max_distance
    
    @staticmethod
    def validate_order_constraints(order: Order) -> bool:
        if order.customer_id == order.store_id:
            return False
        
        if (order.customer_position.x < 0 or order.customer_position.x > config.MAP_WIDTH or
            order.customer_position.y < 0 or order.customer_position.y > config.MAP_HEIGHT or
            order.store_position.x < 0 or order.store_position.x > config.MAP_WIDTH or
            order.store_position.y < 0 or order.store_position.y > config.MAP_HEIGHT):
            return False
        
        return True
