"""
Order management system for dynamic order generation and depot assignment (3D)
"""

import random
import time
import math
from typing import List, Dict, Optional, Tuple, Callable
# (수정) DroneStatus 임포트
from ..models.entities import (
    Order, Depot, Drone, OrderStatus, DroneStatus, Position, Store, Customer
)
from .clustering import MixedClustering
from .routing import DroneRouteOptimizer, SimpleRouting, MultiLevelAStarRouting, RouteValidator
from .multi_delivery.insertion_heuristic import InsertionHeuristicStrategy
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


class DepotSelector:
    
    def __init__(self, map_obj):
        self.map = map_obj
        self.clustering_algorithm = MixedClustering()
    
    def select_best_depot(self, order: Order) -> Optional[Depot]:
        if not self.map.depots:
            return None
        
        scored = self.get_scored_depots(order)
        return scored[0][0] if scored else None

    def get_scored_depots(self, order: Order) -> List[Tuple[Depot, float]]:
        """Return depots sorted by suitability score (desc)."""
        results = []
        for depot in self.map.depots:
            score = self._calculate_depot_score(depot, order)
            results.append((depot, score))
        # Sort high score first
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _calculate_depot_score(self, depot: Depot, order: Order) -> float:
        available_drones = depot.get_available_drones()
        if not available_drones:
            return 0.0
        
        depot_to_store_distance = depot.get_center().distance_to(order.store_position)
        
        store_to_customer_distance = order.get_distance()
        
        total_distance = depot_to_store_distance + store_to_customer_distance
        
        customer_to_depot_distance = order.customer_position.distance_to(depot.get_center())
        
        complete_trip_distance = total_distance + customer_to_depot_distance
        
        drone_bonus = len(available_drones) * 10
        
        distance_score = max(0, 1000 - complete_trip_distance)
        
        battery_score = 100
        
        total_score = distance_score + drone_bonus + battery_score
        
        return total_score
    
    def get_depot_assignment_metrics(self, orders: List[Order]) -> Dict[int, Dict]:
        depot_stats = {}
        
        for depot in self.map.depots:
            depot_stats[depot.id] = {
                'total_orders': 0,
                'available_drones': len(depot.get_available_drones()),
                'total_drones': len(depot.drones),
                'avg_distance': 0.0,
                'total_distance': 0.0
            }
        
        for order in orders:
            if order.assigned_drone and order.assigned_drone.depot:
                depot_id = order.assigned_drone.depot.id
                if depot_id in depot_stats:
                    depot_stats[depot_id]['total_orders'] += 1
                    
                    depot = order.assigned_drone.depot
                    distance = (depot.position.distance_to(order.store_position) + 
                                order.store_position.distance_to(order.customer_position) +
                                order.customer_position.distance_to(depot.position))
                    
                    depot_stats[depot_id]['total_distance'] += distance
        
        for depot_id, stats in depot_stats.items():
            if stats['total_orders'] > 0:
                stats['avg_distance'] = stats['total_distance'] / stats['total_orders']
        
        return depot_stats


class OrderManager:
    
    def __init__(self, map_obj, seed: Optional[int] = None):
        self.map = map_obj
        self.orders: List[Order] = []
        self.completed_orders: List[Order] = []
        self.order_generator = OrderGenerator(map_obj, seed=seed)
        self.depot_selector = DepotSelector(map_obj)
        self.route_connectivity_cache: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        self.connectivity_cache_ttl = getattr(config, "ROUTE_CONNECTIVITY_CACHE_TTL", 300.0)
        
        print("  Initializing 3D routing...")
        self.route_optimizer = DroneRouteOptimizer(MultiLevelAStarRouting(map_obj, k_levels=3))
        
        # 다중 배송 전략 초기화
        self.use_multi_delivery = config.DRONE_CAPACITY > 1
        if self.use_multi_delivery:
            print(f"  Multi-delivery mode enabled (capacity: {config.DRONE_CAPACITY})")
            self.insertion_strategy = InsertionHeuristicStrategy(self.route_optimizer, map_obj)
        else:
            print("  Single-delivery mode")
            self.insertion_strategy = None
        
        # 시각화 설정
        self.first_route_visualized = False
        self.route_failure_handlers: List[Callable[[Order, str], None]] = []

    def register_route_failure_handler(self, handler: Callable[[Order, str], None]):
        """Allow external systems (simulation engine) to handle route failures."""
        if handler not in self.route_failure_handlers:
            self.route_failure_handlers.append(handler)
    
    def process_orders(self, current_time: float) -> List[Order]:
        new_order = self.order_generator.generate_random_order(current_time)
        if new_order:
            self.orders.append(new_order)
            print(f"📝 New order #{new_order.id}: Customer {new_order.customer_id} -> Store {new_order.store_id}")
        
        new_completed = []
        for order in self.orders[:]:
            if order.status == OrderStatus.PENDING:
                self._assign_order_to_depot(order, current_time)
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
        self._assign_order_to_depot(order, current_time)

    def _notify_route_failure(self, order: Order, reason: str):
        for handler in self.route_failure_handlers:
            try:
                handler(order, reason)
            except Exception as exc:
                print(f"Route failure handler error: {exc}")
    
    def _assign_order_to_depot(self, order: Order, current_time: Optional[float] = None):
        if current_time is None:
            current_time = time.time()
        
        # 다중 배송 모드: 삽입 휴리스틱 우선 시도
        if self.use_multi_delivery and self.insertion_strategy:
            success = self.insertion_strategy.assign_order(order, current_time)
            if success:
                print(f"✓ Order {order.id} assigned via multi-delivery insertion heuristic")
                return
            # 실패 시 기존 단일 배송 로직으로 폴백
            print(f"⚠️ Order {order.id}: Multi-delivery insertion failed, trying single-delivery fallback")
        
        candidates = self.depot_selector.get_scored_depots(order)
        if not candidates:
            return

        for depot, score in candidates:
            if self._is_cached_unreachable(depot, order, current_time):
                continue

            assigned_drone = depot.assign_drone(order)
            if not assigned_drone:
                continue

            order.assigned_drone = assigned_drone
            order.status = OrderStatus.ASSIGNED

            try:
                # 시각화 옵션 결정
                should_visualize = False
                if config.VISUALIZE_ALL_ROUTES:
                    should_visualize = True
                elif config.VISUALIZE_FIRST_ROUTE and not self.first_route_visualized:
                    should_visualize = True
                    self.first_route_visualized = True
                    print(f"\n{'='*60}\n📊 Visualizing first delivery route\n{'='*60}")
                
                route = self.route_optimizer.optimize_delivery_route(assigned_drone, order, visualize=should_visualize)
                
                # (수정) 경로 탐색 실패 시 다른 depot 시도
                if not route or len(route) < 2:
                    reason = "fail to find route"
                    print(f"❌ Order {order.id}: Route calculation FAILED (Depot {depot.id})")
                    self._release_drone(assigned_drone, order)
                    self._notify_route_failure(order, reason)
                    self._mark_connectivity(depot, order, reachable=False, current_time=current_time)
                    continue

                is_valid, reason, message = RouteValidator.validate_route_feasibility(route, assigned_drone)
                if not is_valid:
                    print(f"❌ Order {order.id}: {message}")
                    if reason == "time_limit":
                        order.status = OrderStatus.CANCELLED
                        self._release_drone(assigned_drone, order)
                        self._notify_route_failure(order, "time_limit")
                        if order in self.orders:
                            self.orders.remove(order)
                        return
                    else:
                        self._release_drone(assigned_drone, order)
                        order.status = OrderStatus.PENDING
                        self._notify_route_failure(order, reason or "route_invalid")
                        self._mark_connectivity(depot, order, reachable=False, current_time=current_time)
                        continue

                # 성공
                self._mark_connectivity(depot, order, reachable=True, current_time=current_time)
                assigned_drone.start_delivery(route)
                print(f"✓ Order {order.id} assigned to Depot {depot.id}, Drone {assigned_drone.id}")
                return

            except Exception as e:
                reason = "exception"
                print(f"Failed to set route for order {order.id}: {e}")
                import traceback
                traceback.print_exc()
                self._release_drone(assigned_drone, order)
                order.status = OrderStatus.PENDING
                self._notify_route_failure(order, reason)
                self._mark_connectivity(depot, order, reachable=False, current_time=current_time)
                continue

        # 모든 depot 시도 후 실패하면 주문은 PENDING 상태로 남음
        order.status = OrderStatus.PENDING
        order.assigned_drone = None

    def _release_drone(self, drone: Drone, order: Order):
        """Reset drone/order linkage after failed assignment."""
        if not drone:
            return
        drone.current_order = None
        drone.status = DroneStatus.IDLE
        drone.route = None
        order.assigned_drone = None
    
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
            self._assign_order_to_depot(order, current_time)
        
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

    def _mark_connectivity(self, depot: Depot, order: Order, reachable: bool, current_time: float):
        key = (depot.id, order.store_building_id or -1, order.customer_building_id or -1)
        self.route_connectivity_cache[key] = {"reachable": reachable, "timestamp": current_time}

    def _is_cached_unreachable(self, depot: Depot, order: Order, current_time: float) -> bool:
        key = (depot.id, order.store_building_id or -1, order.customer_building_id or -1)
        record = self.route_connectivity_cache.get(key)
        if not record:
            return False
        if current_time - record["timestamp"] > self.connectivity_cache_ttl:
            return False
        return record.get("reachable") is False


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
