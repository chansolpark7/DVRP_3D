"""
Unified insertion strategy for drone delivery routing.

Handles both new drone assignments and insertions into existing routes,
comparing them fairly using:
- For IDLE drones: absolute distance (new route cost)
- For busy drones: cost delta (additional distance from insertion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union, TYPE_CHECKING

import config
from ..models.entities import (
    Drone,
    Motorbike,
    DroneStatus,
    Order,
    OrderStatus,
    Position,
)
from .delivery_types import Route, Visit

if TYPE_CHECKING:
    from .routing import RoutingAlgorithm

# Type alias for vehicle (drone or motorbike)
Vehicle = Union[Drone, Motorbike]


# Optimization parameters
MAX_DEPOT_DISTANCE = getattr(config, 'INSERTION_MAX_DEPOT_DISTANCE', 2000)
GOOD_ENOUGH_THRESHOLD = getattr(config, 'INSERTION_GOOD_ENOUGH_THRESHOLD', 50)
TOP_K_CANDIDATES = getattr(config, 'INSERTION_TOP_K_CANDIDATES', 5)


class RouteEvaluator:
    """Provides fast estimates vs. exact route generation."""

    def __init__(self, routing_algorithm: 'RoutingAlgorithm'):
        self.routing_algorithm = routing_algorithm

    def estimate_route_cost(self, route: Route) -> float:
        """Return straight-line distance as a fast proxy cost."""
        return route.get_total_distance_estimate()

    def estimate_battery_usage(self, route: Route) -> float:
        """Return conservative battery usage estimate."""
        return route.get_battery_required_estimate()

    def check_battery_feasibility(self, route: Route) -> bool:
        """Return True if the vehicle has enough battery/fuel for the route.
        
        Motorbikes have no battery limitation (always returns True).
        Drones check against battery level.
        """
        # Motorbikes have no battery limitation
        if isinstance(route.drone, Motorbike):
            # Check range limit if configured
            range_limit = getattr(config, 'MOTORBIKE_RANGE_LIMIT', None)
            if range_limit is None:
                return True
            required = self.estimate_battery_usage(route)
            return required <= range_limit
        
        # Drone battery check
        required = self.estimate_battery_usage(route)
        available = route.drone.battery_level * config.DRONE_BATTERY_LIFE
        return required <= available

    def calculate_exact_route(self, route: Route) -> List[Position]:
        """Generate the full waypoint list using the configured routing algorithm."""
        if not route.start_position or not route.depot_position:
            return []

        waypoints = [visit.position for visit in route.visits]
        
        exact_route = self.routing_algorithm.calculate_route(
            route.start_position,
            waypoints,
            route.depot_position,
        )
        
        return exact_route


@dataclass
class InsertionCandidate:
    """Represents a candidate for order insertion."""
    drone: Drone
    store_index: int
    customer_index: int
    comparable_cost: float  # Cost used for comparison (delta for busy, absolute for idle)
    route: Route
    is_new_drone: bool  # True if this is an IDLE drone getting a new route


@dataclass
class LightweightCandidate:
    """Lightweight candidate for fast sorting before full validation."""
    drone: Drone
    store_index: int
    customer_index: int
    comparable_cost: float
    base_route: Route
    is_new_drone: bool


class InsertionStrategy:
    """Unified strategy for assigning orders to drones.
    
    Compares all options fairly:
    - Inserting into existing routes: uses cost_delta (additional distance)
    - New drone assignment: uses absolute distance
    
    This allows fair comparison between "adding to existing route" vs "starting new route".
    """

    def __init__(self, routing_algorithm: 'RoutingAlgorithm', map_obj):
        self.routing_algorithm = routing_algorithm
        self.map = map_obj
        self.route_evaluator = RouteEvaluator(routing_algorithm)

    def assign_order(self, order: Order, current_time: float) -> bool:
        """Attempt to assign an order using unified insertion strategy.
        
        Considers both inserting into existing routes and assigning to idle drones,
        selecting the option with lowest cost.
        
        Returns True if successfully assigned, False otherwise.
        """
        candidates = self._generate_all_candidates(order, current_time)
        if not candidates:
            return False

        # Select best candidate (lowest comparable cost)
        best = min(candidates, key=lambda c: c.comparable_cost)
        return self._apply_insertion(best, order, current_time)

    def _generate_all_candidates(self, order: Order, current_time: float) -> List[InsertionCandidate]:
        """Generate all possible insertion candidates with fair cost comparison."""
        
        store_pos = order.store_position
        customer_pos = order.customer_position
        
        # Collect all candidate drones
        drones = self._collect_candidate_drones(order)
        if not drones:
            return []
        
        lightweight_candidates: List[LightweightCandidate] = []
        
        for drone in drones:
            base_route = self._build_route_from_drone(drone, current_time)
            base_positions = [v.position for v in base_route.visits]
            
            # Determine if this is a new drone (IDLE with no existing orders)
            is_new_drone = (drone.status == DroneStatus.IDLE) and len(base_positions) == 0
            
            # Calculate base cost (0 for new drones)
            if is_new_drone:
                base_cost = 0.0
            else:
                base_cost = self._calculate_distance_fast(
                    base_route.start_position, base_positions, base_route.depot_position
                )
            
            n = len(base_positions)
            
            for store_idx in range(n + 1):
                for customer_idx in range(store_idx + 1, n + 2):
                    # Calculate new route cost after insertion
                    new_cost = self._calculate_insertion_cost(
                        base_route.start_position,
                        base_positions,
                        base_route.depot_position,
                        store_pos, customer_pos,
                        store_idx, customer_idx
                    )
                    
                    # Fair comparison:
                    # - New drone: absolute cost (entire new route)
                    # - Existing drone: delta cost (additional distance)
                    if is_new_drone:
                        comparable_cost = new_cost  # Absolute distance
                    else:
                        comparable_cost = new_cost - base_cost  # Delta (additional distance)
                    
                    lightweight_candidates.append(LightweightCandidate(
                        drone=drone,
                        store_index=store_idx,
                        customer_index=customer_idx,
                        comparable_cost=comparable_cost,
                        base_route=base_route,
                        is_new_drone=is_new_drone,
                    ))
                    
                    # Early termination for very good candidates
                    if comparable_cost < GOOD_ENOUGH_THRESHOLD:
                        full_route = self._create_route_with_insertion(
                            base_route, order, store_idx, customer_idx
                        )
                        is_valid, _ = full_route.is_valid(current_time)
                        if is_valid:
                            return [InsertionCandidate(
                                drone=drone,
                                store_index=store_idx,
                                customer_index=customer_idx,
                                comparable_cost=comparable_cost,
                                route=full_route,
                                is_new_drone=is_new_drone,
                            )]
        
        # Sort by comparable cost and validate top K
        lightweight_candidates.sort(key=lambda c: c.comparable_cost)
        top_candidates = lightweight_candidates[:TOP_K_CANDIDATES]
        
        validated_candidates: List[InsertionCandidate] = []
        for lc in top_candidates:
            full_route = self._create_route_with_insertion(
                lc.base_route, order, lc.store_index, lc.customer_index
            )
            is_valid, _ = full_route.is_valid(current_time)
            if is_valid:
                validated_candidates.append(InsertionCandidate(
                    drone=lc.drone,
                    store_index=lc.store_index,
                    customer_index=lc.customer_index,
                    comparable_cost=lc.comparable_cost,
                    route=full_route,
                    is_new_drone=lc.is_new_drone,
                ))
        
        return validated_candidates

    def _collect_candidate_drones(self, order: Order) -> List[Vehicle]:
        """Collect all candidate vehicles (drones or motorbikes, both IDLE and with capacity)."""
        vehicles: List[Vehicle] = []
        simulation_mode = getattr(config, 'SIMULATION_MODE', 'drone')
        
        for depot in self.map.depots:
            # Distance filter: skip depots too far from the order
            # Use 2D distance for motorbikes
            if simulation_mode == "motorbike":
                depot_to_store = depot.get_center().distance_to_2d(order.store_position)
            else:
                depot_to_store = depot.get_center().distance_to(order.store_position)
            
            if depot_to_store > MAX_DEPOT_DISTANCE:
                continue
            
            for vehicle in depot.drones:  # 'drones' field holds either drones or motorbikes
                if vehicle.status == DroneStatus.IDLE or self._has_capacity(vehicle):
                    vehicles.append(vehicle)
        
        return vehicles

    def _has_capacity(self, vehicle: Vehicle) -> bool:
        """Check if vehicle has capacity for additional orders."""
        # Get capacity based on vehicle type
        if isinstance(vehicle, Motorbike):
            capacity = getattr(config, 'MOTORBIKE_CAPACITY', 3)
        else:
            capacity = config.DRONE_CAPACITY
        
        if hasattr(vehicle, "current_orders") and vehicle.current_orders:
            return len(vehicle.current_orders) < capacity
        return vehicle.current_order is None
    
    def _calculate_distance_fast(self, start: Position, waypoints: List[Position], end: Position) -> float:
        """Calculate total straight-line distance."""
        if not waypoints:
            return start.distance_to(end)
        
        total = start.distance_to(waypoints[0])
        for i in range(len(waypoints) - 1):
            total += waypoints[i].distance_to(waypoints[i + 1])
        total += waypoints[-1].distance_to(end)
        return total
    
    def _calculate_insertion_cost(
        self, 
        start: Position, 
        positions: List[Position], 
        end: Position,
        store_pos: Position, 
        customer_pos: Position,
        store_idx: int, 
        customer_idx: int
    ) -> float:
        """Calculate route cost after insertion."""
        new_positions = (
            positions[:store_idx] + 
            [store_pos] + 
            positions[store_idx:customer_idx - 1] + 
            [customer_pos] + 
            positions[customer_idx - 1:]
        )
        return self._calculate_distance_fast(start, new_positions, end)
    
    def _create_route_with_insertion(
        self, 
        base_route: Route, 
        order: Order, 
        store_idx: int, 
        customer_idx: int
    ) -> Route:
        """Create a new route with the order inserted at specified positions."""
        store_visit = Visit(
            order_id=order.id,
            position=order.store_position.copy(),
            visit_type="store",
            order=order,
        )
        customer_visit = Visit(
            order_id=order.id,
            position=order.customer_position.copy(),
            visit_type="customer",
            order=order,
        )
        
        new_visits = list(base_route.visits)
        new_visits.insert(store_idx, store_visit)
        new_visits.insert(customer_idx, customer_visit)
        
        return Route(
            drone_id=base_route.drone_id,
            drone=base_route.drone,
            visits=new_visits,
            start_position=base_route.start_position,
            depot_position=base_route.depot_position,
            start_time=base_route.start_time,
        )

    def _build_route_from_drone(self, drone: Drone, current_time: float) -> Route:
        """Build current route state from drone."""
        route = Route(
            drone_id=drone.id,
            drone=drone,
            visits=[],
            start_position=drone.position.copy(),
            depot_position=drone.depot.get_center().copy(),
            start_time=current_time,
        )

        if drone.status == DroneStatus.IDLE or not drone.route:
            return route

        # Rebuild visit list from current_orders
        if hasattr(drone, "current_orders") and drone.current_orders:
            for order in drone.current_orders:
                route.visits.append(
                    Visit(
                        order_id=order.id,
                        position=order.store_position.copy(),
                        visit_type="store",
                        order=order,
                    )
                )
                route.visits.append(
                    Visit(
                        order_id=order.id,
                        position=order.customer_position.copy(),
                        visit_type="customer",
                        order=order,
                    )
                )

        return route

    def _apply_insertion(
        self,
        candidate: InsertionCandidate,
        order: Order,
        current_time: float,
    ) -> bool:
        """Apply the selected insertion to the vehicle (drone or motorbike)."""
        vehicle = candidate.drone  # Can be either Drone or Motorbike
        route = candidate.route
        
        # Determine vehicle type for logging
        is_motorbike = isinstance(vehicle, Motorbike)
        vehicle_name = "Motorbike" if is_motorbike else "Drone"
        vehicle_emoji = "🏍️" if is_motorbike else "🚀"

        # Calculate exact route (2D for motorbike, 3D for drone)
        exact_route = self.route_evaluator.calculate_exact_route(route)
        if not exact_route or len(exact_route) < 2:
            return False

        # Initialize current_orders if not exists
        if not hasattr(vehicle, "current_orders") or vehicle.current_orders is None:
            vehicle.current_orders = []
        if order not in vehicle.current_orders:
            vehicle.current_orders.append(order)

        vehicle.current_order = order
        order.assigned_drone = vehicle  # Keep field name for compatibility
        order.status = OrderStatus.ASSIGNED

        # Build waypoint order map
        waypoint_order_map = self._build_waypoint_order_map(route, exact_route)

        if vehicle.status == DroneStatus.IDLE:
            vehicle.route_waypoint_order_map = waypoint_order_map
            vehicle._waypoint_index = 0
            vehicle.start_delivery(exact_route)
            if candidate.is_new_drone:
                print(f"{vehicle_emoji} {vehicle_name} {vehicle.id}: Starting delivery with {len(vehicle.current_orders)} order(s)")
            else:
                print(f"{vehicle_emoji} {vehicle_name} {vehicle.id}: Starting multi-delivery with {len(vehicle.current_orders)} orders")
        else:
            vehicle.route_waypoint_order_map = waypoint_order_map
            self._update_drone_route(vehicle, exact_route)
            print(f"📝 {vehicle_name} {vehicle.id}: Updated route, now has {len(vehicle.current_orders)} orders")

        return True

    def _build_waypoint_order_map(self, route: Route, exact_route: List[Position]) -> dict:
        """Build mapping from waypoint index to (Order, visit_type).
        
        Matches waypoints to visit positions using a two-pass approach:
        1. First pass: Find exact or very close matches (within threshold)
        2. Second pass: For unmatched visits, find the closest waypoint
        
        Uses 2D distance (x-z plane) for matching to handle motorbike routes
        where waypoints are at y=0 but store/customer positions have height.
        
        Skips index 0 (start position/depot) to prevent triggering service time
        at the depot when starting a delivery.
        """
        mapping = {}
        
        # Check if this is a motorbike (route waypoints at y=0)
        is_motorbike = isinstance(route.drone, Motorbike)
        
        # Use different thresholds for motorbike (2D matching needs more tolerance)
        # Motorbike detour waypoints can be significantly offset from building centers
        if is_motorbike:
            exact_threshold = 10.0  # Increased from 5.0 for detour waypoints
            fallback_threshold = 50.0  # Increased from 15.0 to handle large detours
        else:
            exact_threshold = 1.0
            fallback_threshold = max(config.NODE_OFFSET, 3.0)
        
        # Get depot position to exclude it from matching
        depot_pos = route.depot_position
        
        # Track which visits have been matched
        matched_visits = set()

        # First pass: Find exact matches for each visit
        for visit_idx, visit in enumerate(route.visits):
            best_idx = -1
            best_dist = float('inf')
            
            for idx, waypoint in enumerate(exact_route):
                # Skip first waypoint (index 0) - it's the start position (depot)
                if idx == 0:
                    continue
                
                # Skip if already mapped
                if idx in mapping:
                    continue
                
                # Skip waypoints that are very close to depot (return path)
                # Use 2D distance for depot check too
                if depot_pos:
                    depot_dist = waypoint.distance_to_2d(depot_pos) if is_motorbike else waypoint.distance_to(depot_pos)
                    if depot_dist < exact_threshold:
                        continue
                
                # Use 2D distance for motorbike, 3D for drone
                if is_motorbike:
                    dist = waypoint.distance_to_2d(visit.position)
                else:
                    dist = waypoint.distance_to(visit.position)
                    
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            # Only map if within exact threshold
            if best_idx != -1 and best_dist <= exact_threshold:
                mapping[best_idx] = (visit.order, visit.visit_type)
                matched_visits.add(visit_idx)

        # Second pass: For unmatched visits, find closest waypoint within fallback threshold
        for visit_idx, visit in enumerate(route.visits):
            if visit_idx in matched_visits:
                continue
            
            best_idx = -1
            best_dist = float('inf')
            
            for idx, waypoint in enumerate(exact_route):
                if idx == 0:
                    continue
                if idx in mapping:
                    continue
                    
                if depot_pos:
                    depot_dist = waypoint.distance_to_2d(depot_pos) if is_motorbike else waypoint.distance_to(depot_pos)
                    if depot_dist < exact_threshold:
                        continue
                
                # Use 2D distance for motorbike, 3D for drone
                if is_motorbike:
                    dist = waypoint.distance_to_2d(visit.position)
                else:
                    dist = waypoint.distance_to(visit.position)
                    
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            if best_idx != -1 and best_dist <= fallback_threshold:
                mapping[best_idx] = (visit.order, visit.visit_type)

        return mapping

    def _update_drone_route(self, drone: Drone, new_route: List[Position]) -> None:
        """Update an in-flight drone's route."""
        current_pos = drone.position
        best_idx = 0
        best_dist = float("inf")

        for idx, waypoint in enumerate(new_route):
            dist = current_pos.distance_to(waypoint)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_dist < config.NODE_OFFSET:
            drone.route = new_route[best_idx + 1:]
            drone._waypoint_index = best_idx + 1
        else:
            drone.route = new_route[best_idx:]
            drone._waypoint_index = best_idx

