"""
Unified insertion strategy for drone delivery routing.

Handles both new drone assignments and insertions into existing routes,
comparing them fairly using:
- For IDLE drones: absolute distance (new route cost)
- For busy drones: cost delta (additional distance from insertion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import config
from ..models.entities import (
    Drone,
    DroneStatus,
    Order,
    OrderStatus,
    Position,
)
from .delivery_types import Route, Visit

if TYPE_CHECKING:
    from .routing import RoutingAlgorithm


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
        """Return True if the drone has enough battery for the route."""
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

    def _collect_candidate_drones(self, order: Order) -> List[Drone]:
        """Collect all candidate drones (both IDLE and with capacity)."""
        drones: List[Drone] = []
        
        for depot in self.map.depots:
            # Distance filter: skip depots too far from the order
            depot_to_store = depot.get_center().distance_to(order.store_position)
            if depot_to_store > MAX_DEPOT_DISTANCE:
                continue
            
            for drone in depot.drones:
                if drone.status == DroneStatus.IDLE or self._has_capacity(drone):
                    drones.append(drone)
        
        return drones

    def _has_capacity(self, drone: Drone) -> bool:
        """Check if drone has capacity for additional orders."""
        if hasattr(drone, "current_orders") and drone.current_orders:
            return len(drone.current_orders) < config.DRONE_CAPACITY
        return drone.current_order is None
    
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
        """Apply the selected insertion to the drone."""
        drone = candidate.drone
        route = candidate.route

        # Calculate exact 3D route
        exact_route = self.route_evaluator.calculate_exact_route(route)
        if not exact_route or len(exact_route) < 2:
            return False

        # Initialize current_orders if not exists
        if not hasattr(drone, "current_orders") or drone.current_orders is None:
            drone.current_orders = []
        if order not in drone.current_orders:
            drone.current_orders.append(order)

        drone.current_order = order
        order.assigned_drone = drone
        order.status = OrderStatus.ASSIGNED

        # Build waypoint order map
        waypoint_order_map = self._build_waypoint_order_map(route, exact_route)

        if drone.status == DroneStatus.IDLE:
            drone.route_waypoint_order_map = waypoint_order_map
            drone._waypoint_index = 0
            drone.start_delivery(exact_route)
            if candidate.is_new_drone:
                print(f"🚀 Drone {drone.id}: Starting delivery with {len(drone.current_orders)} order(s)")
            else:
                print(f"🚀 Drone {drone.id}: Starting multi-delivery with {len(drone.current_orders)} orders")
        else:
            drone.route_waypoint_order_map = waypoint_order_map
            self._update_drone_route(drone, exact_route)
            print(f"📝 Drone {drone.id}: Updated route, now has {len(drone.current_orders)} orders")

        return True

    def _build_waypoint_order_map(self, route: Route, exact_route: List[Position]) -> dict:
        """Build mapping from waypoint index to (Order, visit_type)."""
        mapping = {}
        threshold = max(config.NODE_OFFSET, 5.0)

        visit_idx = 0
        for idx, waypoint in enumerate(exact_route):
            if visit_idx >= len(route.visits):
                break

            visit = route.visits[visit_idx]
            if waypoint.distance_to(visit.position) <= threshold:
                mapping[idx] = (visit.order, visit.visit_type)
                visit_idx += 1

        # Handle remaining visits if any
        while visit_idx < len(route.visits):
            mapping[len(exact_route) - 1 + visit_idx - len(route.visits) + 1] = (
                route.visits[visit_idx].order,
                route.visits[visit_idx].visit_type
            )
            visit_idx += 1

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

