"""
Route evaluation helpers shared by multi-delivery algorithms.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import config
from ...models.entities import Position
from .solution_representation import Route

if TYPE_CHECKING:
    from ..routing import RoutingAlgorithm


class RouteEvaluator:
    """Provides fast estimates vs. exact route generation."""

    def __init__(self, routing_algorithm: 'RoutingAlgorithm'):
        self.routing_algorithm = routing_algorithm

    def estimate_route_cost(self, route: Route) -> float:
        """Return straight-line distance as a fast proxy cost."""
        return route.get_total_distance_estimate()

    def estimate_route_costs(self, routes: List[Route]) -> float:
        """Return aggregated cost for a set of routes."""
        return sum(self.estimate_route_cost(route) for route in routes)

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
        print(f"\n🔍 [RouteEvaluator] calculate_exact_route called for Drone {route.drone.id}")
        print(f"   Route start_position: {route.start_position}")
        print(f"   Route depot_position: {route.depot_position}")
        print(f"   Route visits count: {len(route.visits)}")
        
        if not route.start_position or not route.depot_position:
            print(f"   ❌ Missing start_position or depot_position, returning empty route")
            return []

        waypoints = [visit.position for visit in route.visits]
        print(f"   Waypoints to visit: {len(waypoints)}")
        for i, wp in enumerate(waypoints):
            print(f"      Waypoint {i}: ({wp.x:.1f}, {wp.y:.1f}, {wp.z:.1f}) - {route.visits[i].visit_type} for Order {route.visits[i].order_id}")
        
        exact_route = self.routing_algorithm.calculate_route(
            route.start_position,
            waypoints,
            route.depot_position,
        )
        
        print(f"   ✅ Calculated exact route with {len(exact_route)} waypoints")
        if exact_route:
            print(f"      First waypoint: ({exact_route[0].x:.1f}, {exact_route[0].y:.1f}, {exact_route[0].z:.1f})")
            print(f"      Last waypoint: ({exact_route[-1].x:.1f}, {exact_route[-1].y:.1f}, {exact_route[-1].z:.1f})")
            
            # Check if last waypoint matches depot
            last_waypoint = exact_route[-1]
            distance_to_depot = route.depot_position.distance_to(last_waypoint)
            if distance_to_depot > 0.1:
                print(f"      ⚠️ WARNING: Last waypoint is NOT at depot position (distance={distance_to_depot:.4f}m)")
            else:
                print(f"      ✅ Last waypoint is at depot position (distance={distance_to_depot:.4f}m)")
        else:
            print(f"      ⚠️ Empty route returned!")
        
        return exact_route

