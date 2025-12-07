"""
Data structures shared by multi-delivery routing strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config
from ...models.entities import Drone, Order, Position


@dataclass
class Visit:
    """Represents a single stop (store pickup or customer drop-off)."""

    order_id: int
    position: Position
    visit_type: str  # "store" or "customer"
    order: Order


@dataclass
class Route:
    """Represents a multi-stop route assigned to a single drone."""

    drone_id: int
    drone: Drone
    visits: List[Visit] = field(default_factory=list)
    start_position: Optional[Position] = None
    depot_position: Optional[Position] = None
    start_time: float = 0.0

    def get_total_distance_estimate(self) -> float:
        """Estimate total travel distance using straight-line segments."""
        if not self.start_position or not self.depot_position:
            return 0.0

        if not self.visits:
            return self.start_position.distance_to(self.depot_position)

        total = self.start_position.distance_to(self.visits[0].position)
        for i in range(len(self.visits) - 1):
            total += self.visits[i].position.distance_to(self.visits[i + 1].position)
        total += self.visits[-1].position.distance_to(self.depot_position)
        return total

    def get_battery_required_estimate(self) -> float:
        """Return the conservative travel distance used for battery checks."""
        return self.get_total_distance_estimate() * config.BATTERY_SAFETY_MARGIN

    def get_estimated_arrival_times(self) -> Dict[int, float]:
        """Return estimated arrival time per order (customer), including service times."""
        arrival_times: Dict[int, float] = {}

        if not self.start_position:
            return arrival_times

        current_time = self.start_time
        current_pos = self.start_position

        for visit in self.visits:
            travel_distance = current_pos.distance_to(visit.position)
            travel_time = travel_distance / config.DRONE_SPEED if config.DRONE_SPEED > 0 else 0.0
            current_time += travel_time
            current_time += config.SERVICE_TIME_PER_STOP

            current_pos = visit.position

            if visit.visit_type == "customer":
                arrival_times[visit.order_id] = current_time

        return arrival_times

    def get_orders(self) -> List[Order]:
        """Return unique orders included in this route."""
        orders: List[Order] = []
        seen: set[int] = set()
        for visit in self.visits:
            if visit.order_id not in seen:
                seen.add(visit.order_id)
                orders.append(visit.order)
        return orders

    def is_valid(self, current_time: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """Validate precedence, battery and time window constraints."""
        if not self._check_precedence_constraint():
            return False, "precedence_violation"

        if not self._check_battery_constraint():
            return False, "battery_insufficient"

        if not self._check_time_window_constraint():
            return False, "time_window_violation"
        
        if not self._check_capacity_constraint():
            return False, "capacity_exceeded"

        return True, None

    def _check_precedence_constraint(self) -> bool:
        """Ensure every order's pickup happens before its drop-off."""
        store_indices: Dict[int, int] = {}
        customer_indices: Dict[int, int] = {}

        for idx, visit in enumerate(self.visits):
            if visit.visit_type == "store":
                store_indices[visit.order_id] = idx
            elif visit.visit_type == "customer":
                customer_indices[visit.order_id] = idx

        for order_id, drop_idx in customer_indices.items():
            pickup_idx = store_indices.get(order_id)
            if pickup_idx is None or pickup_idx >= drop_idx:
                return False

        return True

    def _check_battery_constraint(self) -> bool:
        """Ensure the drone has enough battery for the conservative estimate."""
        if not self.drone:
            return False

        required = self.get_battery_required_estimate()
        available = self.drone.battery_level * config.DRONE_BATTERY_LIFE
        if required > available:
            print(
                f"[Routing] Battery check failed for Drone {self.drone.id}: "
                f"required {required:.1f}m (safety {config.BATTERY_SAFETY_MARGIN:.1f}x) "
                f"> available {available:.1f}m"
            )
            return False
        return True

    def _check_time_window_constraint(self) -> bool:
        """Ensure each order can be delivered within the maximum wait time."""
        arrival_times = self.get_estimated_arrival_times()

        for visit in self.visits:
            if visit.visit_type != "customer":
                continue

            order = visit.order
            arrival_time = arrival_times.get(order.id)
            if arrival_time is None:
                continue

            if order.created_time is None:
                continue

            wait_time = arrival_time - order.created_time
            if wait_time > config.CUSTOMER_MAX_WAIT_TIME:
                return False

        return True
    
    def _check_capacity_constraint(self) -> bool:
        """Ensure the number of orders does not exceed drone capacity."""
        unique_orders = len(set(v.order_id for v in self.visits if v.visit_type == "store"))
        return unique_orders <= config.DRONE_CAPACITY


@dataclass
class Solution:
    """Represents a multi-route solution for a batch of orders."""

    routes: List[Route] = field(default_factory=list)
    unassigned_orders: List[Order] = field(default_factory=list)

    def get_total_distance_estimate(self) -> float:
        return sum(route.get_total_distance_estimate() for route in self.routes)

