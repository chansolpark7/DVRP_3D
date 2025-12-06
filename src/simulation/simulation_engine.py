"""
Real-time simulation engine for drone delivery system (3D)
Designed to be integrated with Panda3D for visualization
"""

import time
from typing import List, Dict, Optional, Callable
from ..models.entities import Drone, Order, OrderStatus, DroneStatus
from ..algorithms.order_manager import OrderManager
import config


class SimulationEngine:
    """Main simulation engine for the drone delivery system (3D)
    
    Designed to work with Panda3D visualization.
    Call update_step() each frame to advance the simulation.
    """
    
    def __init__(self, map_obj, order_manager: OrderManager = None):
        self.map = map_obj
        self.order_manager = order_manager or OrderManager(map_obj)
        
        # Simulation state
        self.is_running = False
        self.paused = False
        self.simulation_time = 0.0
        self.speed_multiplier = config.SIMULATION_SPEED
        
        # Event system (simple callback-based)
        self.event_handlers = {}

        # Statistics
        self.stats = {
            'total_orders_processed': 0,
            'total_deliveries_completed': 0,
            'average_delivery_time': 0.0,
            'total_drone_distance': 0.0,
            'simulation_duration': 0.0,
            'depot_cost': config.TOTAL_DEPOTS * config.DEPOT_COST,
            'drone_cost': config.TOTAL_DEPOTS * config.DRONES_PER_DEPOT * config.DRONE_COST,
            'charging_cost': 0,
            'penalty_cost': 0,
            'failed_orders': 0
        }
        self.failed_orders: Dict[int, Dict] = {}
        self.retry_interval = getattr(config, "ROUTE_RETRY_INTERVAL", 60.0)
        self.retry_max_attempts = getattr(config, "ROUTE_RETRY_MAX_ATTEMPTS", 3)
        self.failure_events: List[Dict] = []

        if hasattr(self.order_manager, "register_route_failure_handler"):
            self.order_manager.register_route_failure_handler(self._handle_route_failure)
    
    def start_simulation(self):
        """Start the simulation"""
        if self.is_running:
            print("Simulation is already running")
            return
        
        print("Starting simulation...")
        self.is_running = True
        self.paused = False
        self.simulation_time = 0.0
        
        self._emit_event('simulation_started', {'time': self.simulation_time})
    
    def stop_simulation(self):
        """Stop the simulation"""
        if not self.is_running:
            print("Simulation is not running")
            return
        
        print("Stopping simulation...")
        self.is_running = False
        
        self._emit_event('simulation_stopped', {'time': self.simulation_time})
    
    def pause_simulation(self):
        """Pause the simulation"""
        if not self.paused:
            self.paused = True
            self._emit_event('simulation_paused', {'time': self.simulation_time})
    
    def resume_simulation(self):
        """Resume the simulation"""
        if self.paused:
            self.paused = False
            self._emit_event('simulation_resumed', {'time': self.simulation_time})
    
    def set_speed(self, speed_multiplier: float):
        """Set simulation speed multiplier"""
        self.speed_multiplier = max(0.1, min(10.0, speed_multiplier))
        print(f"Simulation speed set to {self.speed_multiplier}x")
    
    def update_step(self, delta_time: float):
        """Update simulation for one frame
        
        This method should be called each frame by the external loop (e.g., Panda3D).
        
        Args:
            delta_time: Time elapsed since last frame (in seconds)
        """
        # Skip update if not running or paused
        if not self.is_running or self.paused:
            return
        
        try:
            # Apply speed multiplier to delta time
            simulation_delta_time = delta_time * self.speed_multiplier
            self.simulation_time += simulation_delta_time
            
            # Update order manager (process orders, assign to drones)
            completed_orders = self.order_manager.process_orders(self.simulation_time)
            self._retry_failed_orders()
            
            # Update all drones (3D movement)
            self._update_drones(simulation_delta_time)
            
            # Update statistics
            self._update_statistics(completed_orders)
            
            # Emit update event
            self._emit_event('simulation_update', {
                'time': self.simulation_time,
                'delta_time': simulation_delta_time,
                'completed_orders': len(completed_orders)
            })
            
        except Exception as e:
            print(f"Error in simulation update: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_drones(self, delta_time: float):
        """Update all drones in the simulation (3D movement)"""
        for depot in self.map.depots:
            for drone in depot.drones:
                if drone.status != DroneStatus.IDLE:
                    # Calculate distance moved in 3D space
                    # The actual movement is handled in drone.update_position()
                    # which uses 3D vectors and considers vertical_speed
                    distance_moved = drone.speed * delta_time
                    self.stats['total_drone_distance'] += distance_moved
                
                self._update_drone(drone, delta_time)
                
                # Check for building collisions and update collision status
                self._check_drone_collision(drone)
    
    def _update_drone(self, drone: Drone, delta_time: float):
        """Update individual drone position and state
        
        The drone's update_position() method handles:
        - 3D movement along route waypoints
        - Automatic state transitions (FLYING -> DELIVERING -> RETURNING -> IDLE)
        - Horizontal and vertical speed considerations
        
        Args:
            drone: Drone to update
            delta_time: Time elapsed since last update
        """
        if drone.status == DroneStatus.IDLE:
            # Reset collision status when idle
            drone.collision_status = 'none'
            # if drone.battery_level == 1: return
            battery_level = min(1, drone.battery_level + delta_time * config.DRONE_CHARGING_SPEED)
            self.stats['charging_cost'] += (battery_level - drone.battery_level) * config.DRONE_BATTERY_CAPACITY * config.CHARGING_COST
            drone.battery_level = battery_level

            return
        else:
            # Update drone's 3D position and state
            # The Drone.update_position() method in entities.py handles all 3D logic
            drone.update_position(delta_time)
    
    def _check_drone_collision(self, drone: Drone):
        """Check if drone is colliding with a building and update collision status
        
        Args:
            drone: Drone to check for collision
        """
        # Only check collision for active drones
        if drone.status == DroneStatus.IDLE:
            drone.collision_status = 'none'
            return
        
        # Check if drone is inside any building
        collided_building = self.map.get_building_containing_point(drone.position)
        
        if collided_building is not None:
            # Drone is inside a building - determine if it's the destination or accidental
            is_destination = False
            
            if drone.current_order is not None:
                # Check if this building is the destination (store or customer)
                if drone.current_order.store_building_id is not None and collided_building.id == drone.current_order.store_building_id:
                    is_destination = True
                elif drone.current_order.customer_building_id is not None and collided_building.id == drone.current_order.customer_building_id:
                    is_destination = True
                
                # Also check if the target position is in this building (fallback method)
                if not is_destination and drone.route and len(drone.route) > 0:
                    target_pos = drone.route[0]
                    distance_to_target = drone.position.distance_to(target_pos)
                    
                    # If very close to target and inside building containing target
                    if distance_to_target < config.NODE_OFFSET * 2:
                        target_building = self.map.get_building_containing_point(target_pos)
                        if target_building is not None and target_building.id == collided_building.id:
                            is_destination = True
            
            # Set collision status based on whether it's a destination
            if is_destination:
                drone.collision_status = 'destination_entry'
            else:
                drone.collision_status = 'accidental'
        else:
            # Drone is not inside any building
            # If previously in destination_entry state, reset to none when exiting building
            drone.collision_status = 'none'
    
    def _handle_route_failure(self, order: Order, reason: str):
        """Queue failed orders for retry attempts."""
        if reason == "time_limit":
            order.status = OrderStatus.CANCELLED
            self.failed_orders.pop(order.id, None)
            self._record_failed_order(order, reason)
            print(f"❌ Order {order.id}: cannot meet time limit, marking as failed.")
            return

        record = self.failed_orders.get(order.id, {'attempts': 0})
        attempts = record['attempts'] + 1
        if attempts >= self.retry_max_attempts:
            order.status = OrderStatus.CANCELLED
            self.failed_orders.pop(order.id, None)
            self._record_failed_order(order, reason)
            print(f"❌ Order {order.id}: routing failed repeatedly; cancelling (reason={reason}).")
            return
        
        next_retry = self.simulation_time + self.retry_interval
        self.failed_orders[order.id] = {
            'order': order,
            'attempts': attempts,
            'next_retry': next_retry,
            'reason': reason
        }
        print(f"↻ Order {order.id}: routing failed (attempt {attempts}); retry scheduled at t={next_retry:.1f}s.")
    
    def _retry_failed_orders(self):
        if not self.failed_orders:
            return
        
        ready = [
            order_id for order_id, data in self.failed_orders.items()
            if data['next_retry'] <= self.simulation_time
        ]
        
        for order_id in ready:
            data = self.failed_orders.pop(order_id, None)
            if not data:
                continue
            order = data['order']
            if order.status in (OrderStatus.CANCELLED, OrderStatus.COMPLETED):
                continue
            print(f"🔁 Retrying order {order.id} (attempt {data['attempts'] + 1}).")
            self.order_manager.retry_order_assignment(order)
    
    def _update_statistics(self, completed_orders: List[Order]):
        """Update simulation statistics"""
        self.stats['total_orders_processed'] += len(completed_orders)
        
        if completed_orders:
            # Calculate average delivery time
            total_delivery_time = 0
            valid_orders = 0
            for order in completed_orders:
                if hasattr(order, 'created_time') and order.created_time is not None:
                    delivery_time = self.simulation_time - order.created_time
                    if delivery_time >= 0:  # Ensure non-negative delivery time
                        total_delivery_time += delivery_time
                        valid_orders += 1
            
            if valid_orders > 0:
                avg_delivery_time = total_delivery_time / valid_orders
                
                # Update running average
                if self.stats['total_deliveries_completed'] > 0:
                    total_processed = self.stats['total_deliveries_completed'] + valid_orders
                    if total_processed > 0:
                        self.stats['average_delivery_time'] = (
                            (self.stats['average_delivery_time'] * self.stats['total_deliveries_completed'] + 
                             avg_delivery_time * valid_orders) / total_processed
                        )
                else:
                    self.stats['average_delivery_time'] = avg_delivery_time
                
                self.stats['total_deliveries_completed'] += valid_orders
        
        self.stats['simulation_duration'] = self.simulation_time
    
    def assign_order_to_drone(self, order: Order) -> bool:
        """Manually assign an order to a drone using OrderManager"""
        if order.status != OrderStatus.PENDING:
            return False
        
        # Use OrderManager to assign the order (which will handle route setting)
        self.order_manager._assign_order_to_depot(order)
        
        if order.status == OrderStatus.ASSIGNED and order.assigned_drone:
            self._emit_event('order_assigned', {
                'order_id': order.id,
                'drone_id': order.assigned_drone.id,
                'depot_id': order.assigned_drone.depot.id if order.assigned_drone.depot else None
            })
            return True
        
        return False
    
    def get_simulation_state(self) -> Dict:
        """Get current simulation state"""
        return {
            'is_running': self.is_running and not self.paused,
            'simulation_time': self.simulation_time,
            'speed_multiplier': self.speed_multiplier,
            'stats': self.stats.copy(),
            'active_orders': len(self.order_manager.orders),
            'completed_orders': len(self.order_manager.completed_orders),
            'depot_info': self.order_manager.get_depot_load_balancing_info()
        }
    
    def get_drone_positions(self) -> Dict:
        """Get current 3D positions of all drones"""
        drone_positions = {}
        
        for depot in self.map.depots:
            for drone in depot.drones:
                drone_positions[drone.id] = {
                    'position': drone.position,  # 3D Position (x, y, z)
                    'status': drone.status.value,
                    'depot_id': depot.id,
                    'has_route': drone.route is not None and len(drone.route) > 0,
                    'route_length': len(drone.route) if drone.route else 0
                }
        
        return drone_positions
    
    def get_active_drones(self) -> List[Drone]:
        """Get list of all active (non-idle) drones
        
        Returns:
            List of drones that are currently flying/delivering
        """
        active_drones = []
        for depot in self.map.depots:
            for drone in depot.drones:
                if drone.status != DroneStatus.IDLE:
                    active_drones.append(drone)
        
        return active_drones

    def _record_failed_order(self, order: Order, reason: str):
        penalty_per_fail = getattr(
            config,
            "FAILED_ORDER_PENALTY",
            config.TIME_PENALTY * config.TIME_PENALTY_CRITERIA
        )
        self.stats['failed_orders'] += 1
        self.stats['penalty_cost'] += penalty_per_fail
        event = {
            'order_id': order.id,
            'reason': reason,
            'customer_position': order.customer_position.copy() if order.customer_position else None,
            'store_position': order.store_position.copy() if order.store_position else None,
            'timestamp': self.simulation_time,
        }
        self.failure_events.append(event)
    
    def get_failure_events(self) -> List[Dict]:
        """Return recorded failure events (used for visualization)."""
        return list(self.failure_events)
    
    # Event system
    def _emit_event(self, event_type: str, data: Dict):
        """Emit an event to registered handlers
        
        Args:
            event_type: Type of event
            data: Event data dictionary
        """
        event = {
            'type': event_type,
            'data': data,
            'timestamp': self.simulation_time
        }
        
        # Call registered handlers immediately
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Error in event handler for '{event_type}': {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def unregister_event_handler(self, event_type: str, handler: Callable):
        """Unregister an event handler"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.stop_simulation()
        
        # Reset state
        self.simulation_time = 0.0
        self.stats = {
            'total_orders_processed': 0,
            'total_deliveries_completed': 0,
            'average_delivery_time': 0.0,
            'total_drone_distance': 0.0,
            'simulation_duration': 0.0,
            'depot_cost': config.TOTAL_DEPOTS * config.DEPOT_COST,
            'drone_cost': config.TOTAL_DEPOTS * config.DRONES_PER_DEPOT * config.DRONE_COST,
            'charging_cost': 0,
            'penalty_cost': 0,
            'failed_orders': 0
        }
        self.failed_orders.clear()
        self.failure_events.clear()
        
        # Reset order manager
        self.order_manager.orders = []
        self.order_manager.completed_orders = []
        
        # Reset all drones to idle state at depot position (3D)
        for depot in self.map.depots:
            for drone in depot.drones:
                drone.status = DroneStatus.IDLE
                drone.current_order = None
                drone.route = None
                drone.position = depot.get_center().copy()  # Use 3D depot center position
        
        print("Simulation reset to initial state")
