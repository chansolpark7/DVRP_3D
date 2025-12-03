"""
Routing algorithms and path optimization interfaces (3D)
"""

import math
import heapq
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set
from shapely.geometry import LineString
from ..models.entities import Position, Order, Drone, Map, Building
import config

# Import floor height constant
FLOOR_HEIGHT = config.FLOOR_HEIGHT
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NODE_OFFSET = getattr(config, 'NODE_OFFSET', 1.0)
class RoutingAlgorithm(ABC):
    
    @abstractmethod
    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        pass
    
    @abstractmethod
    def calculate_distance(self, route: List[Position]) -> float:
        pass


class SimpleRouting(RoutingAlgorithm):
    
    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        route = [start]
        for waypoint in waypoints:
            route.append(waypoint)
        route.append(end)
        return route
    
    def calculate_distance(self, route: List[Position]) -> float:
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        return total_distance


class MultiLevelAStarRouting(RoutingAlgorithm):
    
    def __init__(self, map_obj: Map, k_levels: int = 10):
        self.map = map_obj
        self.k_levels = k_levels
        self.height_levels = self._get_height_levels()
        
        print(f"  MultiLevelAStarRouting initialized with {len(self.height_levels)} height levels")
        print(f"    Levels: {[f'{h:.1f}' for h in self.height_levels[:5]]}{'...' if len(self.height_levels) > 5 else ''}")

    def _get_height_levels(self) -> List[float]:
        """Generate height levels based on FLOOR_HEIGHT to match Store/Customer positions"""
        if not self.map.buildings:
            return [0.0]
        
        # Get maximum building height
        max_height = max(building.height for building in self.map.buildings)
        
        # Generate levels at each floor height (matching Store/Customer positions)
        # floor_y = floor * FLOOR_HEIGHT + FLOOR_HEIGHT / 2
        levels = []
        
        # Add ground level
        levels.append(0.0)
        
        # Add center of each floor level
        floor = 0
        while True:
            floor_center_y = floor * FLOOR_HEIGHT + FLOOR_HEIGHT / 2
            if floor_center_y > max_height:
                break
            levels.append(floor_center_y)
            floor += 1
        
        # Also add the maximum building height for top coverage
        if max_height not in levels:
            levels.append(max_height)
        
        return sorted(set(levels))

    def sample_ring_accumulated(self, coords):
        MAX_EDGE_STEP = 6
        sampled = []
        n = len(coords)

        # 처음 기준점
        prev_x, prev_z = coords[0]
        sampled.append((prev_x, prev_z))

        accumulated = 0.0  # 누적 길이

        for i in range(1, n + 1):
            cur_x, cur_z = coords[i % n]  # 순환
            dx = cur_x - prev_x
            dz = cur_z - prev_z
            seg_len = (dx*dx + dz*dz) ** 0.5

            while accumulated + seg_len >= MAX_EDGE_STEP:
                remain = MAX_EDGE_STEP - accumulated
                t = remain / seg_len
                sx = prev_x + dx * t
                sz = prev_z + dz * t
                sampled.append((sx, sz))
                accumulated = 0.0
                seg_len -= remain
                prev_x, prev_z = sx, sz
                dx = cur_x - prev_x
                dz = cur_z - prev_z

            accumulated += seg_len
            prev_x, prev_z = cur_x, cur_z

        return sampled


    def _get_building_vertices_3d(self, building: Building) -> List[Position]:
        """건물의 꼭짓점에서 약간 바깥쪽으로 오프셋된 노드 위치를 반환합니다.
        
        지면(0)과 건물의 실제 꼭대기 높이에 노드를 생성합니다.
        층별 노드는 _project_vertices_to_levels에서 생성됩니다.
        오프셋은 XZ 평면(수평)으로만 적용됩니다.
        """
        EPSILON = 1e-5
        def ccw(a, b, c):
            return (b[0] - a[0])*(c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

        def convex_hull(point): # ccw
            point = sorted(point)
            a = []
            for p in point:
                while len(a) >= 2 and ccw(a[-2], a[-1], p) <= EPSILON:
                    a.pop()
                a.append(p)
            b = []
            for p in reversed(point):
                while len(b) >= 2 and ccw(b[-2], b[-1], p) <= EPSILON:
                    b.pop()
                b.append(p)
            return a[:-1] + b[:-1]

        outer_points = list(map(tuple, building.outer_poly.exterior.coords[:-1]))
        inner_points = list(map(tuple, building.inner_poly.exterior.coords[:-1]))
        outer_hull = convex_hull(outer_points)
        inner_hull = convex_hull(inner_points)
        n = len(outer_hull)
        m = len(inner_hull)

        start = 0
        for j in range(1, m):
            if ccw(outer_hull[0], inner_hull[start], inner_hull[j]) <= EPSILON:
                start = j

        i = 0
        j = start
        p = outer_hull[i]

        sampled = []
        sampled.append(p)
        while True:
            while ccw(p, inner_hull[j], inner_hull[(j+1) % m]) <= EPSILON and j != (start - 1) % m:
                j = (j + 1) % m

            while outer_hull[i] == p or ccw(p, inner_hull[j], outer_hull[i]) * ccw(p, inner_hull[j], outer_hull[(i+1)%n]) > EPSILON:
                i = (i + 1) % n
            index = outer_points.index(outer_hull[(i+1) % n])
            while ccw(p, inner_hull[j], outer_points[index]) * ccw(p, inner_hull[j], outer_points[(index-1) % len(outer_points)]) > EPSILON:
                index = (index - 1) % len(outer_points)
            
            A = outer_points[(index-1) % len(outer_points)]
            B = outer_points[index]
            I = inner_hull[j]
            a = A[0] - p[0]
            b = A[1] - p[1]
            c = I[0] - p[0]
            d = I[1] - p[1]
            e = B[0] - A[0]
            f = B[1] - A[1]
            assert c*f - d*e != 0

            alpha = (a*f - b*e) / (c*f - d*e)
            new_p = (p[0] + alpha * c, p[1] + alpha * d)
            if p != sampled[0] and ccw(p, new_p, sampled[0]) <= EPSILON:
                break
            p = new_p
            sampled.append(p)
            i = (i+1) % n
            if j == (start-1) % m: break

        vertices = []
        for level in self.height_levels:
            if level > building.height: break
            for x, y in sampled:
                vertices.append(Position(x, level, y, building.id))

        return vertices

    def _filter_relevant_buildings(self, p1: Position, p2: Position,
                                   start_building_id: Optional[int] = None,
                                   end_building_id: Optional[int] = None) -> List[Building]:
        """주어진 직선 경로(p1-p2)와 교차하는 건물 중 시작/종료 건물을 제외하고 필터링합니다."""
        # (이전 답변의 코드와 동일)
        relevant_buildings = []
        baseline_2d_min_x = min(p1.x, p2.x)
        baseline_2d_max_x = max(p1.x, p2.x)
        baseline_2d_min_z = min(p1.z, p2.z)
        baseline_2d_max_z = max(p1.z, p2.z)

        # 기준선이 건물을 통과하는지 확인
        # destination_building_id는 관련 없으므로 None 전달
        for building in self.map.buildings:
            if building.id == start_building_id or building.id == end_building_id:
                continue

            half_w = building.width / 2
            half_d = building.depth / 2
            b_min_x = building.position.x - half_w
            b_max_x = building.position.x + half_w
            b_min_z = building.position.z - half_d
            b_max_z = building.position.z + half_d

            if b_max_x < baseline_2d_min_x or b_min_x > baseline_2d_max_x or \
               b_max_z < baseline_2d_min_z or b_min_z > baseline_2d_max_z:
                continue

            if self._segment_collides_3d(p1, p2, excluded_building_ids={p1.building_id, p2.building_id}, building_ids_to_check={building.id}):
                 relevant_buildings.append(building)
        return relevant_buildings
            
    def _segment_collides_3d(self, p1: Position, p2: Position,
                               excluded_building_ids: Optional[Set[Optional[int]]] = None,
                               building_ids_to_check: Optional[Set[int]] = None) -> bool:
        """Check if 3D segment intersects any building footprint (polygon) within overlapping height."""

        line_2d = LineString([(p1.x, p1.z), (p2.x, p2.z)])
        seg_y_min, seg_y_max = min(p1.y, p2.y), max(p1.y, p2.y)
        
        building_ids: set = set(self.map.tree.query(line_2d))
        if building_ids_to_check is not None:
            building_ids = building_ids & building_ids_to_check
        if excluded_building_ids is not None:
            building_ids -= excluded_building_ids

        for building_id in building_ids:
            building: Building = self.map.buildings[building_id]
            b_y_min, b_y_max = building._vertical_bounds()
            if seg_y_max < b_y_min or seg_y_min > b_y_max:
                continue

            poly = building.inner_poly
            if poly is not None and line_2d.intersects(poly):
                return True

        return False

    def _find_path_core(self, start: Position, end: Position) -> List[Position]:
        """기준선 기반 필터링된 그래프에서 A* 경로를 찾아 노드 리스트(튜플)를 반환합니다."""
        if start == end:
            return [(start.x, start.y, start.z)]

        nodes: List[Position] = [start, end]

        # 도착점이 속한 건물 ID 찾기 (충돌 예외 처리용)

        # 관련 건물 필터링 (시작 건물은 여기서 필터링 안 함, 어차피 충돌 무시됨)
        relevant_buildings = self._filter_relevant_buildings(start, end, end_building_id=end.building_id)
        relevant_building_ids = set()
        # print(f"       CORE: Found {len(relevant_buildings)} relevant buildings.")

        # 관련 건물의 노드 추가 (오프셋 적용됨)
        for building in relevant_buildings:
            relevant_building_ids.add(building.id)
            # 꼭짓점 노드 추가
            vertices = self._get_building_vertices_3d(building) # 오프셋 적용된 노드
            for vertex in vertices:
                if vertex in nodes: raise
                nodes.append(vertex)

        n = len(nodes)
        dist = [-1] * n
        connection = [None] * n
        dist[0] = 0
        queue = [(0, 0, 0)]

        visited = [False] * n
        while queue:
            _, d, x = heapq.heappop(queue)
            if visited[x] and d < dist[x]: raise
            visited[x] = True

            if x == end: break
            if dist[x] != d: continue
            for y in range(n):
                if x == y: continue
                p1 = nodes[x]
                p2 = nodes[y]
                d_ = self._euclidean_distance_3d(nodes[x], nodes[y]) + d
                if dist[y] != -1 and d_ >= dist[y]: continue
                if self._segment_collides_3d(
                    p1,
                    p2,
                    {p1.building_id, p2.building_id},
                    building_ids_to_check=relevant_building_ids):
                    continue

                dist[y] = d_
                connection[y] = x
                heapq.heappush(queue, (d_ + self._euclidean_distance_3d((p2.x, p2.y, p2.z), (end.x, end.y, end.z)), d_, y))

        x = 1
        if connection[x] == None:
            print(f"⚠️  Routing: No path found")
            return []

        route = [nodes[x]]
        while x != 0:
            x = connection[x]
            route.append(nodes[x])
        route.reverse()
        return route

    def _segment_intersects_rect_2d(self, x1: float, z1: float, x2: float, z2: float,
                                      rect_x_min: float, rect_z_min: float,
                                      rect_x_max: float, rect_z_max: float) -> bool:
        
        # 선분의 AABB가 사각형과 겹치는지 빠른 검사
        if max(x1, x2) < rect_x_min or min(x1, x2) > rect_x_max or \
           max(z1, z2) < rect_z_min or min(z1, z2) > rect_z_max:
            return False

        # Liang-Barsky 알고리즘 (또는 Cohen-Sutherland)
        dx = x2 - x1
        dz = z2 - z1
        
        t_min = 0.0
        t_max = 1.0
        
        edges = [
            (-dx, x1 - rect_x_min),  # Left
            ( dx, rect_x_max - x1),  # Right
            (-dz, z1 - rect_z_min),  # Bottom
            ( dz, rect_z_max - z1)   # Top
        ]
        
        for p, q in edges:
            if p == 0:
                if q < 0:
                    return False
            else:
                t = q / p
                if p < 0:
                    if t > t_max: return False
                    t_min = max(t_min, t)
                else:
                    if t < t_min: return False
                    t_max = min(t_max, t)

        return t_min <= t_max

    
    def _euclidean_distance_3d(self, pos1: Tuple[float, float, float], 
                               pos2: Tuple[float, float, float]) -> float:
        return math.sqrt(
            (pos1[0] - pos2[0])**2 + 
            (pos1[1] - pos2[1])**2 + 
            (pos1[2] - pos2[2])**2
        )

    def _visualize_full_route(self, route: List[Position], title: str = "Full Delivery Route", 
                              store_pos: Position = None, customer_pos: Position = None):
        """전체 배달 경로를 3D로 시각화합니다.
        
        Args:
            route: 전체 경로 리스트
            title: 그래프 제목
            store_pos: Store 위치 (정확한 표시를 위해)
            customer_pos: Customer 위치 (정확한 표시를 위해)
        """
        if not route or len(route) < 2:
            print("No route to visualize")
            return
        
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # --- Draw Building Outlines ---
        for building in self.map.buildings:
            half_w = building.width / 2
            half_d = building.depth / 2
            cx = building.position.x
            cz = building.position.z
            h = building.height
            x_coords = [cx - half_w, cx + half_w, cx + half_w, cx - half_w, cx - half_w]
            z_coords = [cz - half_d, cz - half_d, cz + half_d, cz + half_d, cz - half_d]
            ax.plot(x_coords, z_coords, zs=0, color='black', alpha=0.3, linewidth=1)
            ax.plot(x_coords, z_coords, zs=h, color='black', alpha=0.3, linewidth=1)
            for i in range(4):
                ax.plot([x_coords[i], x_coords[i]], [z_coords[i], z_coords[i]], [0, h], color='black', alpha=0.3, linewidth=1)
        
        # --- Draw Route Path ---
        route_xs = [pos.x for pos in route]
        route_ys = [pos.z for pos in route]  # Z -> Y in matplotlib
        route_zs = [pos.y for pos in route]  # Y (height) -> Z in matplotlib
        
        ax.plot(route_xs, route_ys, route_zs, color='red', linewidth=3, marker='o', markersize=6, label='Delivery Route', alpha=0.9)
        
        # --- Highlight Key Points ---
        # Start (Depot)
        ax.scatter(route[0].x, route[0].z, route[0].y, s=200, c='green', marker='o', label='Start (Depot)', depthshade=True, edgecolors='black', linewidths=2)
        # End (back to Depot)
        ax.scatter(route[-1].x, route[-1].z, route[-1].y, s=200, c='blue', marker='s', label='End (Depot)', depthshade=True, edgecolors='black', linewidths=2)
        
        # Key Waypoints and Intermediate Points
        if len(route) > 2:
            store_drawn = False
            customer_drawn = False
            
            for pos in route[1:-1]:  # Start와 End 제외
                # Store 위치 확인 (정확한 위치가 주어진 경우)
                is_store = False
                is_customer = False
                
                if store_pos and abs(pos.x - store_pos.x) < 0.1 and abs(pos.y - store_pos.y) < 0.1 and abs(pos.z - store_pos.z) < 0.1:
                    is_store = True
                elif customer_pos and abs(pos.x - customer_pos.x) < 0.1 and abs(pos.y - customer_pos.y) < 0.1 and abs(pos.z - customer_pos.z) < 0.1:
                    is_customer = True
                
                if is_store and not store_drawn:
                    # Store (주황색 별)
                    ax.scatter(pos.x, pos.z, pos.y, s=300, c='orange', marker='*', 
                              label='Store (Pickup)', depthshade=True, edgecolors='darkred', linewidths=2.5)
                    store_drawn = True
                elif is_customer and not customer_drawn:
                    # Customer (보라색 다이아몬드)
                    ax.scatter(pos.x, pos.z, pos.y, s=250, c='purple', marker='D', 
                              label='Customer (Delivery)', depthshade=True, edgecolors='darkviolet', linewidths=2)
                    customer_drawn = True
                else:
                    # 일반 경유지 (작은 회색 다이아몬드)
                    ax.scatter(pos.x, pos.z, pos.y, s=50, c='lightgray', marker='d', 
                              alpha=0.5, depthshade=True, edgecolors='gray', linewidths=0.5)
        
        # --- Setup Plot ---
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Z (Depth)', fontsize=12)
        ax.set_zlabel('Y (Height)', fontsize=12)
        ax.set_xlim(0, self.map.width)
        ax.set_ylim(0, self.map.depth)
        ax.set_zlim(0, self.map.max_height * 1.1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        plt.show()

    def calculate_route_rec(self, start: Position, end: Position, depth=0) -> List[Position]:
        if depth > 100: return None

        is_direct_path_safe = not self._segment_collides_3d(start, end, excluded_building_ids={start.building_id, end.building_id})
        if is_direct_path_safe:
            return [start, end]

        route = self._find_path_core(start, end)
        if not route or len(route) < 2:
            return None

        #print(route)
        full_route = [route[0]]
        for i in range(len(route) - 1):
            a = route[i]
            b = route[i+1]
            is_step_safe = not self._segment_collides_3d(a, b, excluded_building_ids={a.building_id, b.building_id, end.building_id})
            if is_step_safe:
                full_route.append(b)
            else:
                #print(a, b)
                extended_route = self.calculate_route_rec(a, b, depth + 1)
                if not extended_route: return None

                full_route.extend(extended_route[1:])

        return full_route


    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        """점진적 경로 탐색(Incremental Pathfinding)을 사용하여 전체 경로를 계산합니다."""
        full_route = [start]
        segment_targets = waypoints + [end] # 거쳐갈 목표 지점들

        for p in full_route + segment_targets:
            building = self.map.get_building_containing_point(start)
            p.building_id = building.id if building else None

        current_segment_start = start

        for segment_end in segment_targets:
            working_path_segment = self.calculate_route_rec(current_segment_start, segment_end)
            if not working_path_segment:
                print(f"❌ Routing Error: Path calculation failed")
                return []
            else:
                full_route.extend(working_path_segment[1:])

            current_segment_start = segment_end

        return full_route
    
    def calculate_distance(self, route: List[Position]) -> float:
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        return total_distance


class DroneRouteOptimizer:
    
    def __init__(self, routing_algorithm: RoutingAlgorithm = None):
        self.routing_algorithm = routing_algorithm or SimpleRouting()
    
    def optimize_delivery_route(self, drone: Drone, order: Order, visualize: bool = False) -> List[Position]:
        """드론 배달 경로를 최적화합니다.
        
        Args:
            drone: 배달을 수행할 드론
            order: 배달할 주문
            visualize: True일 경우 전체 경로를 3D로 시각화 (기본값: False)
        
        Returns:
            전체 배달 경로 (depot → store → customer → depot)
        """
        if not drone.current_order or drone.current_order.id != order.id:
            raise ValueError("Drone is not assigned to this order")
        
        start = drone.position.copy()
        waypoints = [order.store_position.copy()]
        end = order.customer_position.copy()
        
        route = self.routing_algorithm.calculate_route(start, waypoints, end)
        
        # (수정) 경로 탐색 실패 시 빈 리스트 반환
        if not route:
            return []

        depot_pos = drone.depot.get_center().copy()
        return_route = self.routing_algorithm.calculate_route(
            order.customer_position, [], depot_pos
        )
        
        # (수정) 복귀 경로 탐색 실패 시 빈 리스트 반환
        if not return_route:
            return []
        
        full_route = route + return_route[1:]
        
        # 전체 경로 시각화 (옵션)
        if visualize and isinstance(self.routing_algorithm, MultiLevelAStarRouting):
            self.routing_algorithm._visualize_full_route(
                full_route, 
                title=f"Full Delivery Route - Order {order.id} (Drone {drone.id})",
                store_pos=order.store_position,
                customer_pos=order.customer_position
            )
        
        return [position.copy() for position in full_route]

    def calculate_delivery_time(self, route: List[Position], drone_speed: float = None) -> float:
        if drone_speed is None:
            drone_speed = config.DRONE_SPEED
        
        total_distance = self.routing_algorithm.calculate_distance(route)
        delivery_time = total_distance / drone_speed
        
        return delivery_time
    
    def optimize_multiple_deliveries(self, drone: Drone, orders: List[Order]) -> List[Position]:
        if not orders:
            return []
        
        if len(orders) == 1:
            return self.optimize_delivery_route(drone, orders[0])
        
        start = drone.position
        waypoints = []
        
        for order in orders:
            waypoints.append(order.store_position)
        
        for order in orders:
            waypoints.append(order.customer_position)
        
        end = drone.depot.get_center()
        
        route = self.routing_algorithm.calculate_route(start, waypoints, end)
        return route


class RouteValidator:
    
    @staticmethod
    def validate_route_feasibility(route: List[Position], drone: Drone) -> Tuple[bool, Optional[str], str]:
        if not route:
            return False, "empty_route", "Empty route"
        
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        max_distance = drone.battery_level * config.DRONE_BATTERY_LIFE
        
        if total_distance > max_distance:
            return False, "battery", f"Route distance {total_distance:.2f} exceeds battery range {max_distance:.2f}"
        
        estimated_time = total_distance / drone.speed
        max_delivery_time = config.MAX_ORDER_DELAY
        
        if estimated_time > max_delivery_time:
            return False, "time_limit", f"Estimated delivery time {estimated_time:.2f}s exceeds maximum {max_delivery_time}s"
        
        return True, None, "Route is feasible"
    
    @staticmethod
    def validate_route_safety(route: List[Position], map_obj) -> Tuple[bool, str]:
        for position in route:
            if (position.x < 0 or position.x > map_obj.width or
                position.z < 0 or position.z > map_obj.depth or
                position.y < 0 or position.y > map_obj.max_height):
                return False, f"Position ({position.x:.1f}, {position.y:.1f}, {position.z:.1f}) is outside map bounds"
            
            building_at_pos = map_obj.get_building_at_position(position)
            if building_at_pos:
                return False, f"Position ({position.x:.1f}, {position.y:.1f}, {position.z:.1f}) collides with building {building_at_pos.id}"
        
        return True, "Route is safe"


class RouteAnalyzer:
    
    @staticmethod
    def analyze_route_efficiency(route: List[Position]) -> dict:
        if len(route) < 2:
            return {
                'total_distance': 0,
                'straight_line_distance': 0,
                'efficiency_ratio': 1.0,
                'number_of_segments': 0
            }
        
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += route[i].distance_to(route[i + 1])
        
        straight_line_distance = route[0].distance_to(route[-1])
        
        efficiency_ratio = straight_line_distance / max(total_distance, 0.001)
        
        return {
            'total_distance': total_distance,
            'straight_line_distance': straight_line_distance,
            'efficiency_ratio': efficiency_ratio,
            'number_of_segments': len(route) - 1
        }
    
    @staticmethod
    def compare_routes(routes: List[List[Position]]) -> dict:
        if not routes:
            return {'best_route_index': -1, 'comparison': {}}
        
        route_analyses = []
        for i, route in enumerate(routes):
            analysis = RouteAnalyzer.analyze_route_efficiency(route)
            analysis['route_index'] = i
            route_analyses.append(analysis)
        
        best_route = min(route_analyses, key=lambda x: x['total_distance'])
        
        return {
            'best_route_index': best_route['route_index'],
            'comparison': route_analyses,
            'best_route': best_route
        }


class DynamicRouteUpdater:
    
    def __init__(self, routing_algorithm: RoutingAlgorithm):
        self.routing_algorithm = routing_algorithm
    
    def update_route_for_traffic(self, original_route: List[Position], 
                               traffic_conditions: dict) -> List[Position]:
        return original_route
    
    def update_route_for_weather(self, original_route: List[Position], 
                               weather_conditions: dict) -> List[Position]:
        return original_route
    
    def reroute_for_emergency(self, current_position: Position, 
                            emergency_location: Position,
                            original_destination: Position) -> List[Position]:
        waypoints = []
        return self.routing_algorithm.calculate_route(current_position, waypoints, original_destination)
