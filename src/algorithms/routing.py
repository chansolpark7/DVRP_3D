"""
Routing algorithms and path optimization interfaces (3D)
"""

import math
import heapq
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set
from shapely.geometry import LineString, LineString as ShapelyLineString, Point
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
        """Check if 3D segment actually passes through any building volume.
        
        For each building, finds the portion of the segment that overlaps with the
        building's 2D footprint, then checks if the segment's height at that portion
        is within the building's vertical bounds.
        
        Note: excluded_building_ids and building_ids_to_check use building.id values,
        which are converted to list indices for comparison with tree query results.
        """
        line_2d = LineString([(p1.x, p1.z), (p2.x, p2.z)])
        
        # Calculate segment direction for height interpolation
        dx = p2.x - p1.x
        dz = p2.z - p1.z
        dy = p2.y - p1.y
        seg_len_2d = math.sqrt(dx*dx + dz*dz)
        
        # Tree query returns list indices, not building IDs
        building_indices: set = set(self.map.tree.query(line_2d))
        
        # Convert building IDs to indices for filtering
        id_to_idx = getattr(self.map, 'building_id_to_index', None)
        
        if building_ids_to_check is not None and id_to_idx:
            # Convert IDs to indices
            indices_to_check = {id_to_idx.get(bid) for bid in building_ids_to_check if bid is not None}
            indices_to_check.discard(None)
            building_indices = building_indices & indices_to_check
        
        if excluded_building_ids is not None and id_to_idx:
            # Convert IDs to indices
            excluded_indices = {id_to_idx.get(bid) for bid in excluded_building_ids if bid is not None}
            excluded_indices.discard(None)
            building_indices -= excluded_indices

        for building_idx in building_indices:
            building: Building = self.map.buildings[building_idx]
            b_y_min, b_y_max = building._vertical_bounds()
            
            poly = building.inner_poly
            if poly is None:
                continue
            
            # Get the intersection of the 2D line with the polygon
            intersection = line_2d.intersection(poly)
            if intersection.is_empty:
                continue
            
            # Handle different intersection geometry types
            if intersection.geom_type == 'Point':
                # Single point intersection - get t parameter and check height
                t = self._get_t_for_point(p1, dx, dz, seg_len_2d, intersection.x, intersection.y)
                y_at_t = p1.y + dy * t
                if b_y_min <= y_at_t <= b_y_max:
                    return True
                    
            elif intersection.geom_type == 'LineString':
                # Line segment intersection - check if segment height range overlaps building
                coords = list(intersection.coords)
                if len(coords) >= 2:
                    # Get t values for entry and exit points
                    t_entry = self._get_t_for_point(p1, dx, dz, seg_len_2d, coords[0][0], coords[0][1])
                    t_exit = self._get_t_for_point(p1, dx, dz, seg_len_2d, coords[-1][0], coords[-1][1])
                    
                    # Calculate y values at entry and exit
                    y_entry = p1.y + dy * t_entry
                    y_exit = p1.y + dy * t_exit
                    
                    # Get the height range of the segment within the building footprint
                    seg_y_min = min(y_entry, y_exit)
                    seg_y_max = max(y_entry, y_exit)
                    
                    # Check if height ranges overlap
                    if not (seg_y_max < b_y_min or seg_y_min > b_y_max):
                        return True
                        
            elif intersection.geom_type == 'MultiLineString':
                # Multiple line segments - check each one
                for geom in intersection.geoms:
                    coords = list(geom.coords)
                    if len(coords) >= 2:
                        t_entry = self._get_t_for_point(p1, dx, dz, seg_len_2d, coords[0][0], coords[0][1])
                        t_exit = self._get_t_for_point(p1, dx, dz, seg_len_2d, coords[-1][0], coords[-1][1])
                        y_entry = p1.y + dy * t_entry
                        y_exit = p1.y + dy * t_exit
                        seg_y_min = min(y_entry, y_exit)
                        seg_y_max = max(y_entry, y_exit)
                        if not (seg_y_max < b_y_min or seg_y_min > b_y_max):
                            return True
                            
            elif intersection.geom_type == 'MultiPoint':
                for geom in intersection.geoms:
                    t = self._get_t_for_point(p1, dx, dz, seg_len_2d, geom.x, geom.y)
                    y_at_t = p1.y + dy * t
                    if b_y_min <= y_at_t <= b_y_max:
                        return True
            
            elif intersection.geom_type == 'Polygon':
                # Segment passes through polygon interior - use boundary
                boundary = intersection.exterior
                coords = list(boundary.coords)
                if len(coords) >= 2:
                    # Find min and max t along the segment within the polygon
                    t_values = [self._get_t_for_point(p1, dx, dz, seg_len_2d, c[0], c[1]) for c in coords]
                    t_min, t_max = min(t_values), max(t_values)
                    y_at_min = p1.y + dy * t_min
                    y_at_max = p1.y + dy * t_max
                    seg_y_min = min(y_at_min, y_at_max)
                    seg_y_max = max(y_at_min, y_at_max)
                    if not (seg_y_max < b_y_min or seg_y_min > b_y_max):
                        return True
            
            elif intersection.geom_type == 'GeometryCollection':
                # Handle mixed geometry types
                for geom in intersection.geoms:
                    if geom.geom_type == 'LineString':
                        coords = list(geom.coords)
                        if len(coords) >= 2:
                            t_entry = self._get_t_for_point(p1, dx, dz, seg_len_2d, coords[0][0], coords[0][1])
                            t_exit = self._get_t_for_point(p1, dx, dz, seg_len_2d, coords[-1][0], coords[-1][1])
                            y_entry = p1.y + dy * t_entry
                            y_exit = p1.y + dy * t_exit
                            seg_y_min = min(y_entry, y_exit)
                            seg_y_max = max(y_entry, y_exit)
                            if not (seg_y_max < b_y_min or seg_y_min > b_y_max):
                                return True
                    elif geom.geom_type == 'Point':
                        t = self._get_t_for_point(p1, dx, dz, seg_len_2d, geom.x, geom.y)
                        y_at_t = p1.y + dy * t
                        if b_y_min <= y_at_t <= b_y_max:
                            return True

        return False
    
    def _get_t_for_point(self, p1: Position, dx: float, dz: float, seg_len_2d: float, 
                         px: float, pz: float) -> float:
        """Calculate parameter t (0 to 1) for a point along the 2D projection of segment.
        
        t=0 corresponds to p1, t=1 corresponds to p2.
        """
        if seg_len_2d < 1e-9:
            return 0.0
        
        # Project the point onto the line direction
        vec_x = px - p1.x
        vec_z = pz - p1.z
        
        # t = dot(vec, dir) / len^2 = dot(vec, dir/len) / len
        t = (vec_x * dx + vec_z * dz) / (seg_len_2d * seg_len_2d)
        return max(0.0, min(1.0, t))

    def _find_path_core(self, start: Position, end: Position) -> List[Position]:
        """기준선 기반 필터링된 그래프에서 A* 경로를 찾아 노드 리스트(튜플)를 반환합니다.
        
        시작/끝점이 건물 내부에 있는 경우:
        1. 해당 건물의 외부 꼭짓점도 노드에 포함
        2. 시작점 → 시작 건물 꼭짓점, 끝 건물 꼭짓점 → 끝점 연결만 허용
        3. 다른 모든 경로는 건물 충돌 검사 적용
        """
        if start == end:
            return [(start.x, start.y, start.z)]

        nodes: List[Position] = [start, end]
        
        # 시작/끝 건물 ID
        start_building_id = start.building_id
        end_building_id = end.building_id

        # 관련 건물 필터링 (시작/끝 건물 제외)
        relevant_buildings = self._filter_relevant_buildings(
            start, end, 
            start_building_id=start_building_id,
            end_building_id=end_building_id
        )
        relevant_building_ids = set()

        # 관련 건물의 노드 추가 (오프셋 적용됨)
        for building in relevant_buildings:
            relevant_building_ids.add(building.id)
            vertices = self._get_building_vertices_3d(building)
            for vertex in vertices:
                if vertex in nodes: raise
                nodes.append(vertex)
        
        # 시작/끝 건물의 꼭짓점도 추가 (건물 내부에서 외부로 나가기 위해)
        start_building_vertices_start_idx = len(nodes)
        if start_building_id is not None:
            start_building = self.map.buildings[start_building_id]
            vertices = self._get_building_vertices_3d(start_building)
            for vertex in vertices:
                if vertex not in nodes:
                    nodes.append(vertex)
        start_building_vertices_end_idx = len(nodes)
        
        end_building_vertices_start_idx = len(nodes)
        if end_building_id is not None and end_building_id != start_building_id:
            end_building = self.map.buildings[end_building_id]
            vertices = self._get_building_vertices_3d(end_building)
            for vertex in vertices:
                if vertex not in nodes:
                    nodes.append(vertex)
        end_building_vertices_end_idx = len(nodes)
        
        # 충돌 검사에 시작/끝 건물도 포함
        if start_building_id is not None:
            relevant_building_ids.add(start_building_id)
        if end_building_id is not None:
            relevant_building_ids.add(end_building_id)

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

            if x == 1: break  # 끝점(index 1)에 도달
            if dist[x] != d: continue
            
            for y in range(n):
                if x == y: continue
                p1 = nodes[x]
                p2 = nodes[y]
                d_ = self._euclidean_distance_3d(nodes[x], nodes[y]) + d
                if dist[y] != -1 and d_ >= dist[y]: continue
                
                # 특수 케이스 체크
                is_start_to_start_building = (x == 0 and start_building_vertices_start_idx <= y < start_building_vertices_end_idx)
                is_end_building_to_end = (y == 1 and end_building_vertices_start_idx <= x < end_building_vertices_end_idx)
                is_same_building_direct = (x == 0 and y == 1 and start_building_id == end_building_id and start_building_id is not None)
                is_any_to_end = (y == 1)  # 끝점으로 가는 모든 경로
                is_start_to_any = (x == 0)  # 시작점에서 나가는 모든 경로
                
                if is_start_to_start_building or is_end_building_to_end or is_same_building_direct:
                    # 시작/끝 건물 내 이동은 항상 허용 (충돌 검사 생략)
                    pass
                elif is_any_to_end:
                    # 끝점으로 들어가는 경우: 끝 건물만 제외하고 다른 건물은 충돌 검사
                    excluded = {end_building_id} if end_building_id is not None else set()
                    if self._segment_collides_3d(p1, p2, excluded, building_ids_to_check=relevant_building_ids):
                        continue
                elif is_start_to_any:
                    # 시작점에서 나가는 경우: 시작 건물만 제외하고 다른 건물은 충돌 검사
                    excluded = {start_building_id} if start_building_id is not None else set()
                    if self._segment_collides_3d(p1, p2, excluded, building_ids_to_check=relevant_building_ids):
                        continue
                else:
                    # 일반 충돌 검사
                    excluded = set()
                    
                    # 같은 건물의 꼭짓점 간 이동은 해당 건물 제외
                    if p1.building_id == p2.building_id and p1.building_id is not None:
                        excluded.add(p1.building_id)
                    
                    if self._segment_collides_3d(p1, p2, excluded, building_ids_to_check=relevant_building_ids):
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
        
        # 직접 경로 안전 검사
        # 시작/끝 건물은 제외 (드론이 건물 내부로 들어가거나 나가는 것은 허용)
        if start.building_id is not None and start.building_id == end.building_id:
            # 같은 건물 내 이동은 직접 경로 허용
            is_direct_path_safe = True
        else:
            # 시작/끝 건물을 제외하고 다른 건물만 충돌 검사
            excluded = {start.building_id, end.building_id} - {None}
            is_direct_path_safe = not self._segment_collides_3d(start, end, excluded_building_ids=excluded)
        
        if is_direct_path_safe:
            return [start, end]

        route = self._find_path_core(start, end)
        if not route or len(route) < 2:
            return None

        full_route = [route[0]]
        for i in range(len(route) - 1):
            a = route[i]
            b = route[i+1]
            # 각 단계도 건물 충돌 검사 (해당 노드의 건물만 제외)
            is_step_safe = not self._segment_collides_3d(a, b, excluded_building_ids={a.building_id, b.building_id})
            if is_step_safe:
                full_route.append(b)
            else:
                extended_route = self.calculate_route_rec(a, b, depth + 1)
                if not extended_route: return None

                full_route.extend(extended_route[1:])

        return full_route


    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        """점진적 경로 탐색(Incremental Pathfinding)을 사용하여 전체 경로를 계산합니다."""
        full_route = [start]
        segment_targets = waypoints + [end] # 거쳐갈 목표 지점들

        for p in full_route + segment_targets:
            building = self.map.get_building_containing_point(p)
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


class MotorbikeRouting(RoutingAlgorithm):
    """2D Ground-level routing algorithm for motorbikes.
    
    Uses the same algorithm as MultiLevelAStarRouting for drones,
    but operates on the 2D ground plane (y=0) without height consideration.
    """
    
    def __init__(self, map_obj: Map = None):
        self.map = map_obj
        
        if self.map and self.map.buildings:
            print(f"  MotorbikeRouting initialized for 2D ground-level navigation")
    
    def set_map(self, map_obj: Map):
        self.map = map_obj

    def sample_ring_accumulated(self, coords):
        """Sample points along a polygon ring at regular intervals."""
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

    def _get_building_vertices_2d(self, building: Building) -> List[Position]:
        """건물의 꼭짓점에서 약간 바깥쪽으로 오프셋된 노드 위치를 반환합니다.
        
        지면(y=0)에서만 노드를 생성합니다.
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

        # 2D: 지면 레벨(y=0)에서만 노드 생성
        vertices = []
        for x, z in sampled:
            vertices.append(Position(x, 0, z, building.id))

        return vertices

    def _filter_relevant_buildings(self, p1: Position, p2: Position,
                                   start_building_id: Optional[int] = None,
                                   end_building_id: Optional[int] = None) -> List[Building]:
        """주어진 직선 경로(p1-p2)와 교차하는 건물 중 시작/종료 건물을 제외하고 필터링합니다."""
        relevant_buildings = []
        baseline_2d_min_x = min(p1.x, p2.x)
        baseline_2d_max_x = max(p1.x, p2.x)
        baseline_2d_min_z = min(p1.z, p2.z)
        baseline_2d_max_z = max(p1.z, p2.z)

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

            if self._segment_collides_2d(p1, p2, excluded_building_ids={p1.building_id, p2.building_id}, building_ids_to_check={building.id}):
                 relevant_buildings.append(building)
        return relevant_buildings
            
    def _segment_collides_2d(self, p1: Position, p2: Position,
                               excluded_building_ids: Optional[Set[Optional[int]]] = None,
                               building_ids_to_check: Optional[Set[int]] = None) -> bool:
        """Check if 2D segment (on ground plane) passes through any building footprint.
        
        Unlike 3D version, this only checks XZ plane intersection without height consideration.
        """
        line_2d = LineString([(p1.x, p1.z), (p2.x, p2.z)])
        
        # Tree query returns list indices, not building IDs
        building_indices: set = set(self.map.tree.query(line_2d))
        
        # Convert building IDs to indices for filtering
        id_to_idx = getattr(self.map, 'building_id_to_index', None)
        
        if building_ids_to_check is not None and id_to_idx:
            # Convert IDs to indices
            indices_to_check = {id_to_idx.get(bid) for bid in building_ids_to_check if bid is not None}
            indices_to_check.discard(None)
            building_indices = building_indices & indices_to_check
        
        if excluded_building_ids is not None and id_to_idx:
            # Convert IDs to indices
            excluded_indices = {id_to_idx.get(bid) for bid in excluded_building_ids if bid is not None}
            excluded_indices.discard(None)
            building_indices -= excluded_indices

        for building_idx in building_indices:
            building: Building = self.map.buildings[building_idx]
            
            poly = building.inner_poly
            if poly is None:
                continue
            
            # 2D: 단순히 XZ 평면에서의 교차만 확인 (높이 무시)
            intersection = line_2d.intersection(poly)
            if not intersection.is_empty:
                return True

        return False
    
    def _euclidean_distance_2d(self, pos1, pos2) -> float:
        """Calculate 2D Euclidean distance (ignoring y/height)."""
        if isinstance(pos1, Position):
            x1, z1 = pos1.x, pos1.z
        else:
            x1, z1 = pos1[0], pos1[2]
        if isinstance(pos2, Position):
            x2, z2 = pos2.x, pos2.z
        else:
            x2, z2 = pos2[0], pos2[2]
        return math.sqrt((x1 - x2)**2 + (z1 - z2)**2)

    def _find_path_core(self, start: Position, end: Position) -> List[Position]:
        """기준선 기반 필터링된 그래프에서 A* 경로를 찾아 노드 리스트를 반환합니다.
        
        드론의 _find_path_core와 동일한 로직이지만 2D(지면)에서만 동작합니다.
        """
        if start == end:
            return [Position(start.x, 0, start.z)]

        nodes: List[Position] = [start, end]
        
        # 시작/끝 건물 ID
        start_building_id = start.building_id
        end_building_id = end.building_id

        # 관련 건물 필터링 (시작/끝 건물 제외)
        relevant_buildings = self._filter_relevant_buildings(
            start, end, 
            start_building_id=start_building_id,
            end_building_id=end_building_id
        )
        relevant_building_ids = set()

        # 관련 건물의 노드 추가 (오프셋 적용됨)
        for building in relevant_buildings:
            relevant_building_ids.add(building.id)
            vertices = self._get_building_vertices_2d(building)
            for vertex in vertices:
                if vertex in nodes: raise
                nodes.append(vertex)
        
        # 시작/끝 건물의 꼭짓점도 추가 (건물 내부에서 외부로 나가기 위해)
        start_building_vertices_start_idx = len(nodes)
        if start_building_id is not None:
            start_building = self.map.buildings[start_building_id]
            vertices = self._get_building_vertices_2d(start_building)
            for vertex in vertices:
                if vertex not in nodes:
                    nodes.append(vertex)
        start_building_vertices_end_idx = len(nodes)
        
        end_building_vertices_start_idx = len(nodes)
        if end_building_id is not None and end_building_id != start_building_id:
            end_building = self.map.buildings[end_building_id]
            vertices = self._get_building_vertices_2d(end_building)
            for vertex in vertices:
                if vertex not in nodes:
                    nodes.append(vertex)
        end_building_vertices_end_idx = len(nodes)
        
        # 충돌 검사에 시작/끝 건물도 포함
        if start_building_id is not None:
            relevant_building_ids.add(start_building_id)
        if end_building_id is not None:
            relevant_building_ids.add(end_building_id)

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

            if x == 1: break  # 끝점(index 1)에 도달
            if dist[x] != d: continue
            
            for y in range(n):
                if x == y: continue
                p1 = nodes[x]
                p2 = nodes[y]
                d_ = self._euclidean_distance_2d(nodes[x], nodes[y]) + d
                if dist[y] != -1 and d_ >= dist[y]: continue
                
                # 특수 케이스 체크
                is_start_to_start_building = (x == 0 and start_building_vertices_start_idx <= y < start_building_vertices_end_idx)
                is_end_building_to_end = (y == 1 and end_building_vertices_start_idx <= x < end_building_vertices_end_idx)
                is_same_building_direct = (x == 0 and y == 1 and start_building_id == end_building_id and start_building_id is not None)
                is_any_to_end = (y == 1)  # 끝점으로 가는 모든 경로
                is_start_to_any = (x == 0)  # 시작점에서 나가는 모든 경로
                
                if is_start_to_start_building or is_end_building_to_end or is_same_building_direct:
                    # 시작/끝 건물 내 이동은 항상 허용 (충돌 검사 생략)
                    pass
                elif is_any_to_end:
                    # 끝점으로 들어가는 경우: 끝 건물만 제외하고 다른 건물은 충돌 검사
                    excluded = {end_building_id} if end_building_id is not None else set()
                    if self._segment_collides_2d(p1, p2, excluded, building_ids_to_check=relevant_building_ids):
                        continue
                elif is_start_to_any:
                    # 시작점에서 나가는 경우: 시작 건물만 제외하고 다른 건물은 충돌 검사
                    excluded = {start_building_id} if start_building_id is not None else set()
                    if self._segment_collides_2d(p1, p2, excluded, building_ids_to_check=relevant_building_ids):
                        continue
                else:
                    # 일반 충돌 검사
                    excluded = set()
                    
                    # 같은 건물의 꼭짓점 간 이동은 해당 건물 제외
                    if p1.building_id == p2.building_id and p1.building_id is not None:
                        excluded.add(p1.building_id)
                    
                    if self._segment_collides_2d(p1, p2, excluded, building_ids_to_check=relevant_building_ids):
                        continue

                dist[y] = d_
                connection[y] = x
                heapq.heappush(queue, (d_ + self._euclidean_distance_2d((p2.x, 0, p2.z), (end.x, 0, end.z)), d_, y))

        x = 1
        if connection[x] == None:
            print(f"⚠️  MotorbikeRouting: No path found")
            return []

        route = [nodes[x]]
        while x != 0:
            x = connection[x]
            route.append(nodes[x])
        route.reverse()
        return route

    def calculate_route_rec(self, start: Position, end: Position, depth=0) -> List[Position]:
        """재귀적으로 2D 경로를 계산합니다."""
        if depth > 100: return None
        
        # 직접 경로 안전 검사
        if start.building_id is not None and start.building_id == end.building_id:
            # 같은 건물 내 이동은 직접 경로 허용
            is_direct_path_safe = True
        else:
            # 시작/끝 건물을 제외하고 다른 건물만 충돌 검사
            excluded = {start.building_id, end.building_id} - {None}
            is_direct_path_safe = not self._segment_collides_2d(start, end, excluded_building_ids=excluded)
        
        if is_direct_path_safe:
            return [start, end]

        route = self._find_path_core(start, end)
        if not route or len(route) < 2:
            return None

        full_route = [route[0]]
        for i in range(len(route) - 1):
            a = route[i]
            b = route[i+1]
            # 각 단계도 건물 충돌 검사 (해당 노드의 건물만 제외)
            is_step_safe = not self._segment_collides_2d(a, b, excluded_building_ids={a.building_id, b.building_id})
            if is_step_safe:
                full_route.append(b)
            else:
                extended_route = self.calculate_route_rec(a, b, depth + 1)
                if not extended_route: return None

                full_route.extend(extended_route[1:])

        return full_route

    def calculate_route(self, start: Position, waypoints: List[Position], end: Position) -> List[Position]:
        """점진적 경로 탐색을 사용하여 전체 2D 경로를 계산합니다.
        
        드론의 calculate_route와 동일한 로직이지만 y좌표를 0으로 고정합니다.
        """
        # 모든 포인트를 지면 레벨로 설정
        start_2d = start.copy()
        start_2d.y = 0
        
        waypoints_2d = []
        for wp in waypoints:
            wp_2d = wp.copy()
            wp_2d.y = 0
            waypoints_2d.append(wp_2d)
        
        end_2d = end.copy()
        end_2d.y = 0
        
        full_route = [start_2d]
        segment_targets = waypoints_2d + [end_2d]

        for p in full_route + segment_targets:
            building = self.map.get_building_containing_point(p)
            p.building_id = building.id if building else None

        current_segment_start = start_2d

        for segment_end in segment_targets:
            working_path_segment = self.calculate_route_rec(current_segment_start, segment_end)
            if not working_path_segment:
                print(f"❌ MotorbikeRouting Error: Path calculation failed")
                return []
            else:
                full_route.extend(working_path_segment[1:])

            current_segment_start = segment_end

        # 모든 경로 포인트의 y좌표를 0으로 보장
        for pos in full_route:
            pos.y = 0

        return full_route
    
    def calculate_distance(self, route: List[Position]) -> float:
        """Calculate total 2D distance of route (ignoring height)."""
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self._euclidean_distance_2d(route[i], route[i + 1])
        
        return total_distance


class MotorbikeRouteOptimizer:
    """Route optimizer for motorbike deliveries."""
    
    def __init__(self, routing_algorithm: RoutingAlgorithm = None):
        self.routing_algorithm = routing_algorithm or MotorbikeRouting()
    
    def set_routing_algorithm(self, routing_algorithm: RoutingAlgorithm):
        self.routing_algorithm = routing_algorithm
    
    def optimize_delivery_route(self, motorbike, order, visualize: bool = False) -> List[Position]:
        """Calculate optimal route for motorbike delivery.
        
        Route: depot -> store (ground entry) -> customer (ground entry) -> depot
        All waypoints are at ground level.
        """
        from ..models.entities import Motorbike
        
        depot_pos = motorbike.depot.get_center().copy()
        depot_pos.y = 0
        
        # Get ground-level entry points for store and customer
        store_pos = order.store_position.copy()
        store_pos.y = 0
        
        customer_pos = order.customer_position.copy()
        customer_pos.y = 0
        
        # Calculate route
        waypoints = [store_pos]
        route = self.routing_algorithm.calculate_route(depot_pos, waypoints, customer_pos)
        
        # Add return to depot
        if route:
            return_route = self.routing_algorithm.calculate_route(customer_pos, [], depot_pos)
            if return_route:
                route.extend(return_route[1:])
        
        return route
    
    def optimize_multi_delivery_route(self, motorbike, orders: List, visualize: bool = False) -> List[Position]:
        """Calculate optimal route for multiple deliveries.
        
        Simple approach: visit stores then customers in order.
        """
        from ..models.entities import Motorbike
        
        depot_pos = motorbike.depot.get_center().copy()
        depot_pos.y = 0
        
        # Collect all stops at ground level
        waypoints = []
        for order in orders:
            store_pos = order.store_position.copy()
            store_pos.y = 0
            waypoints.append(store_pos)
        
        for order in orders:
            customer_pos = order.customer_position.copy()
            customer_pos.y = 0
            waypoints.append(customer_pos)
        
        # Calculate route through all waypoints
        if not waypoints:
            return [depot_pos]
        
        end_pos = depot_pos.copy()
        route = self.routing_algorithm.calculate_route(depot_pos, waypoints, end_pos)
        
        return route if route else [depot_pos]
