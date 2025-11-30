"""
Real-world map generator that consumes GeoJSON footprints and converts them
into 3D buildings for the Ursina visualization.
"""

import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import config
from ..map import buildings_processing
from ..models.entities import Building, Map, Position
from .map_generator import MapGenerator
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.ops import unary_union


class RealMapGenerator(MapGenerator):
    """Generate maps by extruding polygon footprints from GeoJSON data."""

    def __init__(
        self,
        map_width: float = config.MAP_WIDTH,
        map_depth: float = config.MAP_DEPTH,
        max_height: float = config.MAX_MAP_HEIGHT,
        seed: Optional[int] = None,
        geojson_path: Optional[str] = None,
        building_limit: Optional[int] = None,
        margin_ratio: float = 0.05,
        simplify_tolerance: float = getattr(config, "FOOTPRINT_SIMPLIFY_TOLERANCE", 0.0),
    ):
        """
        Args:
            geojson_path: Optional override for the buildings.geojson path.
            building_limit: Cap on the number of buildings to import (for testing).
            margin_ratio: Fraction of map size reserved as empty border when scaling.
        """
        super().__init__(map_width, map_depth, max_height, seed)
        repo_root = Path(__file__).resolve().parents[2]
        default_geojson = Path(config.MAP_GEOJSON_PATH)
        if not default_geojson.is_absolute():
            default_geojson = repo_root / default_geojson

        if geojson_path:
            candidate = Path(geojson_path)
            if not candidate.is_absolute():
                candidate = repo_root / candidate
        else:
            candidate = default_geojson

        self.geojson_path = candidate
        self.building_limit = building_limit
        self.margin_ratio = max(0.0, min(0.2, margin_ratio))
        self.simplify_tolerance = max(0.0, simplify_tolerance)

    def generate_map(
        self,
        store_ratio: float = config.STORE_RATIO,
        customer_ratio: float = config.CUSTOMER_RATIO,
    ) -> Map:
        """Generate the full map using GeoJSON footprints."""
        self._ensure_geojson()

        if self.seed is not None:
            random.seed(self.seed)

        features = self._load_features()
        buildings = self._buildings_from_features(features)
        print(f"Loaded {len(buildings)} buildings from GeoJSON")

        for building in buildings:
            self.map.add_building(building)
        self.assign_entities_to_buildings(buildings, store_ratio, customer_ratio)
        return self.map

    def _ensure_geojson(self) -> None:
        """Generate GeoJSON from raw shapefiles if it does not exist."""
        if self.geojson_path.exists():
            return

        print(
            f"GeoJSON 파일이 '{self.geojson_path}'에 존재하지 않습니다. "
            "buildings_processing 파이프라인을 실행합니다..."
        )
        overrides = {"output_geojson_filename": self.geojson_path}
        buildings_processing.generate_building_assets(overrides)

        if not self.geojson_path.exists():
            raise FileNotFoundError(
                f"GeoJSON 생성 실패: {self.geojson_path} 경로에 파일이 없습니다."
            )

    def _load_features(self) -> List[dict]:
        with self.geojson_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        features = data.get("features", [])
        if self.building_limit is not None:
            features = features[: self.building_limit]
        if not features:
            raise ValueError(f"No features found in {self.geojson_path}")
        return features

    def _buildings_from_features(self, features: Sequence[dict]) -> List[Building]:
        def find(parent, x):
            if parent[x] != x:
                parent[x] = find(parent, parent[x])
            return parent[x]

        def union(parent, a, b):
            a = find(parent, a)
            b = find(parent, b)
            if a != b:
                parent[b] = a
                return True
            return False

        bounds = self._compute_bounds(features)
        scale_x, scale_z, offset_x, offset_z = self._compute_scaling(bounds)

        footprints: List[List[Tuple[float, float]]] = []
        polygons: List[Polygon] = []
        heights: List[int] = []

        for idx, feature in enumerate(features):
            geometry = feature.get("geometry")
            if not geometry:
                continue

            outer_ring = self._get_outer_ring(geometry)
            if not outer_ring:
                continue

            scaled = self._scale_polygon(
                outer_ring, bounds, scale_x, scale_z, offset_x, offset_z
            )
            footprint = self._clean_polygon(scaled)
            if len(footprint) < 3:
                continue

            area = self._polygon_area(footprint)
            if area <= 1e-6:
                # Skip degenerate footprints (zero area) even if HEIGHT exists
                continue

            polygon = Polygon(footprint).buffer(config.BUILDING_SAFETY_MARGIN, resolution=1)
            height = self._derive_height(feature.get("properties", {}))

            footprints.append(footprint)
            polygons.append(polygon)
            heights.append(height)

        n = len(polygons)
        independent_polygon_num = n
        parent = [i for i in range(n)]
        strtree = STRtree(polygons)

        for i, polygon_a in enumerate(polygons):
            candidates = strtree.query(polygon_a)
            for j in candidates:
                if i >= j: continue
                polygon_b = polygons[j]
                if polygon_a.intersects(polygon_b):
                    if union(parent, i, j):
                        independent_polygon_num  -= 1

        to_new_index: dict[int, int] = dict()
        for i in range(n):
            if parent[i] == i:
                to_new_index[i] = len(to_new_index)
        print(n, independent_polygon_num, len(to_new_index)) #####

        unioned_polygons: List[List[Polygon]] = [[] for _ in range(independent_polygon_num)]
        unioned_footprints: List[List[Tuple[float, float]]] = [[] for _ in range(independent_polygon_num)]
        unioned_heights = [0] * independent_polygon_num
        for i in range(n):
            index = to_new_index[find(parent, i)]
            unioned_polygons[index].append(polygons[i])
            unioned_footprints[index].append(footprints[i])
            unioned_heights[index] = max(unioned_heights[index], heights[i])

        for i in range(independent_polygon_num):
            unioned_polygons[i] = unary_union(unioned_polygons[i])
            if not isinstance(unioned_polygons[i], Polygon):
                print('type error')
                exit()

        
        buildings: List[Building] = []
        for i in range(independent_polygon_num):
            xs, zs = [], []
            for footprint in unioned_footprints[i]:
                for x, z in footprint:
                    xs.append(x)
                    zs.append(z)
            width = max(max(xs) - min(xs), 5.0)
            depth = max(max(zs) - min(zs), 5.0)
            centroid_x = sum(xs) / len(xs)
            centroid_z = sum(zs) / len(zs)
            position = Position(centroid_x, unioned_heights[i] / 2, centroid_z)
            building = Building(
                id=i,
                position=position,
                width=width,
                height=unioned_heights[i],
                depth=depth,
                footprints=unioned_footprints[i],
                poly=unioned_polygons[i]
            )
            buildings.append(building)

        return buildings

    def _compute_bounds(self, features: Sequence[dict]) -> Tuple[float, float, float, float]:
        min_x = float("inf")
        max_x = float("-inf")
        min_z = float("inf")
        max_z = float("-inf")

        for feature in features:
            geometry = feature.get("geometry")
            if not geometry:
                continue

            for ring in self._iter_rings(geometry):
                for x, z in ring:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_z = min(min_z, z)
                    max_z = max(max_z, z)

        if min_x == float("inf") or min_z == float("inf"):
            raise ValueError("Unable to determine bounds from GeoJSON data")

        return min_x, max_x, min_z, max_z

    def _compute_scaling(
        self, bounds: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        min_x, max_x, min_z, max_z = bounds
        border_x = self.map_width * self.margin_ratio
        border_z = self.map_depth * self.margin_ratio

        usable_width = max(self.map_width - 2 * border_x, 1.0)
        usable_depth = max(self.map_depth - 2 * border_z, 1.0)

        scale_x = usable_width / max(max_x - min_x, 1.0)
        scale_z = usable_depth / max(max_z - min_z, 1.0)

        # Use uniform scaling to avoid footprint distortion
        uniform_scale = min(scale_x, scale_z)
        scale_x = scale_z = uniform_scale

        return scale_x, scale_z, border_x, border_z

    def _derive_height(self, properties: dict) -> float:
        def _as_float(value) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        height = _as_float(properties.get("HEIGHT"))
        return height
        '''
        if not height or height <= 0:
            absolute = _as_float(properties.get("ABSOLUTE_HEIGHT"))
            base = _as_float(properties.get("CONT"))
            if absolute is not None and base is not None:
                derived = absolute - base
                if derived > 0:
                    height = derived

        if not height or height <= 0:
            floors = _as_float(properties.get("GRND_FLR"))
            if floors and floors > 0:
                height = floors * config.FLOOR_HEIGHT

        if not height or height <= 0:
            height = config.BUILDING_MIN_HEIGHT

        floors = max(1, int(round(height / config.FLOOR_HEIGHT)))
        quantized_height = floors * config.FLOOR_HEIGHT

        return max(
            config.BUILDING_MIN_HEIGHT * 0.5,
            min(quantized_height, config.BUILDING_MAX_HEIGHT),
        )
        '''
    def _get_outer_ring(self, geometry: dict) -> Optional[List[Tuple[float, float]]]:
        for ring in self._iter_rings(geometry):
            if ring:
                return [(float(x), float(z)) for x, z in ring]
        return None

    def _iter_rings(self, geometry: dict) -> Iterable[List[Tuple[float, float]]]:
        g_type = geometry.get("type")
        coords = geometry.get("coordinates", [])

        if g_type == "Polygon":
            for ring in coords[:1]:
                yield [(pt[0], pt[1]) for pt in ring]
        elif g_type == "MultiPolygon":
            for polygon in coords:
                if polygon:
                    ring = polygon[0]
                    yield [(pt[0], pt[1]) for pt in ring]

    def _scale_polygon(
        self,
        ring: List[Tuple[float, float]],
        bounds: Tuple[float, float, float, float],
        scale_x: float,
        scale_z: float,
        offset_x: float,
        offset_z: float,
    ) -> List[Tuple[float, float]]:
        min_x, _, min_z, _ = bounds
        scaled = []
        for x, z in ring:
            sx = offset_x + (x - min_x) * scale_x
            sz = offset_z + (z - min_z) * scale_z
            scaled.append((sx, sz))
        return scaled

    def _clean_polygon(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Normalize footprint: dedup, optional simplify, ensure CCW orientation."""
        if not points:
            return []

        cleaned: List[Tuple[float, float]] = []
        for x, z in points:
            if not cleaned or math.dist((x, z), cleaned[-1]) > 1e-4:
                cleaned.append((x, z))

        if len(cleaned) > 2 and math.dist(cleaned[0], cleaned[-1]) <= 1e-4:
            cleaned.pop()

        # Ensure CCW orientation to keep intersection results consistent
        area2 = 0.0
        n = len(cleaned)
        if n >= 3:
            for i in range(n):
                x1, z1 = cleaned[i]
                x2, z2 = cleaned[(i + 1) % n]
                area2 += x1 * z2 - x2 * z1
            if area2 < 0:  # clockwise -> reverse
                cleaned = list(reversed(cleaned))

        return cleaned

    def _polygon_area(self, points: Sequence[Tuple[float, float]]) -> float:
        """Return absolute area of polygon footprint."""
        if not points or len(points) < 3:
            return 0.0

        twice_area = 0.0
        n = len(points)
        for i in range(n):
            x1, z1 = points[i]
            x2, z2 = points[(i + 1) % n]
            twice_area += x1 * z2 - x2 * z1
        return abs(twice_area) / 2.0

    def _polygon_centroid(self, points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate centroid of a polygon footprint."""
        if not points:
            return 0.0, 0.0

        twice_area = 0.0
        cx = 0.0
        cz = 0.0
        n = len(points)

        for i in range(n):
            x1, z1 = points[i]
            x2, z2 = points[(i + 1) % n]
            cross = x1 * z2 - x2 * z1
            twice_area += cross
            cx += (x1 + x2) * cross
            cz += (z1 + z2) * cross

        if abs(twice_area) < 1e-6:
            avg_x = sum(p[0] for p in points) / n
            avg_z = sum(p[1] for p in points) / n
            return avg_x, avg_z

        area = twice_area / 2.0
        cx /= (6.0 * area)
        cz /= (6.0 * area)
        return cx, cz
