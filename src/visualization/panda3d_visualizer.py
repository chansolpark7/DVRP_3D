"""
Panda3D-based 3D Visualizer for DVRP Simulation

Coordinate System Note:
- Simulation/Ursina uses Y-up: (X=horizontal, Y=height, Z=depth)
- Panda3D uses Z-up: (X=horizontal, Y=depth, Z=height)
- This visualizer converts coordinates: sim(x, y, z) -> panda3d(x, z, y)
"""

import math
from typing import Dict, List, Optional, Tuple
from typing import Sequence as TypingSequence

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import ClockObject
from panda3d.core import (
    NodePath, GeomNode, Geom, GeomVertexFormat, GeomVertexData,
    GeomVertexWriter, GeomTriangles, Vec3, Vec4, Point3, LVector3f,
    AmbientLight, DirectionalLight, TextNode,
    CardMaker, Texture, PNMImage, WindowProperties, AntialiasAttrib,
    TransparencyAttrib, CollisionNode, CollisionBox, CollisionSphere,
    loadPrcFileData
)
from direct.gui.OnscreenText import OnscreenText

import config
from src.models.entities import Map, Building, Depot, Drone, EntityType, Position


# Configure Panda3D before ShowBase initialization
loadPrcFileData('', 'window-title DVRP 3D Simulation')
loadPrcFileData('', 'show-frame-rate-meter true')
loadPrcFileData('', 'sync-video false')


def _sim_to_panda3d(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert simulation coordinates (Y-up) to Panda3D coordinates (Z-up)
    
    Simulation: (X=horizontal, Y=height, Z=depth)
    Panda3D:    (X=horizontal, Y=depth, Z=height)
    """
    return (x, z, y)


def _sim_pos_to_panda3d(pos: Position) -> Tuple[float, float, float]:
    """Convert Position object to Panda3D coordinates"""
    return _sim_to_panda3d(pos.x, pos.y, pos.z)


def _polygon_area(points: TypingSequence[Tuple[float, float]]) -> float:
    """Calculate signed area of a 2D polygon (in X-Z plane of simulation)"""
    area = 0.0
    for i in range(len(points)):
        x1, z1 = points[i]
        x2, z2 = points[(i + 1) % len(points)]
        area += x1 * z2 - x2 * z1
    return area / 2.0


def _ensure_ccw(points: TypingSequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Ensure points are in counter-clockwise order"""
    pts = list(points)
    if _polygon_area(pts) < 0:
        pts.reverse()
    return pts


def _is_convex(a, b, c) -> bool:
    """Check if three points form a convex angle"""
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 0


def _point_in_triangle(p, a, b, c) -> bool:
    """Check if point p is inside triangle abc using barycentric coordinates"""
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denom) < 1e-8:
        return False
    w1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
    w2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
    w3 = 1 - w1 - w2
    return (0 < w1 < 1) and (0 < w2 < 1) and (0 < w3 < 1)


def _triangulate_polygon(points: TypingSequence[Tuple[float, float]]) -> List[Tuple[int, int, int]]:
    """Triangulate a polygon using ear clipping algorithm"""
    pts = _ensure_ccw(points)
    if len(pts) < 3:
        return []

    indices = list(range(len(pts)))
    triangles: List[Tuple[int, int, int]] = []
    guard = 0

    while len(indices) > 3 and guard < 2000:
        ear_found = False
        for i in range(len(indices)):
            prev_idx = indices[i - 1]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % len(indices)]

            a, b, c = pts[prev_idx], pts[curr_idx], pts[next_idx]
            if not _is_convex(a, b, c):
                continue

            if any(
                _point_in_triangle(pts[k], a, b, c)
                for k in indices
                if k not in (prev_idx, curr_idx, next_idx)
            ):
                continue

            triangles.append((prev_idx, curr_idx, next_idx))
            del indices[i]
            ear_found = True
            break

        if not ear_found:
            break
        guard += 1

    if len(indices) == 3:
        triangles.append(tuple(indices))

    if not triangles:
        for i in range(1, len(pts) - 1):
            triangles.append((0, i, i + 1))

    return triangles


def _build_prism_geom(points: TypingSequence[Tuple[float, float]], height: float, color: Vec4) -> Optional[GeomNode]:
    """Build a prism geometry from polygon base points
    
    Points are in simulation X-Z plane (horizontal ground plane).
    Height is along simulation Y axis (vertical).
    Output geometry is in Panda3D coordinates (Z-up).
    """
    if len(points) < 3 or height <= 0:
        return None

    pts = _ensure_ccw(points)
    n = len(pts)
    half_height = height / 2

    # Create vertex data format
    vformat = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData('prism', vformat, Geom.UH_static)
    
    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    color_writer = GeomVertexWriter(vdata, 'color')

    # Points are in simulation (x, z) which maps to Panda3D (x, y)
    # Height (simulation y) maps to Panda3D z
    
    # Bottom vertices (z = -half_height in Panda3D)
    for sim_x, sim_z in pts:
        # Panda3D: (x, y, z) = (sim_x, sim_z, -half_height)
        vertex_writer.add_data3(sim_x, sim_z, -half_height)
        color_writer.add_data4(color)
    
    # Top vertices (z = half_height in Panda3D)
    for sim_x, sim_z in pts:
        vertex_writer.add_data3(sim_x, sim_z, half_height)
        color_writer.add_data4(color)

    # Create triangles
    prim = GeomTriangles(Geom.UH_static)
    
    # Triangulate top and bottom faces
    tri_indices = _triangulate_polygon(pts)
    
    for a, b, c in tri_indices:
        # Top face (in Panda3D, looking down -Z)
        prim.add_vertices(a + n, c + n, b + n)
        # Bottom face
        prim.add_vertices(a, b, c)

    # Side faces
    for i in range(n):
        j = (i + 1) % n
        prim.add_vertices(i, j + n, j)
        prim.add_vertices(i, i + n, j + n)

    geom = Geom(vdata)
    geom.add_primitive(prim)
    
    node = GeomNode('prism')
    node.add_geom(geom)
    
    return node


def _create_box_geom(width: float, height: float, depth: float, color: Vec4) -> GeomNode:
    """Create a simple box geometry in Panda3D coordinates
    
    Args:
        width: Size along X axis (simulation and Panda3D)
        height: Size along simulation Y axis (Panda3D Z)
        depth: Size along simulation Z axis (Panda3D Y)
    """
    vformat = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData('box', vformat, Geom.UH_static)
    
    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    color_writer = GeomVertexWriter(vdata, 'color')

    # Panda3D coordinates: width=X, depth=Y, height=Z
    hw = width / 2   # half width (X)
    hd = depth / 2   # half depth (Y in Panda3D, Z in simulation)
    hh = height / 2  # half height (Z in Panda3D, Y in simulation)
    
    # 8 vertices of a box in Panda3D coords (X, Y, Z)
    vertices = [
        (-hw, -hd, -hh), (hw, -hd, -hh), (hw, hd, -hh), (-hw, hd, -hh),  # bottom (Z-)
        (-hw, -hd, hh), (hw, -hd, hh), (hw, hd, hh), (-hw, hd, hh)       # top (Z+)
    ]
    
    for v in vertices:
        vertex_writer.add_data3(*v)
        color_writer.add_data4(color)

    prim = GeomTriangles(Geom.UH_static)
    
    # 6 faces (2 triangles each)
    faces = [
        (0, 2, 1), (0, 3, 2),  # bottom (Z-)
        (4, 5, 6), (4, 6, 7),  # top (Z+)
        (0, 1, 5), (0, 5, 4),  # front (Y-)
        (2, 3, 7), (2, 7, 6),  # back (Y+)
        (0, 4, 7), (0, 7, 3),  # left (X-)
        (1, 2, 6), (1, 6, 5),  # right (X+)
    ]
    
    for f in faces:
        prim.add_vertices(*f)

    geom = Geom(vdata)
    geom.add_primitive(prim)
    
    node = GeomNode('box')
    node.add_geom(geom)
    
    return node


def _create_cylinder_geom(radius: float, height: float, segments: int, color: Vec4) -> GeomNode:
    """Create a cylinder geometry aligned with Panda3D Z axis (simulation Y)"""
    vformat = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData('cylinder', vformat, Geom.UH_static)
    
    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    color_writer = GeomVertexWriter(vdata, 'color')

    hh = height / 2  # half height along Z (Panda3D)
    
    # Bottom center
    vertex_writer.add_data3(0, 0, -hh)
    color_writer.add_data4(color)
    
    # Bottom ring
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertex_writer.add_data3(x, y, -hh)
        color_writer.add_data4(color)
    
    # Top center
    vertex_writer.add_data3(0, 0, hh)
    color_writer.add_data4(color)
    
    # Top ring
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertex_writer.add_data3(x, y, hh)
        color_writer.add_data4(color)

    prim = GeomTriangles(Geom.UH_static)
    
    # Bottom face
    for i in range(segments):
        next_i = (i + 1) % segments + 1
        prim.add_vertices(0, next_i, i + 1)
    
    # Top face
    top_center = segments + 1
    for i in range(segments):
        curr = top_center + 1 + i
        next_i = top_center + 1 + (i + 1) % segments
        prim.add_vertices(top_center, curr, next_i)
    
    # Side faces
    for i in range(segments):
        b1 = i + 1
        b2 = (i + 1) % segments + 1
        t1 = top_center + 1 + i
        t2 = top_center + 1 + (i + 1) % segments
        prim.add_vertices(b1, b2, t2)
        prim.add_vertices(b1, t2, t1)

    geom = Geom(vdata)
    geom.add_primitive(prim)
    
    node = GeomNode('cylinder')
    node.add_geom(geom)
    
    return node


def _create_sphere_geom(radius: float, segments: int, rings: int, color: Vec4) -> GeomNode:
    """Create a sphere geometry"""
    vformat = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData('sphere', vformat, Geom.UH_static)
    
    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    color_writer = GeomVertexWriter(vdata, 'color')

    # Generate vertices - sphere aligned with Z-up
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            vertex_writer.add_data3(x, y, z)
            color_writer.add_data4(color)

    prim = GeomTriangles(Geom.UH_static)
    
    # Generate triangles
    for i in range(rings):
        for j in range(segments):
            curr = i * segments + j
            next_j = i * segments + (j + 1) % segments
            below = (i + 1) * segments + j
            below_next = (i + 1) * segments + (j + 1) % segments
            
            if i != 0:
                prim.add_vertices(curr, below, next_j)
            if i != rings - 1:
                prim.add_vertices(next_j, below, below_next)

    geom = Geom(vdata)
    geom.add_primitive(prim)
    
    node = GeomNode('sphere')
    node.add_geom(geom)
    
    return node


class Panda3DVisualizer(ShowBase):
    """3D visualization for DVRP simulation using Panda3D engine
    
    Handles coordinate conversion from simulation (Y-up) to Panda3D (Z-up).
    """
    
    def __init__(self, map_width: float = 1000, map_depth: float = 1000):
        """Initialize Panda3D visualizer
        
        Args:
            map_width: Width of the map (X-axis, same in both coordinate systems)
            map_depth: Depth of the map (simulation Z-axis, Panda3D Y-axis)
        """
        ShowBase.__init__(self)
        
        # Store map dimensions (in simulation coordinates - actual meters)
        self.map_width = map_width
        self.map_depth = map_depth
        self.map_size = max(map_width, map_depth)  # Reference size for scaling
        self.height_scale = getattr(config, "VISUALIZATION_HEIGHT_SCALE", 1.0)
        
        # Global visualization scale - makes everything bigger/smaller uniformly
        self.vis_scale = getattr(config, "VISUALIZATION_SCALE", 1.0)
        
        # Auto-calculate camera speed and distances based on map size
        # Base reference: 2000m map with speed 500
        self.map_scale_factor = self.map_size / 2000.0
        
        # Entity scale will be calculated based on average building size
        # This ensures depot/drone sizes are proportional to actual buildings
        self.entity_scale = 1.0  # Default, will be updated in create_map_entities
        
        # Fallback size_scale based on map dimensions (used for initial setup)
        self.size_scale = min(map_width, map_depth) / 2000.0
        
        # Enable antialiasing
        self.render.setAntialias(AntialiasAttrib.MAuto)
        
        # Create scene root node for global scaling
        self.scene_root = self.render.attachNewNode("scene_root")
        self.scene_root.setScale(self.vis_scale)
        
        print(f"Map size: {map_width:.0f}m x {map_depth:.0f}m (scale factor: {self.map_scale_factor:.2f})")
        
        # Setup camera
        self._setup_camera()
        
        # Create ground plane
        self._create_ground()
        
        # Setup lighting
        self._setup_lighting()
        
        # Create sky (background color)
        self.setBackgroundColor(0.53, 0.81, 0.92, 1)  # Sky blue
        
        # Entity storage
        self.building_nodes: List[NodePath] = []
        self.depot_nodes: List[NodePath] = []
        self.drone_nodes: Dict[int, NodePath] = {}
        self.drone_labels: Dict[int, NodePath] = {}
        self.failure_markers: Dict[int, NodePath] = {}
        
        # Camera control state (in Panda3D coordinates, auto-scaled to map size)
        self._mouse_pressed = False
        self._last_mouse_pos = None
        self._camera_heading = 45
        self._camera_pitch = -30
        # Camera distance - start close for detail view
        self._camera_distance = self.map_size * 0.15 * self.vis_scale
        # Camera target in Panda3D coords: center of map at ground level
        self._camera_target = Point3(
            map_width/2 * self.vis_scale, 
            map_depth/2 * self.vis_scale, 
            0
        )
        
        # Setup input handling
        self._setup_input()
        
        # Update camera position
        self._update_camera_position()
        
        # Key state tracking for input handling
        self._key_states = {}
        
    def _setup_camera(self):
        """Setup camera for 3D view"""
        # Disable default mouse camera control
        self.disableMouse()
        
        # Set initial camera position (Panda3D coords)
        self.camera.setPos(self.map_width/2, -500, 500)
        self.camera.lookAt(self.map_width/2, self.map_depth/2, 0)
        
    def _setup_input(self):
        """Setup keyboard and mouse input handling"""
        # Mouse button events
        self.accept('mouse1', self._on_mouse_press, ['left'])
        self.accept('mouse1-up', self._on_mouse_release, ['left'])
        self.accept('mouse2', self._on_mouse_press, ['middle'])
        self.accept('mouse2-up', self._on_mouse_release, ['middle'])
        self.accept('mouse3', self._on_mouse_press, ['right'])
        self.accept('mouse3-up', self._on_mouse_release, ['right'])
        
        # Scroll wheel
        self.accept('wheel_up', self._on_scroll, [-50])
        self.accept('wheel_down', self._on_scroll, [50])
        
        # Keyboard
        self.accept('escape', self._on_escape)
        self.accept('h', self._toggle_help)
        
        # Camera movement keys
        for key in ['w', 'a', 's', 'd', 'q', 'e']:
            self.accept(key, self._set_key_state, [key, True])
            self.accept(f'{key}-up', self._set_key_state, [key, False])
        
        # Add task for continuous camera movement
        self.taskMgr.add(self._camera_move_task, 'camera_move_task')
        self.taskMgr.add(self._mouse_look_task, 'mouse_look_task')
        
    def _set_key_state(self, key, state):
        """Track key press state"""
        self._key_states[key] = state
        
    def _camera_move_task(self, task):
        """Task to handle continuous camera movement"""
        dt = ClockObject.getGlobalClock().getDt()
        # Speed scales with map size - reduced for finer control
        base_speed = getattr(config, "CAMERA_MOVE_SPEED", 500.0)
        speed = base_speed * self.map_scale_factor * self.vis_scale * dt * 0.3
        
        # Calculate forward and right vectors based on camera heading
        heading_rad = math.radians(self._camera_heading)
        
        # Forward vector (direction camera is looking in X-Y plane)
        forward_x = -math.sin(heading_rad)
        forward_y = math.cos(heading_rad)
        
        # Right vector (perpendicular to forward)
        right_x = math.cos(heading_rad)
        right_y = math.sin(heading_rad)
        
        # W/S: Move forward/backward relative to camera view
        if self._key_states.get('w'):
            self._camera_target.x += forward_x * speed
            self._camera_target.y += forward_y * speed
        if self._key_states.get('s'):
            self._camera_target.x -= forward_x * speed
            self._camera_target.y -= forward_y * speed
            
        # A/D: Move left/right relative to camera view (swapped direction)
        if self._key_states.get('a'):
            self._camera_target.x -= right_x * speed
            self._camera_target.y -= right_y * speed
        if self._key_states.get('d'):
            self._camera_target.x += right_x * speed
            self._camera_target.y += right_y * speed
            
        # Q/E: Move down/up (along Panda3D Z axis)
        if self._key_states.get('q'):
            self._camera_target.z -= speed
        if self._key_states.get('e'):
            self._camera_target.z += speed
            
        if any(self._key_states.get(k) for k in ['w', 'a', 's', 'd', 'q', 'e']):
            self._update_camera_position()
            
        return Task.cont
    
    def _mouse_look_task(self, task):
        """Task to handle mouse look (camera rotation)"""
        if not self._mouse_pressed:
            self._last_mouse_pos = None
            return Task.cont
            
        if not self.mouseWatcherNode.hasMouse():
            return Task.cont
            
        mouse_x = self.mouseWatcherNode.getMouseX()
        mouse_y = self.mouseWatcherNode.getMouseY()
        
        if self._last_mouse_pos is not None:
            dx = mouse_x - self._last_mouse_pos[0]
            dy = mouse_y - self._last_mouse_pos[1]
            
            # Rotate camera - adjusted sensitivity and corrected direction
            # Horizontal mouse movement rotates around Z axis (heading)
            # Vertical mouse movement changes pitch (looking up/down)
            self._camera_heading -= dx * 150  # Drag right = view moves right
            self._camera_pitch = max(-85, min(-5, self._camera_pitch - dy * 100))  # Positive dy = look up
            
            self._update_camera_position()
            
        self._last_mouse_pos = (mouse_x, mouse_y)
        
        return Task.cont
        
    def _on_mouse_press(self, button):
        """Handle mouse button press"""
        if button in ['middle', 'right']:
            self._mouse_pressed = True
            
    def _on_mouse_release(self, button):
        """Handle mouse button release"""
        if button in ['middle', 'right']:
            self._mouse_pressed = False
            self._last_mouse_pos = None
            
    def _on_scroll(self, delta):
        """Handle mouse scroll wheel"""
        # Zoom speed and range scale with map size
        scaled_delta = delta * self.map_scale_factor * self.vis_scale * 0.5  # Slower zoom for precision
        min_dist = 5 * self.vis_scale  # Can zoom in extremely close (5m)
        max_dist = self.map_size * 1.5 * self.vis_scale  # Can zoom out to see full map
        self._camera_distance = max(min_dist, min(max_dist, self._camera_distance + scaled_delta))
        self._update_camera_position()
        
    def _on_escape(self):
        """Handle escape key"""
        self.userExit()
        
    def _toggle_help(self):
        """Toggle help text visibility"""
        if hasattr(self, 'help_text') and self.help_text:
            if self.help_text.isHidden():
                self.help_text.show()
            else:
                self.help_text.hide()
                
    def _update_camera_position(self):
        """Update camera position based on orbit parameters (Panda3D Z-up)"""
        heading_rad = math.radians(self._camera_heading)
        pitch_rad = math.radians(self._camera_pitch)
        
        # Calculate camera position in Panda3D coords
        # Orbit around target in X-Y plane, with Z as height
        x = self._camera_target.x + self._camera_distance * math.sin(heading_rad) * math.cos(pitch_rad)
        y = self._camera_target.y - self._camera_distance * math.cos(heading_rad) * math.cos(pitch_rad)
        z = self._camera_target.z + self._camera_distance * math.sin(-pitch_rad)
        
        self.camera.setPos(x, y, z)
        self.camera.lookAt(self._camera_target)
        
    def _create_ground(self):
        """Create ground plane (in Panda3D X-Y plane at Z=0)"""
        # Create a large flat plane using CardMaker
        cm = CardMaker('ground')
        cm.setFrame(0, self.map_width, 0, self.map_depth)
        
        ground_node = self.scene_root.attachNewNode(cm.generate())
        # Card is in X-Y plane by default in Panda3D, which is correct for Z-up
        ground_node.setP(-90)  # Rotate to be horizontal (X-Y plane)
        ground_node.setPos(0, 0, 0)
        ground_node.setColor(0.3, 0.6, 0.3, 1)  # Grass green
        
        self.ground = ground_node
        
    def _setup_lighting(self):
        """Setup scene lighting"""
        # Ambient light
        ambient = AmbientLight('ambient')
        ambient.setColor(Vec4(0.4, 0.4, 0.4, 1))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)
        
        # Directional light (sun) - shining from above and side
        sun = DirectionalLight('sun')
        sun.setColor(Vec4(0.8, 0.8, 0.8, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -45, 0)
        self.render.setLight(sun_np)
        
        self.ambient_light = ambient_np
        self.sun_light = sun_np
        
    def _format_drone_label(self, drone: Drone) -> str:
        """Format drone label text"""
        battery_pct = max(0.0, min(1.0, getattr(drone, 'battery_level', 0.0))) * 100
        return f"D{drone.id}: {battery_pct:.0f}%"
        
    def _get_scaled_height_and_center(self, building: Building) -> Tuple[float, float]:
        """Get scaled height and center Y position for a building (in simulation coords)"""
        scaled_height = building.height * self.height_scale
        base_y = building.position.y - (building.height / 2)
        scaled_center_y = base_y + (scaled_height / 2)
        return scaled_height, scaled_center_y
        
    def _create_building_node(
        self,
        building: Building,
        bldg_color: Vec4,
        scaled_height: float,
        scaled_center_y: float
    ) -> Optional[NodePath]:
        """Create a building node with proper coordinate conversion"""
        if scaled_height <= 0:
            return None
            
        footprints = getattr(building, "footprints", None)
        
        if footprints:
            # Create prism from footprints
            center_x = building.position.x
            center_z = building.position.z  # simulation Z = depth
            
            parent_node = self.scene_root.attachNewNode(f"building_{building.id}")
            
            for footprint in footprints:
                # footprint points are in simulation (x, z) coords
                local_points = [(x - center_x, z - center_z) for x, z in footprint]
                geom_node = _build_prism_geom(local_points, scaled_height, bldg_color)
                if geom_node:
                    np = parent_node.attachNewNode(geom_node)
                    np.setTransparency(TransparencyAttrib.MAlpha)
            
            # Position in Panda3D coords: (sim_x, sim_z, sim_y)
            parent_node.setPos(center_x, center_z, scaled_center_y)
            return parent_node
        
        # Fallback to box
        geom_node = _create_box_geom(building.width, scaled_height, building.depth, bldg_color)
        np = self.scene_root.attachNewNode(geom_node)
        # Position in Panda3D coords
        np.setPos(building.position.x, building.position.z, scaled_center_y)
        np.setTransparency(TransparencyAttrib.MAlpha)
        return np
        
    def _create_3d_text(self, text: str, sim_pos: Tuple[float, float, float], 
                        scale: float = 2, color: Vec4 = Vec4(1, 1, 1, 1)) -> NodePath:
        """Create 3D text that billboards toward camera
        
        Args:
            text: Text to display
            sim_pos: Position in simulation coordinates (x, y, z) where y=height
            scale: Text scale
            color: Text color
        """
        text_node = TextNode(f'text_{id(text)}')
        text_node.setText(text)
        text_node.setAlign(TextNode.ACenter)
        text_node.setTextColor(color)
        
        text_np = self.scene_root.attachNewNode(text_node)
        # Convert to Panda3D coords
        panda_pos = _sim_to_panda3d(sim_pos[0], sim_pos[1], sim_pos[2])
        text_np.setPos(panda_pos[0], panda_pos[1], panda_pos[2])
        text_np.setScale(scale)
        text_np.setBillboardPointEye()  # Always face camera
        
        return text_np
        
    def create_map_entities(self, map_data: Map):
        """Render map entities (buildings, stores, customers, depots) in 3D
        
        Args:
            map_data: Map object containing buildings and depots
        """
        # Clear existing entities
        self.clear_map_entities()
        
        # Calculate entity scale based on average building size
        if map_data.buildings:
            avg_building_size = sum(
                (b.width + b.depth) / 2 for b in map_data.buildings
            ) / len(map_data.buildings)
            # Scale relative to a "standard" building size of 50 units
            self.entity_scale = avg_building_size / 50.0
        else:
            self.entity_scale = self.size_scale
        
        # Color definitions (RGBA 0-1)
        color_store = Vec4(0, 1, 0, 0.8)      # Green
        color_customer = Vec4(1, 0, 0, 0.8)   # Red
        color_building = Vec4(1, 1, 1, 0.9)   # White
        color_depot = Vec4(0, 0, 1, 0.9)      # Blue
        
        # Render buildings
        for building in map_data.buildings:
            # Determine color based on entity type
            if building.entity_type == EntityType.STORE:
                bldg_color = color_store
            elif building.entity_type == EntityType.CUSTOMER:
                bldg_color = color_customer
            else:
                bldg_color = color_building
                
            scaled_height, scaled_center_y = self._get_scaled_height_and_center(building)
            if scaled_height <= 0:
                continue
                
            building_node = self._create_building_node(
                building,
                bldg_color,
                scaled_height,
                scaled_center_y
            )
            if building_node is None:
                continue
            self.building_nodes.append(building_node)
            
            # Add label for stores and customers
            if building.entity_type in [EntityType.STORE, EntityType.CUSTOMER]:
                label_text = "STORE" if building.entity_type == EntityType.STORE else "CUSTOMER"
                # Label position in simulation coords
                label_offset = 5 * self.entity_scale
                label_sim_pos = (
                    building.position.x, 
                    scaled_center_y + scaled_height/2 + label_offset,  # y = height
                    building.position.z
                )
                label_scale = 8.0 * self.entity_scale  # Larger labels for visibility
                label = self._create_3d_text(
                    f"{label_text}\n{building.id}",
                    label_sim_pos,
                    scale=label_scale
                )
                self.building_nodes.append(label)
                
        # Render depots
        for depot in map_data.depots:
            # Depot size scales with average building size
            depot_radius = 2.0 * self.entity_scale
            depot_height = 1.0 * self.entity_scale
            
            geom_node = _create_cylinder_geom(depot_radius, depot_height, 16, color_depot)
            depot_node = self.scene_root.attachNewNode(geom_node)
            # Position in Panda3D coords
            panda_pos = _sim_pos_to_panda3d(depot.position)
            depot_node.setPos(panda_pos[0], panda_pos[1], panda_pos[2] + depot_height/2)
            depot_node.setTransparency(TransparencyAttrib.MAlpha)
            
            self.depot_nodes.append(depot_node)
            
            # Add depot label (simulation coords)
            label_scale = 6.0 * self.entity_scale  # Larger labels for visibility
            depot_label = self._create_3d_text(
                f"DEPOT\n{depot.id}",
                (depot.position.x, depot.position.y + depot_height + 3, depot.position.z),
                scale=label_scale
            )
            self.depot_nodes.append(depot_label)
            
    def clear_map_entities(self):
        """Clear all map entities from the scene"""
        for node in self.building_nodes:
            node.removeNode()
        for node in self.depot_nodes:
            node.removeNode()
            
        self.building_nodes.clear()
        self.depot_nodes.clear()
        
    def update_drone_visuals(self, drones: List[Drone]):
        """Update drone positions and create new drone entities if needed
        
        Args:
            drones: List of active drones to visualize
        """
        if not drones:
            return
            
        active_drone_ids = set()
        
        color_normal = Vec4(1, 1, 0, 1)   # Yellow
        color_collision = Vec4(1, 0, 0, 1)  # Red
        
        for drone in drones:
            active_drone_ids.add(drone.id)
            
            # Get collision status
            collision_status = getattr(drone, 'collision_status', 'none')
            is_visible = (collision_status != 'destination_entry')
            
            # Determine color
            if collision_status == 'accidental':
                drone_color = color_collision
            else:
                drone_color = color_normal
            
            # Convert drone position to Panda3D coords
            panda_pos = _sim_pos_to_panda3d(drone.position)
                
            # Create new drone node if it doesn't exist
            if drone.id not in self.drone_nodes:
                # Drone size scales with average building size
                drone_radius = 1.0 * self.entity_scale
                geom_node = _create_sphere_geom(drone_radius, 12, 8, drone_color)
                drone_node = self.scene_root.attachNewNode(geom_node)
                drone_node.setPos(panda_pos[0], panda_pos[1], panda_pos[2])
                self.drone_nodes[drone.id] = drone_node
                
                # Create label (simulation coords with height offset)
                label_offset = 3 * self.entity_scale
                label_scale = 5.0 * self.entity_scale  # Larger labels for visibility
                label = self._create_3d_text(
                    self._format_drone_label(drone),
                    (drone.position.x, drone.position.y + label_offset, drone.position.z),
                    scale=label_scale,
                    color=Vec4(0, 0, 0, 1)
                )
                self.drone_labels[drone.id] = label
            else:
                # Update existing drone position
                drone_node = self.drone_nodes[drone.id]
                drone_node.setPos(panda_pos[0], panda_pos[1], panda_pos[2])
                
            # Update visibility
            drone_node = self.drone_nodes[drone.id]
            if is_visible:
                drone_node.show()
            else:
                drone_node.hide()
                
            # Update label
            if drone.id in self.drone_labels:
                label = self.drone_labels[drone.id]
                if is_visible:
                    label.show()
                    # Update label position (Panda3D coords)
                    label_offset = 3 * self.entity_scale
                    label_panda_pos = _sim_to_panda3d(
                        drone.position.x, 
                        drone.position.y + label_offset, 
                        drone.position.z
                    )
                    label.setPos(label_panda_pos[0], label_panda_pos[1], label_panda_pos[2])
                    # Update text
                    text_node = label.node()
                    if isinstance(text_node, TextNode):
                        text_node.setText(self._format_drone_label(drone))
                else:
                    label.hide()
                    
        # Remove inactive drones
        inactive_drone_ids = set(self.drone_nodes.keys()) - active_drone_ids
        for drone_id in inactive_drone_ids:
            if drone_id in self.drone_nodes:
                self.drone_nodes[drone_id].removeNode()
                del self.drone_nodes[drone_id]
                
            if drone_id in self.drone_labels:
                self.drone_labels[drone_id].removeNode()
                del self.drone_labels[drone_id]
                
    def clear_drones(self):
        """Clear all drone entities from the scene"""
        for node in self.drone_nodes.values():
            node.removeNode()
        for label in self.drone_labels.values():
            label.removeNode()
            
        self.drone_nodes.clear()
        self.drone_labels.clear()
        self.clear_failure_markers()
        
    def update_failure_markers(self, failure_events: List[Dict]):
        """Update failure markers on the map"""
        if failure_events is None:
            return
            
        seen_ids = set()
        for event in failure_events:
            order_id = event.get('order_id')
            if order_id is None:
                continue
            seen_ids.add(order_id)
            if order_id in self.failure_markers:
                continue
            position = event.get('customer_position') or event.get('store_position')
            if position is None:
                continue
            
            # Create marker at simulation position (with height offset)
            marker_offset = 8 * self.entity_scale
            marker_scale = 8.0 * self.entity_scale  # Larger for visibility
            marker = self._create_3d_text(
                f"FAIL #{order_id}",
                (position.x, max(position.y, 5) + marker_offset, position.z),
                scale=marker_scale,
                color=Vec4(1, 0, 0, 1)
            )
            self.failure_markers[order_id] = marker
            
        stale_ids = set(self.failure_markers.keys()) - seen_ids
        for order_id in stale_ids:
            node = self.failure_markers.pop(order_id, None)
            if node:
                node.removeNode()
                
    def clear_failure_markers(self):
        """Clear all failure markers"""
        for marker in self.failure_markers.values():
            marker.removeNode()
        self.failure_markers.clear()
        
    def update(self):
        """Update function called every frame (compatibility method)"""
        # Panda3D uses tasks instead, but this can be called for compatibility
        pass
        
    def run(self):
        """Start the Panda3D application loop"""
        # This calls ShowBase.run() which starts the main loop
        ShowBase.run(self)
        
    def cleanup(self):
        """Cleanup resources"""
        self.clear_map_entities()
        self.clear_drones()
        
        if hasattr(self, 'ground') and self.ground:
            self.ground.removeNode()


if __name__ == '__main__':
    # Test visualization
    visualizer = Panda3DVisualizer(map_width=1000, map_depth=1000)
    
    # Create test map
    from src.models.entities import Map, Building, Depot, EntityType
    
    test_map = Map(width=1000, depth=1000, max_height=100)
    
    # Add test buildings (simulation coords: x=horizontal, y=height, z=depth)
    test_buildings = [
        Building(1, Position(200, 15, 200), 30, 30, 30, EntityType.STORE),
        Building(2, Position(500, 20, 300), 40, 40, 40, EntityType.CUSTOMER),
        Building(3, Position(700, 25, 600), 50, 50, 50, EntityType.STORE),
        Building(4, Position(300, 10, 700), 20, 20, 20, None),
    ]
    
    for building in test_buildings:
        test_map.add_building(building)
    
    # Add test depot
    test_depot = Depot(1, Position(100, 0, 100), [])
    test_map.add_depot(test_depot)
    
    # Create map entities
    visualizer.create_map_entities(test_map)
    
    # Create test drone
    test_drone = Drone(
        id=1,
        position=Position(150, 50, 150),
        depot=test_depot,
        speed=50
    )
    
    visualizer.update_drone_visuals([test_drone])
    
    # Run visualization
    visualizer.run()
