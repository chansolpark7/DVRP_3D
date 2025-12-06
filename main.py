#!/usr/bin/env python3
"""
Main application for Drone Vehicle Routing Problem (DVRP) 3D Simulation
도심환경에서의 드론 음식 배달 경로 최적화 및 시뮬레이션 (3D Panda3D)

Author: DVRP Team
Date: 2025
"""
import cProfile
import argparse
import sys
from typing import Optional, Union
import time
import threading

# Add src to path
sys.path.append('src')

from src.models.entities import Map
from src.algorithms.map_generator import MapGenerator, DepotPlacer
from src.algorithms.real_map_generator import RealMapGenerator
from src.algorithms.clustering import MixedClustering, ClusterAnalyzer
from src.algorithms.order_manager import OrderManager
from src.simulation.simulation_engine import SimulationEngine
import config

# Conditional import based on visualizer setting
if config.RUN_VISUALIZER:
    from src.visualization.panda3d_visualizer import Panda3DVisualizer
    from direct.gui.OnscreenText import OnscreenText
    from panda3d.core import Vec4, TextNode

target = None


class DVRP3DApplication:
    """Main 3D application class for DVRP simulation using Panda3D"""
    
    def __init__(
        self,
        map_seed: Optional[int] = None,
        order_seed: Optional[int] = None,
        map_source: str = "real",
        geojson_path: Optional[str] = None,
        building_limit: Optional[int] = None,
    ):
        self.map_seed = map_seed
        self.order_seed = order_seed
        self.map_source = map_source
        self.geojson_path = geojson_path
        self.building_limit = building_limit
        
        # Core components
        self.map: Optional[Map] = None
        self.simulation_engine: Optional[SimulationEngine] = None
        self.visualizer: Optional['Panda3DVisualizer'] = None
        self.order_manager: Optional[OrderManager] = None
        
        # UI elements
        self.stats_text = None
        self.help_text = None
        self.legend_items = []
        
        # Key state tracking
        self._key_states = {}
        
        print("=" * 60)
        print("도심환경에서의 드론 음식 배달 경로 최적화 및 시뮬레이션")
        print("Drone Vehicle Routing Problem (DVRP) 3D Simulation")
        print("=" * 60)
    
    def initialize(self):
        """Initialize the application in the correct order"""
        try:
            # 1. Generate 3D map with buildings
            print("\n1. 3D 지도 생성 중...")
            map_generator = self._create_map_generator()
            self.map = map_generator.generate_map()
            self.map.build_tree()
            
            # 2. Setup depots using clustering
            print("\n2. 클러스터링 및 Depot 배치 중...")
            self._setup_depots()
            
            # 3. Initialize order manager
            print("\n3. 주문 관리 시스템 초기화 중...")
            self.order_manager = OrderManager(self.map, seed=self.order_seed)
            
            # 4. Initialize simulation engine
            print("\n4. 시뮬레이션 엔진 초기화 중...")
            self.simulation_engine = SimulationEngine(self.map, self.order_manager)
            
            if config.RUN_VISUALIZER:
                # 5. Initialize Visualizer (creates Panda3D app)
                # Use actual map dimensions (may differ from config for real maps)
                print("\n5. 3D 시각화 시스템 초기화 중...")
                self.visualizer = Panda3DVisualizer(
                    map_width=self.map.width,
                    map_depth=self.map.depth
                )

                # 6. Create map entities in visualizer
                print("\n6. 3D 맵 엔티티 생성 중...")
                self.visualizer.create_map_entities(self.map)
                
                # 7. Setup UI
                print("\n7. UI 패널 생성 중...")
                self._setup_ui()
                
                # 8. Setup input handling
                self._setup_input()
                
                # 9. Add update task
                self.visualizer.taskMgr.add(self._update_task, 'update_task')
                
                print("\n초기화 완료!")
                print("\n컨트롤:")
                print("  SPACE: 시뮬레이션 시작/일시정지")
                print("  R: 시뮬레이션 리셋")
                print("  +/-: 시뮬레이션 속도 조절")
                print("  H: 도움말 표시/숨기기")
                print("  ESC: 종료")
                print("  마우스 중간/오른쪽 버튼: 카메라 회전")
                print("  스크롤: 줌")
                print("  WASD: 카메라 이동")
                print("  Q/E: 카메라 상하 이동")
            else:
                self.visualizer = None
            
            return True
            
        except Exception as e:
            print(f"초기화 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_depots(self):
        """Setup depots using clustering with automatic k determination"""
        # 1. Create MixedClustering object
        clustering = MixedClustering()
        
        # 2. Prepare 2D coordinates list (x, z) for all stores and customers
        # Include duplicates to reflect floor density
        projected_points = []
        
        for store in self.map.stores:
            pos = store.get_center()
            projected_points.append((pos.x, pos.z))
        
        for customer in self.map.customers:
            pos = customer.get_center()
            projected_points.append((pos.x, pos.z))
        
        print(f"  - Total data points for clustering: {len(projected_points)}")
        
        # 3. Find optimal k automatically using kneed
        optimal_k = clustering.find_optimal_k_auto(projected_points, max_k=10)
        
        # 4. Get optimal depot positions using the determined k value
        depot_candidate_positions = clustering.get_optimal_depot_positions(projected_points, optimal_k)
        
        # 5. Use DepotPlacer to validate and adjust positions
        depot_placer = DepotPlacer(self.map)
        
        # Validate positions and find nearest alternatives if needed
        validated_positions = []
        for i, candidate_pos in enumerate(depot_candidate_positions):
            print(f"  Processing depot {i+1}/{len(depot_candidate_positions)}...")
            
            # DepotPlacer will automatically find nearest valid position
            # if candidate_pos overlaps with buildings
            valid_pos = depot_placer._find_valid_ground_depot_position(
                preferred_pos=candidate_pos,
                width=config.DEPOT_SIZE,
                depth=config.DEPOT_SIZE,
                existing_depots=validated_positions
            )
            
            if valid_pos:
                validated_positions.append(valid_pos)
                if valid_pos.x == candidate_pos.x and valid_pos.z == candidate_pos.z:
                    print(f"  ✓ Depot {i} placed at optimal position: ({valid_pos.x:.1f}, {valid_pos.z:.1f})")
            else:
                print(f"  ✗ Could not place depot {i} (no valid space found)")
        
        # 6. Create depots with drones at validated positions
        depots = depot_placer.create_depots_with_drones(validated_positions)
        
        print(f"\n  - Automatically determined optimal k: {optimal_k}")
        print(f"  - Successfully placed Depot 수: {len(depots)}")
        print(f"  - 총 드론 수: {sum(len(depot.drones) for depot in depots)}")

    def _create_map_generator(self) -> Union[MapGenerator, RealMapGenerator]:
        """Instantiate either the real or synthetic map generator."""
        common_kwargs = dict(
            map_width=config.MAP_WIDTH,
            map_depth=config.MAP_DEPTH,
            max_height=config.MAX_MAP_HEIGHT,
            seed=self.map_seed,
        )

        if self.map_source == "real":
            print("  - Using GeoJSON-based RealMapGenerator")
            return RealMapGenerator(
                **common_kwargs,
                geojson_path=self.geojson_path,
                building_limit=self.building_limit,
            )

        print("  - Using synthetic test MapGenerator")
        return MapGenerator(**common_kwargs)
    
    def _setup_ui(self):
        """Setup UI panels and text using Panda3D"""
        if not self.visualizer:
            return
            
        # Stats panel (top-left)
        self.stats_text = OnscreenText(
            text='',
            pos=(-1.3, 0.9),
            scale=0.05,
            fg=(1, 1, 1, 1),
            bg=(0, 0, 0, 0.5),
            align=TextNode.ALeft,
            mayChange=True
        )
        
        # Legend (bottom-left)
        legend_items = [
            ('Store Buildings', (0, 1, 0, 1)),
            ('Customer Buildings', (1, 0, 0, 1)),
            ('Empty Buildings', (1, 1, 1, 1)),
            ('Depots', (0, 0, 1, 1)),
            ('Drones', (1, 1, 0, 1)),
        ]
        
        self.legend_items = []
        base_y = -0.7
        for i, (label, item_color) in enumerate(legend_items):
            legend_item = OnscreenText(
                text=label,
                pos=(-1.3, base_y - i * 0.06),
                scale=0.04,
                fg=item_color,
                align=TextNode.ALeft
            )
            self.legend_items.append(legend_item)
        
        # Help text (hidden by default)
        help_content = """Controls:
SPACE: Start/Pause
R: Reset
+/-: Speed
H: Toggle Help
WASD: Move Camera
Q/E: Up/Down
Mouse: Rotate
Scroll: Zoom
ESC: Quit"""
        
        self.help_text = OnscreenText(
            text=help_content,
            pos=(1.0, 0.9),
            scale=0.04,
            fg=(1, 1, 1, 1),
            bg=(0, 0, 0, 0.7),
            align=TextNode.ARight
        )
        self.help_text.hide()
    
    def _setup_input(self):
        """Setup additional input handling for simulation control"""
        if not self.visualizer:
            return
            
        # Simulation control keys
        self.visualizer.accept('space', self._toggle_simulation)
        self.visualizer.accept('r', self._reset_simulation)
        self.visualizer.accept('+', self._increase_speed)
        self.visualizer.accept('=', self._increase_speed)
        self.visualizer.accept('-', self._decrease_speed)
        self.visualizer.accept('_', self._decrease_speed)
        self.visualizer.accept('h', self._toggle_help)
        
    def _toggle_simulation(self):
        """Toggle simulation start/pause"""
        if not self.simulation_engine:
            return
            
        if self.simulation_engine.is_running and not self.simulation_engine.paused:
            self.simulation_engine.pause_simulation()
            print("Simulation paused")
        elif self.simulation_engine.paused:
            self.simulation_engine.resume_simulation()
            print("Simulation resumed")
        elif not self.simulation_engine.is_running:
            self.simulation_engine.start_simulation()
            print("Simulation started")
            
    def _reset_simulation(self):
        """Reset simulation"""
        if not self.simulation_engine:
            return
            
        self.simulation_engine.reset_simulation()
        if self.visualizer:
            self.visualizer.clear_drones()
        print("Simulation reset")
        
    def _increase_speed(self):
        """Increase simulation speed"""
        if not self.simulation_engine:
            return
        new_speed = min(10.0, self.simulation_engine.speed_multiplier + 0.5)
        self.simulation_engine.set_speed(new_speed)
        
    def _decrease_speed(self):
        """Decrease simulation speed"""
        if not self.simulation_engine:
            return
        new_speed = max(0.1, self.simulation_engine.speed_multiplier - 0.5)
        self.simulation_engine.set_speed(new_speed)
        
    def _toggle_help(self):
        """Toggle help text visibility"""
        if self.help_text:
            if self.help_text.isHidden():
                self.help_text.show()
            else:
                self.help_text.hide()
    
    def _update_ui(self):
        """Update UI text with current simulation state"""
        if not self.simulation_engine or not self.stats_text:
            return
        
        state = self.simulation_engine.get_simulation_state()
        stats = state['stats']
        fixed_cost = (stats['depot_cost'] + stats['drone_cost']) * config.FIXED_COST_WEIGHT
        variable_cost = stats['charging_cost'] + stats['penalty_cost']
        total_cost = fixed_cost + variable_cost
        
        status = "RUNNING" if state['is_running'] else "PAUSED"
        
        ui_text = f"""DVRP 3D Simulation
Status: {status}
Speed: {state['speed_multiplier']:.1f}x
Time: {state['simulation_time']:.1f}s

Orders:
  Active: {state['active_orders']}
  Completed: {state['completed_orders']}
  
Stats:
  Deliveries: {stats['total_deliveries_completed']}
  Fails: {stats.get('failed_orders', 0)}
  Avg Time: {stats['average_delivery_time']:.1f}s
  Distance: {stats['total_drone_distance']:.1f}m

Costs (Won):
  Fixed: {fixed_cost:,.0f}
  Charging: {stats['charging_cost']:,.0f}
  Penalty: {stats['penalty_cost']:,.0f}
  Total: {total_cost:,.0f}
"""
        
        self.stats_text.setText(ui_text)
    
    def _update_task(self, task):
        """Panda3D task for updating simulation visuals"""
        if self.simulation_engine and self.visualizer:
            # Update drone visuals
            active_drones = self.simulation_engine.get_active_drones()
            self.visualizer.update_drone_visuals(active_drones)
            
            # Update failure markers
            failure_events = self.simulation_engine.get_failure_events()
            self.visualizer.update_failure_markers(failure_events)
            
            # Update UI
            self._update_ui()
            
        return task.cont
    
    def update_simulation_with_visualizer(self):
        """Background thread for simulation updates"""
        start_t = time.time()
        count = 1
        while True:
            time.sleep(max(count * config.SIMULATION_DELTA_TIME - (time.time() - start_t), 0))
            count += 1

            self.simulation_engine.update_step(config.SIMULATION_DELTA_TIME)

    def update_simulation_without_visualizer(self):
        """Run simulation without visualizer"""
        if self.simulation_engine:
            t = 0
            while config.RUN_VISUALIZER or t < config.SIMULATION_TIME:
                self.simulation_engine.update_step(config.SIMULATION_DELTA_TIME)
                t += config.SIMULATION_DELTA_TIME
    
    def run(self):
        """Run the 3D simulation with Panda3D"""
        print("\n3D 시뮬레이션 실행 중...")
        
        # Start simulation automatically
        self.simulation_engine.start_simulation()
        
        if config.RUN_VISUALIZER:
            # Start simulation update thread
            thread = threading.Thread(target=self.update_simulation_with_visualizer)
            thread.daemon = True
            thread.start()
            
            # Run Panda3D main loop
            self.visualizer.run()
        else:
            t = time.perf_counter()
            global target
            target = self.update_simulation_without_visualizer
            cProfile.run('target()', sort='tottime')
            print(time.perf_counter() - t)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.simulation_engine:
            self.simulation_engine.stop_simulation()
            self._print_final_statistics()
        
        if self.visualizer:
            self.visualizer.cleanup()
    
    def _print_final_statistics(self):
        """Print final simulation statistics"""
        stats = self.simulation_engine.get_simulation_state()['stats']
        
        print("\n" + "=" * 60)
        print("최종 시뮬레이션 결과")
        print("=" * 60)
        print(f"총 시뮬레이션 시간: {stats['simulation_duration']:.1f}초")
        print(f"총 처리된 주문: {stats['total_orders_processed']}")
        print(f"완료된 배달: {stats['total_deliveries_completed']}")
        print(f"평균 배달 시간: {stats['average_delivery_time']:.1f}초")
        print(f"총 드론 이동 거리: {stats['total_drone_distance']:.1f}m")
        print(f"실패한 주문: {stats.get('failed_orders', 0)}")
        
        if stats['total_orders_processed'] > 0:
            success_rate = (stats['total_deliveries_completed'] / stats['total_orders_processed']) * 100
            print(f"배달 성공률: {success_rate:.1f}%\n")
        
        print("고정비")
        print(f"depot 비용: {stats['depot_cost']: ,.2f}원")
        print(f"드론 이용: {stats['drone_cost']: ,.2f}원\n")

        print("변동비")
        print(f"충전 비용: {stats['charging_cost']: ,.2f}원")
        print(f"패널티 비용: {stats['penalty_cost']: ,.2f}원\n")

        print(f"총 비용: {(stats['depot_cost'] + stats['drone_cost']) * config.FIXED_COST_WEIGHT + stats['charging_cost'] + stats['penalty_cost']: ,.2f}원")
        
        print("=" * 60)


def _parse_args():
    parser = argparse.ArgumentParser(description="DVRP 3D simulator")
    parser.add_argument(
        "--map-source",
        choices=["real", "random"],
        default=None,
        help="Override config.MAP_SOURCE to select GeoJSON ('real') or synthetic ('random').",
    )
    parser.add_argument(
        "--geojson-path",
        type=str,
        default=None,
        help="Override config.MAP_GEOJSON_PATH with a custom GeoJSON file.",
    )
    parser.add_argument(
        "--building-limit",
        type=int,
        default=None,
        help="Override config.MAP_BUILDING_LIMIT to cap buildings for quick tests.",
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = _parse_args()

    MAP_SEED: Optional[int] = config.MAP_SEED
    ORDER_SEED: Optional[int] = config.ORDER_SEED
    
    map_source = args.map_source or config.MAP_SOURCE
    geojson_path = args.geojson_path if args.geojson_path is not None else config.MAP_GEOJSON_PATH
    building_limit = (
        args.building_limit if args.building_limit is not None else config.MAP_BUILDING_LIMIT
    )

    app_instance = DVRP3DApplication(
        map_seed=MAP_SEED,
        order_seed=ORDER_SEED,
        map_source=map_source,
        geojson_path=geojson_path,
        building_limit=building_limit,
    )
    
    try:
        if app_instance.initialize():
            app_instance.run()
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"애플리케이션 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if app_instance:
            app_instance.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
