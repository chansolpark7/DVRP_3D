# 필요한 라이브러리들을 불러옵니다.
from pathlib import Path
from typing import Any, Dict, Optional
import sys

import geopandas as gpd  # 지리공간 데이터를 다루기 위한 라이브러리 (Shapefile 등)
import matplotlib.cm as cm  # 컬러맵 사용을 위한 라이브러리
import matplotlib.colors as colors  # 색상 정규화를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화 라이브러리
import numpy as np  # 수치 계산을 위한 라이브러리
import pandas as pd  # 데이터 분석 및 조작을 위한 라이브러리 (GeoPandas의 기반)
from mpl_toolkits.mplot3d import Axes3D  # 3D 시각화를 위한 도구
from scipy.interpolate import griddata  # 공간 보간법을 위한 라이브러리

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

MIN_BUILDING_AREA_SQM = 50.0  # 건물 최소 면적 필터링 기준 (제곱미터) 1평=3.3제곱미터
DEFAULT_FLOOR_HEIGHT = getattr(config, "FLOOR_HEIGHT", 3.0)


def _ensure_path(path_like: Any) -> Path:
    """Convert 다양한 경로 표현을 pathlib.Path 객체로 통일합니다."""
    path = path_like if isinstance(path_like, Path) else Path(path_like)
    if not path.is_absolute():
        return config.PROJECT_ROOT / path
    return path


def _standardize_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Column 이름을 대문자로 통일해 KeyError를 방지합니다 (geometry 컬럼은 유지)."""
    if gdf.empty:
        return gdf
    geom_col = gdf.geometry.name if gdf._geometry_column_name in gdf.columns else "geometry"
    rename_map = {col: col.upper() for col in gdf.columns if col != geom_col}
    return gdf.rename(columns=rename_map)


def _sanitize_geometries(gdf: gpd.GeoDataFrame, min_area: float) -> gpd.GeoDataFrame:
    """Invalid/degenerate polygon 제거 및 buffer(0)로 기하 정리."""
    if gdf.empty:
        return gdf

    gdf = gdf.copy()
    cleaned = gdf.geometry.buffer(0)

    invalid_mask = ~cleaned.is_valid
    if invalid_mask.any():
        print(f"    - ⚠️  buffer(0) 후에도 유효하지 않은 기하 {invalid_mask.sum()}개를 제외합니다.")
        cleaned[invalid_mask] = None

    gdf.geometry = cleaned
    gdf = gdf.dropna(subset=["geometry"]).copy()
    if gdf.empty:
        return gdf

    small_mask = gdf.geometry.area < max(min_area, 0)
    removed = int(small_mask.sum())
    if removed:
        print(f"    - ⚠️  면적 {min_area}㎡ 미만 건물 {removed}개를 제외합니다.")
        gdf = gdf[~small_mask].copy()

    gdf.reset_index(drop=True, inplace=True)
    return gdf


def _resolve_heights(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """HEIGHT가 없는 경우 다른 속성을 활용하고, 극단값을 통계적으로 클램프합니다."""
    if buildings.empty:
        return buildings

    result = buildings.copy()
    heights = pd.to_numeric(result['HEIGHT'], errors='coerce').fillna(0.0)
    cont = pd.to_numeric(result['CONT'], errors='coerce').fillna(0.0)
    floors_raw = result['GRND_FLR'] if 'GRND_FLR' in result.columns else pd.Series(0, index=result.index, dtype=float)
    floors = pd.to_numeric(floors_raw, errors='coerce').fillna(0.0)
    abs_raw = result['ABSOLUTE_HEIGHT_SOURCE'] if 'ABSOLUTE_HEIGHT_SOURCE' in result.columns else pd.Series(0, index=result.index, dtype=float)
    abs_source = pd.to_numeric(abs_raw, errors='coerce').fillna(0.0)

    height_from_abs = abs_source - cont
    height_from_abs = height_from_abs.where(height_from_abs > 0)
    heights = heights.where(heights > 0, height_from_abs)

    derived_from_floor = floors * DEFAULT_FLOOR_HEIGHT
    derived_from_floor = derived_from_floor.where(derived_from_floor > 0)
    heights = heights.where(heights > 0, derived_from_floor)

    positive = heights[heights > 0]
    if positive.empty:
        heights = pd.Series(DEFAULT_FLOOR_HEIGHT, index=heights.index, dtype=float)
    else:
        heights = heights.where(heights > 0, positive.median())

    result['HEIGHT'] = heights
    return result

def load_data(config):
    """모든 Shapefile을 불러오고, 데이터를 병합하며, 건물 데이터의 컬럼 이름을 변경합니다."""
    print("✅ 1. 데이터 로딩을 시작합니다...")

    # 내부 헬퍼 함수: 지정된 경로의 Shapefile들을 읽어 하나의 GeoDataFrame으로 합칩니다.
    def _load_and_concat_shp(paths, encoding):
        gdf_list = []  # 개별 GeoDataFrame을 담을 리스트
        if not paths or not paths[0]: return gpd.GeoDataFrame() # 경로가 비어있으면 빈 프레임 반환
        for path in paths:
            try:
                path = _ensure_path(path)
                # GeoPandas를 이용해 Shapefile을 읽습니다. 한글 깨짐 방지를 위해 인코딩 지정.
                gdf_list.append(gpd.read_file(path, encoding=encoding))
            except Exception as e:
                print(f"    - 🚨 오류: '{path}' 파일 로딩 실패: {e}")
        # 리스트에 담긴 모든 GeoDataFrame을 하나로 합칩니다.
        return pd.concat(gdf_list, ignore_index=True) if gdf_list else gpd.GeoDataFrame()

    # 설정값(config)에 따라 각 데이터 로딩
    terrain_gdf = _standardize_columns(_load_and_concat_shp(config["terrain_contour_paths"], 'UTF-8')) # 등고선 데이터
    building_gdf = _standardize_columns(_load_and_concat_shp(config["building_paths"], 'EUC-KR'))   # 건물 데이터 (주로 EUC-KR 인코딩)
    spot_gdf = _standardize_columns(_load_and_concat_shp(config["spot_elevation_paths"], 'UTF-8'))   # 표고점 데이터

    
    # 로딩된 데이터의 개수를 출력합니다.
    print(f"    - 총 {len(building_gdf)}개의 건물, {len(terrain_gdf)}개의 등고선, {len(spot_gdf)}개의 표고점 데이터를 로딩 및 병합했습니다.")
    
    # 필수 데이터(지형, 건물)가 없으면 처리를 중단합니다.
    if terrain_gdf.empty or building_gdf.empty:
        print("🚨 [중요] 등고선 또는 건물 데이터가 비어있어 처리를 중단합니다.")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
    # 로딩된 데이터들을 반환합니다.
    return terrain_gdf, building_gdf, spot_gdf

def preprocess_data(terrain_gdf, building_gdf, spot_elevation_gdf):
    """모든 전처리 과정(좌표계 통일, 필터링, 고도 계산 등)을 수행합니다."""
    print("✅ 2. 데이터 전처리를 시작합니다...")

    if 'CONT' not in terrain_gdf.columns:
        raise KeyError(f"등고선 데이터에 'CONT' 컬럼이 없습니다. 현재 컬럼: {list(terrain_gdf.columns)}")
    if not spot_elevation_gdf.empty and 'NUME' not in spot_elevation_gdf.columns:
        print(f"    - ⚠️  표고점 데이터에 'NUME' 컬럼이 없어 표고점 기반 보간을 건너뜁니다. 현재 컬럼: {list(spot_elevation_gdf.columns)}")
        spot_elevation_gdf = gpd.GeoDataFrame(geometry=spot_elevation_gdf.geometry, crs=spot_elevation_gdf.crs)
    
    # 1. 좌표계 통일(CRS Unification): 모든 데이터를 하나의 좌표계로 맞춰야 위치를 정확히 비교할 수 있습니다.
    # 기준 좌표계는 등고선 데이터의 좌표계로 설정합니다.
    target_crs = terrain_gdf.crs
    if building_gdf.crs != target_crs:
        building_gdf = building_gdf.to_crs(target_crs)
    if not spot_elevation_gdf.empty and spot_elevation_gdf.crs != target_crs:
        spot_elevation_gdf = spot_elevation_gdf.to_crs(target_crs)

    # 2. 공간 필터링(Spatial Filtering): 등고선 데이터가 포함하는 전체 영역을 계산합니다.
    minx, miny, maxx, maxy = terrain_gdf.total_bounds
    # 이 영역 내에 있는 건물들만 필터링하여 처리 효율을 높입니다.
    filtered_building_gdf = building_gdf.cx[minx:maxx, miny:maxy].copy()
    print(f"    - 지형 범위 내 필터링 후 건물 수: {len(filtered_building_gdf)}")
    
    # 지형과 표고점 데이터도 동일한 범위로 필터링합니다.
    filtered_terrain_gdf = terrain_gdf.cx[minx:maxx, miny:maxy].copy()
    filtered_spot_elevation_gdf = spot_elevation_gdf.cx[minx:maxx, miny:maxy].copy() if not spot_elevation_gdf.empty else gpd.GeoDataFrame()
    
    # 나중에 데이터를 합칠 때 기준이 될 고유 ID를 각 건물에 부여합니다.
    filtered_building_gdf = filtered_building_gdf.reset_index(drop=True)
    filtered_building_gdf['unique_id'] = filtered_building_gdf.index


    # 3. 건물 높이 계산: 3D 시각화를 위해 각 건물의 높이를 결정합니다.
    height_col, floor_col, abs_col = 'HEIGHT', 'GRND_FLR', 'ABSOLUTE_HEIGHT'
    print(f"    - 건물 높이를 계산합니다: '{height_col}' 값 사용, 0일 경우 '{abs_col}' 또는 '{floor_col}' * {DEFAULT_FLOOR_HEIGHT}m로 추정")

    # 'HEIGHT' 컬럼을 숫자형으로 변환합니다. 존재하지 않거나 변환 실패 시 0으로 채웁니다.
    if height_col in filtered_building_gdf.columns:
        filtered_building_gdf[height_col] = pd.to_numeric(filtered_building_gdf[height_col], errors='coerce').fillna(0)
    else:
        print(f"    - 🚨 경고: '{height_col}' 컬럼이 없어 높이를 0으로 간주하고 시작합니다.")
        filtered_building_gdf[height_col] = 0

    # 'GRND_FLR' 컬럼을 숫자형으로 변환합니다. 존재하지 않거나 변환 실패 시 0으로 채웁니다.
    if floor_col in filtered_building_gdf.columns:
        filtered_building_gdf[floor_col] = pd.to_numeric(filtered_building_gdf[floor_col], errors='coerce').fillna(0)
    else:
        print(f"    - 🚨 경고: '{floor_col}' 컬럼이 없어 높이 추정이 불가능합니다.")
        filtered_building_gdf[floor_col] = 0

    # GRND_FLR가 0층인 건물은 제외합니다.
    floor_zero_mask = filtered_building_gdf[floor_col] == 0
    removed_zero = int(floor_zero_mask.sum())
    if removed_zero > 0:
        filtered_building_gdf = filtered_building_gdf[~floor_zero_mask].copy()
        filtered_building_gdf.reset_index(drop=True, inplace=True)
        filtered_building_gdf['unique_id'] = filtered_building_gdf.index
        print(f"    - GRND_FLR가 0층인 건물 {removed_zero}개를 제외했습니다.")

    if abs_col in filtered_building_gdf.columns:
        filtered_building_gdf[abs_col] = pd.to_numeric(filtered_building_gdf[abs_col], errors='coerce').fillna(0)
    else:
        filtered_building_gdf[abs_col] = 0
        
    # 'HEIGHT'가 0인 건물의 인덱스를 찾습니다.
    indices_to_estimate = filtered_building_gdf[height_col] == 0
    
    # 해당 인덱스의 건물들에 대해 높이를 재계산합니다: 지상층수 * 3
    # .loc[indices, column]을 사용하여 특정 행과 열을 선택해 값을 변경합니다.
    estimated_heights = filtered_building_gdf.loc[indices_to_estimate, floor_col] * DEFAULT_FLOOR_HEIGHT
    filtered_building_gdf.loc[indices_to_estimate, height_col] = estimated_heights

    # 최종 결과 보고
    num_estimated = indices_to_estimate.sum()
    if num_estimated > 0:
        print(f"    - '{height_col}'가 0이었던 {num_estimated}개 건물에 대해 '{floor_col}'를 이용해 높이를 추정했습니다.")
    
    # 4. 건물 바닥 고도 계산 (1단계 - Spatial Join)
    print("    - 1단계: Spatial Join으로 고도를 할당합니다...")
    # sjoin: 공간 정보를 기준으로 두 데이터를 합칩니다. 'intersects'는 '서로 만나는' 경우를 의미합니다.
    # 각 건물이 어떤 등고선(CONT)과 만나는지 찾아 고도를 할당합니다.
    buildings_with_terrain = gpd.sjoin(filtered_building_gdf, filtered_terrain_gdf[['CONT', 'geometry']], how="left", predicate='intersects')
    
    # 한 건물이 여러 등고선과 만날 수 있으므로, 고유 ID로 그룹화하여 평균 고도를 계산합니다.
    # 동시에 다른 중요 정보(건물명, 높이 등)는 그대로 유지합니다.
    agg_dict = {
        'CONT': 'mean',
        'HEIGHT': 'first',
        'geometry': 'first',
        'GRND_FLR': 'first',
        'UFID': 'first',
        'ABSOLUTE_HEIGHT': 'first'
    }
    agg_df = buildings_with_terrain.groupby('unique_id').agg(agg_dict).reset_index()
    processed_buildings = gpd.GeoDataFrame(agg_df, geometry='geometry', crs=target_crs)
    processed_buildings = processed_buildings.rename(columns={'ABSOLUTE_HEIGHT': 'ABSOLUTE_HEIGHT_SOURCE'})
    print(f"    - Spatial Join 후 건물 수: {len(processed_buildings)}")

    # 5. 건물 바닥 고도 계산 (2단계 - 공간 보간법)
    # sjoin으로 고도를 찾지 못한 건물들(등고선 사이에 위치)을 대상으로 보간법을 수행합니다.
    buildings_to_interpolate = processed_buildings[processed_buildings['CONT'].isna()].copy()
    if not buildings_to_interpolate.empty:
        print(f"    - 2단계: {len(buildings_to_interpolate)}개 건물에 대해 공간 보간법을 수행합니다...")
        # 주변의 알려진 고도 지점을 모두 수집합니다. (등고선 + 표고점)
        # 1. 등고선 위의 모든 점들의 좌표(x, y)와 고도값(CONT)을 추출합니다.
        contour_points_gdf = filtered_terrain_gdf[pd.notna(filtered_terrain_gdf['CONT'])].explode(index_parts=False).get_coordinates()
        known_points_list = [contour_points_gdf[['x', 'y']].values]
        known_values_list = [filtered_terrain_gdf.loc[contour_points_gdf.index, 'CONT'].values]
        
        # 2. 표고점 데이터가 있으면, 표고점의 좌표(x, y)와 고도값(NUME)도 추가합니다.
        if not filtered_spot_elevation_gdf.empty and 'NUME' in filtered_spot_elevation_gdf.columns:
            spot_points_gdf = filtered_spot_elevation_gdf[pd.notna(filtered_spot_elevation_gdf['NUME'])].get_coordinates()
            if not spot_points_gdf.empty:
                known_points_list.append(spot_points_gdf[['x', 'y']].values)
                known_values_list.append(filtered_spot_elevation_gdf.loc[spot_points_gdf.index, 'NUME'].values)
        
        # 수집된 점들을 하나의 배열로 합칩니다.
        known_points = np.vstack(known_points_list)
        known_values = np.concatenate(known_values_list)
        
        if len(known_points) > 0:
            # 고도를 추정할 건물들의 중심점 좌표를 가져옵니다.
            centroids = buildings_to_interpolate.geometry.centroid
            target_points = np.vstack((centroids.x, centroids.y)).T
            # griddata: 주변의 알려진 점들(known_points, known_values)을 이용해,
            # 목표 지점(target_points)의 값을 선형으로 추정(method='linear')합니다.
            interpolated_values = griddata(known_points, known_values, target_points, method='linear', fill_value=0)
            # 추정된 고도값을 원래 데이터프레임에 채워넣습니다.
            processed_buildings.loc[buildings_to_interpolate.index, 'CONT'] = interpolated_values

    processed_buildings = _resolve_heights(processed_buildings)

    # 5-bis. 높이=0 또는 층수=0 건물 제거 (안전장치)
    floors_now = pd.to_numeric(processed_buildings.get('GRND_FLR', 0), errors='coerce').fillna(0)
    heights_now = pd.to_numeric(processed_buildings.get('HEIGHT', 0), errors='coerce').fillna(0)
    keep_mask = (floors_now > 0) & (heights_now > 0)
    removed_zero_dim = int((~keep_mask).sum())
    if removed_zero_dim > 0:
        processed_buildings = processed_buildings[keep_mask].copy()
        processed_buildings.reset_index(drop=True, inplace=True)
        print(f"    - 높이=0 또는 GRND_FLR=0 건물 {removed_zero_dim}개를 추가로 제외했습니다.")

    # 6. 건물의 절대 높이 계산: 건물의 최종 높이 = 바닥의 해발고도(CONT) + 건물 자체 높이(HEIGHT)
    processed_buildings['ABSOLUTE_HEIGHT'] = processed_buildings['CONT'].fillna(0) + processed_buildings['HEIGHT']

    # 7. 3D 렌더링을 위한 필터: 높이와 면적이 모두 양수인 건물만 유지합니다.
    valid_height = processed_buildings['HEIGHT'] > 0
    valid_area = processed_buildings.geometry.area > 0
    before_filter = len(processed_buildings)
    processed_buildings = processed_buildings[valid_height & valid_area].copy()
    removed = before_filter - len(processed_buildings)
    if removed > 0:
        print(f"    - 품질 필터로 {removed}개 건물을 제외했습니다. (높이/면적 조건 불충족)")

    processed_buildings = _sanitize_geometries(processed_buildings, MIN_BUILDING_AREA_SQM)
    if 'ABSOLUTE_HEIGHT_SOURCE' in processed_buildings.columns:
        processed_buildings = processed_buildings.drop(columns=['ABSOLUTE_HEIGHT_SOURCE'])

    print("✅ 데이터 전처리 및 계산 완료.")
    return filtered_terrain_gdf, processed_buildings

def export_building_list_with_coords(buildings_gdf, config):
    """건물의 경위도 좌표 및 주요 정보를 추출하여 CSV 파일로 저장합니다."""
    if buildings_gdf.empty: return # 처리할 건물이 없으면 함수 종료
    print("\n✅ 3. 시각화된 건물 목록과 좌표를 파일로 저장합니다...")
    
    buildings_to_export = buildings_gdf.copy()
    # 1. 각 건물의 중심점 좌표를 계산합니다. (현재는 미터 기반 좌표계)
    centroids_projected = buildings_to_export.geometry.centroid
    # 2. 중심점 좌표를 전 세계 표준인 위도/경도(WGS84, EPSG:4326)로 변환합니다.
    centroids_wgs84 = centroids_projected.to_crs("EPSG:4326")
    # 3. 변환된 위도(latitude)와 경도(longitude)를 새 컬럼으로 추가합니다.
    buildings_to_export['longitude'] = centroids_wgs84.x
    buildings_to_export['latitude'] = centroids_wgs84.y
    
    # CSV 파일로 저장할 경로를 설정에서 가져옵니다.
    output_path = _ensure_path(config["output_csv_filename"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV에 저장할 컬럼 목록을 지정합니다.
    columns_to_save = ['UFID', 'GRND_FLR', 'HEIGHT', 'CONT', 'ABSOLUTE_HEIGHT', 'latitude', 'longitude']
    # 데이터에 존재하는 컬럼만 최종적으로 선택합니다.
    final_columns = [col for col in columns_to_save if col in buildings_to_export.columns]
    
    # 지정된 컬럼만 CSV 파일로 저장합니다. index=False는 불필요한 인덱스 저장을 방지합니다.
    # encoding='utf-8-sig'는 Excel에서 한글이 깨지지 않도록 보장합니다.
    buildings_to_export[final_columns].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"    - 건물 목록을 '{output_path}' 파일로 저장했습니다.")

def export_geojson(buildings_gdf, config):
    """최종 처리된 건물 데이터를 GeoJSON으로 저장합니다."""
    if buildings_gdf.empty:
        print("🚨 내보낼 건물 데이터가 없습니다.")
        return
    
    print("\n✅ 4. GeoJSON 저장합니다...")
    
    # GeoDataFrame을 GeoJSON 파일로 저장
    # 이 파일에는 각 건물의 'geometry'(모양), 'CONT'(바닥고도), 'HEIGHT'(건물높이)가 모두 포함됩니다.
    output_path = _ensure_path(config["output_geojson_filename"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        buildings_gdf.to_file(output_path, driver="GeoJSON")
        print(f"    - 성공: {len(buildings_gdf)}개 건물 데이터를 '{output_path}'로 저장했습니다.")
    except Exception as e:
        print(f"    - 🚨 오류: GeoJSON 저장 실패: {e}")

def visualize_2d(terrain, buildings, config):
    """2D 시각화 결과물을 생성하고 저장합니다."""
    print("\n✅ 5. 2D 시각화를 생성합니다...")

    # 1. 2D 지도 축을 위도/경도로 표시하기 위해 데이터를 WGS84 좌표계로 변환합니다.
    print("    - 2D 지도용 좌표를 위도/경도로 변환합니다...")
    terrain_wgs84 = terrain.to_crs("EPSG:4326")
    buildings_wgs84 = buildings.to_crs("EPSG:4326")

    # 2. 시각화를 위한 그림판(fig)과 좌표축(ax)을 생성합니다.
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    # 3. 배경으로 지형(등고선)을 연한 회색으로 그립니다.
    terrain_wgs84.plot(ax=ax, color='gainsboro', linewidth=0.5)
    # 4. 건물들을 그립니다. 이때 'ABSOLUTE_HEIGHT' 값에 따라 색상을 다르게 표현합니다.
    buildings_wgs84.plot(column='ABSOLUTE_HEIGHT', # 색상 기준이 될 컬럼
                         cmap='plasma',           # 사용할 컬러맵 (낮으면 보라, 높으면 노랑)
                         ax=ax,
                         legend=True,             # 색상 범례 표시
                         legend_kwds={'label': "Absolute Height (m)", 'orientation': "vertical", 'shrink': 0.5, 'aspect': 30})
    
    # 5. 그래프의 x, y축 레이블과 제목을 설정합니다.
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title('2D Map (Latitude/Longitude Axes)')
    
    # 6. 완성된 그래프를 이미지 파일로 저장합니다.
    output_path = _ensure_path(config["output_2d_filename"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config["dpi_2d"], bbox_inches='tight')
    plt.close(fig)
    print(f"    - 2D 지도를 '{output_path}' 파일로 저장했습니다.")
   
def visualize_3d(terrain, buildings, config):
    """3D 시각화 결과물을 생성하고 저장합니다."""
    print("\n✅ 6. 3D 시각화를 생성합니다...")
    fig = plt.figure(figsize=(18, 15))
    # 3D 그래프를 그릴 수 있는 축(ax)을 생성합니다.
    ax = fig.add_subplot(111, projection='3d')

    # 1. 지형 표면 시각화
    if not terrain.empty and 'CONT' in terrain.columns:
        # 등고선 데이터에서 좌표(x, y)와 고도(z)를 추출합니다.
        points = terrain[pd.notna(terrain['CONT'])].explode(index_parts=False).get_coordinates()
        if not points.empty:
            points['Z'] = terrain.loc[points.index, 'CONT']
            # plot_trisurf: 점들을 삼각형으로 연결하여 3D 표면을 만듭니다.
            ax.plot_trisurf(points['x'], points['y'], points['Z'], cmap='Greens', alpha=0.5, zorder=1)

    # 2. 건물 시각화
    if not buildings.empty:
        # 건물의 절대 높이에 따라 색상을 매핑하기 위한 준비
        min_h, max_h = buildings['ABSOLUTE_HEIGHT'].min(), buildings['ABSOLUTE_HEIGHT'].max()
        cmap, norm = plt.get_cmap('plasma'), colors.Normalize(vmin=min_h, vmax=max_h if max_h > min_h else min_h + 1)
        
        # 각 건물을 하나씩 순회하며 3D로 그립니다.
        for _, row in buildings.iterrows():
            if row['HEIGHT'] > 0 and row.geometry and hasattr(row.geometry, 'exterior'):
                # 높이에 맞는 색상 지정
                color = cmap(norm(row['ABSOLUTE_HEIGHT']))
                # 건물의 바닥 높이(z_bottom)와 꼭대기 높이(z_top)
                z_bottom, z_top = row['CONT'], row['ABSOLUTE_HEIGHT']
                # 건물의 2D 외곽선 좌표(x, y)
                x, y = row.geometry.exterior.xy
                # 건물 밑면 그리기 (z_bottom 높이에 외곽선)
                ax.plot(x, y, z_bottom, color=color, linewidth=0.5, zorder=2)
                # 건물 윗면 그리기 (z_top 높이에 외곽선)
                ax.plot(x, y, z_top, color=color, linewidth=1, zorder=3)
                # 건물 벽(기둥) 그리기: 각 꼭짓점에서 밑면과 윗면을 잇는 수직선
                for i in range(len(x)):
                    ax.plot([x[i], x[i]], [y[i], y[i]], [z_bottom, z_top], color=color, linewidth=1, zorder=2)
        
        # 3. 컬러바 추가: 색상이 어떤 높이 값을 의미하는지 보여주는 범례
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, label='Absolute Height (m)')
        
    # 4. 3D 그래프의 제목과 축 레이블 설정
    ax.set_title('3D Map (Meter-based Axes for True Scale)')
    ax.set_xlabel("X Coordinate (meters)")
    ax.set_ylabel("Y Coordinate (meters)")
    ax.set_zlabel("Z Coordinate (meters, Elevation)")
    # 5. 3D 뷰의 시점(카메라 각도) 설정: elev는 높이, azim은 방위각
    ax.view_init(elev=30, azim=-45)
    # 6. 완성된 3D 뷰를 이미지 파일로 저장
    output_path = _ensure_path(config["output_3d_filename"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config["dpi_3d"], bbox_inches='tight')
    plt.close(fig)
    print(f"    - 3D 지도를 '{output_path}' 파일로 저장했습니다.")

def generate_building_assets(config_overrides: Optional[Dict[str, Any]] = None):
    """전체 데이터 처리 및 시각화 파이프라인을 실행하고 산출물을 생성합니다."""
    pipeline_config = config.get_buildings_data_config(config_overrides)
    try:
        terrain_data, building_data, spot_data = load_data(pipeline_config)
        if terrain_data.empty or building_data.empty:
            return None, None

        final_terrain, final_buildings = preprocess_data(
            terrain_data, building_data, spot_data
        )
        export_building_list_with_coords(final_buildings, pipeline_config)
        export_geojson(final_buildings, pipeline_config)
        visualize_2d(final_terrain, final_buildings, pipeline_config)
        visualize_3d(final_terrain, final_buildings, pipeline_config)
        return final_terrain, final_buildings
    except FileNotFoundError as e:
        print(f"🚨 [오류] 파일을 찾을 수 없습니다. 설정된 경로를 확인하세요. 상세 정보: {e}")
        raise
    except Exception as e:
        print(f"🚨 [오류] 예상치 못한 오류가 발생했습니다: {e}")
        raise


def main():
    """직접 실행 시 기본 설정으로 산출물을 생성합니다."""
    generate_building_assets()

# 이 스크립트 파일이 직접 실행될 때만 main() 함수를 호출합니다.
if __name__ == "__main__":
    main()
