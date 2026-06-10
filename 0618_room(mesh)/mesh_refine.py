import glob
import os

import numpy as np
import open3d as o3d


# ============================================================
# 입력 / 출력 경로
# ============================================================
# 이 파일(mesh_refine.py)이 있는 폴더를 기준으로 동작
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLY_DIR = os.path.join(SCRIPT_DIR, "ply")

# object_detecting_final.py에서 생성한 객체 Mesh
INPUT_MESH_PATTERN = os.path.join(
    PLY_DIR,
    "07_object_mesh_*.ply"
)

# 객체 검출 후 남긴 원본 색 Point Cloud
# 가짜 면 판별과 Mesh 색 복원에 사용
SOURCE_POINT_CLOUD_PATH = os.path.join(
    PLY_DIR,
    "04_detected_object_points_original_color.ply"
)

# 출력 예시: 08_refined_object_mesh_000.ply
OUTPUT_MESH_PREFIX = os.path.join(
    PLY_DIR,
    "08_refined_object_mesh_"
)

OUTPUT_MESH_PATTERN = os.path.join(
    PLY_DIR,
    "08_refined_object_mesh_*.ply"
)


# ============================================================
# 가짜 막 제거 설정
# ============================================================
# 각 Mesh 주변에서 원본 Point Cloud를 가져올 여유 범위
SOURCE_CROP_MARGIN = 0.03

# 원본 점에서 이 거리보다 먼 Mesh 꼭짓점 제거
# 더 강하게 제거: 0.020
# 표면이 너무 많이 뚫림: 0.030~0.040
MAX_VERTEX_DISTANCE = 0.05

# 삼각형 중심과 세 변의 중간점이 원본 점에서 멀면 제거
# 사진처럼 빈 공간을 덮는 막 제거에 가장 중요한 값
MAX_TRIANGLE_SAMPLE_DISTANCE = 0.025

# 한 변이 너무 긴 삼각형 제거
MAX_TRIANGLE_EDGE = 0.050

# 면적이 지나치게 큰 삼각형 제거
MAX_TRIANGLE_AREA = 0.0010

# 너무 작은 분리 조각 제거
MIN_COMPONENT_TRIANGLES = 100

# True면 가장 큰 Mesh 덩어리 하나만 남김
# 의자 다리처럼 여러 부분이 끊겨 있을 수 있으므로 기본은 False
KEEP_ONLY_LARGEST_COMPONENT = False


# ============================================================
# Mesh 부드럽게 만들기
# ============================================================
SMOOTH_ITERATIONS = 35
SMOOTH_LAMBDA = 0.5
SMOOTH_MU = -0.53

# 삼각형이 너무 많으면 단순화
# 0이면 단순화하지 않음
TARGET_TRIANGLES = 30000

# smoothing 후 원본 점에서 너무 멀어진 부분을 한 번 더 제거
FINAL_MAX_VERTEX_DISTANCE = 0.030


# ============================================================
# 시각화
# ============================================================
SHOW_RESULT = True


# ------------------------------------------------------------
# 기본 Mesh 정리
# ------------------------------------------------------------
def basic_cleanup(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


# ------------------------------------------------------------
# Mesh 주변 원본 Point Cloud만 추출
# ------------------------------------------------------------
def crop_source_points(source_pcd, mesh):
    bbox = mesh.get_axis_aligned_bounding_box()

    min_bound = bbox.get_min_bound() - SOURCE_CROP_MARGIN
    max_bound = bbox.get_max_bound() + SOURCE_CROP_MARGIN

    crop_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound,
        max_bound
    )

    return source_pcd.crop(crop_box)


# ------------------------------------------------------------
# 원본 Point Cloud와 먼 Mesh 꼭짓점 제거
# ------------------------------------------------------------
def remove_far_vertices(mesh, source_pcd, max_distance):
    if len(mesh.vertices) == 0 or len(source_pcd.points) == 0:
        return mesh, 0

    mesh_vertex_pcd = o3d.geometry.PointCloud()
    mesh_vertex_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertices)
    )

    distances = np.asarray(
        mesh_vertex_pcd.compute_point_cloud_distance(source_pcd)
    )

    remove_mask = distances > max_distance
    removed_count = int(np.sum(remove_mask))

    mesh.remove_vertices_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()

    return mesh, removed_count


# ------------------------------------------------------------
# 긴 삼각형 / 빈 공간을 덮는 삼각형 제거
# ------------------------------------------------------------
def remove_bad_triangles(mesh, source_pcd):
    if len(mesh.triangles) == 0 or len(source_pcd.points) == 0:
        return mesh, 0

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_points = vertices[triangles]

    p0 = triangle_points[:, 0]
    p1 = triangle_points[:, 1]
    p2 = triangle_points[:, 2]

    edge_01 = np.linalg.norm(p0 - p1, axis=1)
    edge_12 = np.linalg.norm(p1 - p2, axis=1)
    edge_20 = np.linalg.norm(p2 - p0, axis=1)

    max_edge = np.maximum.reduce([
        edge_01,
        edge_12,
        edge_20
    ])

    # 삼각형 면적
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(p1 - p0, p2 - p0),
        axis=1
    )

    # 중심점 1개만 검사하면 큰 막을 놓칠 수 있어서
    # 중심 + 각 변의 중간점까지 총 4곳을 검사한다.
    centers = (p0 + p1 + p2) / 3.0
    middle_01 = (p0 + p1) / 2.0
    middle_12 = (p1 + p2) / 2.0
    middle_20 = (p2 + p0) / 2.0

    sample_points = np.vstack([
        centers,
        middle_01,
        middle_12,
        middle_20
    ])

    sample_pcd = o3d.geometry.PointCloud()
    sample_pcd.points = o3d.utility.Vector3dVector(sample_points)

    sample_distances = np.asarray(
        sample_pcd.compute_point_cloud_distance(source_pcd)
    )

    triangle_count = len(triangles)

    center_distance = sample_distances[0:triangle_count]
    middle_01_distance = sample_distances[
        triangle_count:triangle_count * 2
    ]
    middle_12_distance = sample_distances[
        triangle_count * 2:triangle_count * 3
    ]
    middle_20_distance = sample_distances[
        triangle_count * 3:triangle_count * 4
    ]

    max_sample_distance = np.maximum.reduce([
        center_distance,
        middle_01_distance,
        middle_12_distance,
        middle_20_distance
    ])

    remove_mask = (
        (max_edge > MAX_TRIANGLE_EDGE) |
        (triangle_area > MAX_TRIANGLE_AREA) |
        (
            max_sample_distance >
            MAX_TRIANGLE_SAMPLE_DISTANCE
        )
    )

    removed_count = int(np.sum(remove_mask))

    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()

    return mesh, removed_count


# ------------------------------------------------------------
# 작은 분리 조각 제거
# ------------------------------------------------------------
def remove_small_components(mesh):
    if len(mesh.triangles) == 0:
        return mesh, 0

    triangle_clusters, cluster_counts, _ = (
        mesh.cluster_connected_triangles()
    )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_counts = np.asarray(cluster_counts)

    if len(cluster_counts) == 0:
        return mesh, 0

    if KEEP_ONLY_LARGEST_COMPONENT:
        largest_cluster = int(np.argmax(cluster_counts))
        remove_mask = triangle_clusters != largest_cluster
    else:
        remove_mask = np.array([
            cluster_counts[cluster_id] < MIN_COMPONENT_TRIANGLES
            for cluster_id in triangle_clusters
        ])

    removed_count = int(np.sum(remove_mask))

    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()

    return mesh, removed_count


# ------------------------------------------------------------
# 원본 Point Cloud 색을 Mesh에 복사
# ------------------------------------------------------------
def transfer_source_colors(mesh, source_pcd):
    if len(mesh.vertices) == 0:
        return mesh

    if not source_pcd.has_colors():
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
        return mesh

    source_colors = np.asarray(source_pcd.colors)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_colors = np.zeros(
        (len(mesh_vertices), 3),
        dtype=np.float64
    )

    kd_tree = o3d.geometry.KDTreeFlann(source_pcd)

    for index, vertex in enumerate(mesh_vertices):
        found, nearest_indices, _ = (
            kd_tree.search_knn_vector_3d(vertex, 1)
        )

        if found > 0:
            mesh_colors[index] = source_colors[nearest_indices[0]]
        else:
            mesh_colors[index] = [0.7, 0.7, 0.7]

    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    return mesh


# ------------------------------------------------------------
# Mesh 하나 다듬기
# ------------------------------------------------------------
def refine_mesh(mesh, source_pcd, mesh_name):
    print(f"\n========== {mesh_name} ==========")
    print(
        f"[BEFORE] vertices={len(mesh.vertices):,}, "
        f"triangles={len(mesh.triangles):,}"
    )

    mesh = basic_cleanup(mesh)

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("[SKIP] 기본 정리 후 Mesh가 비어 있음")
        return None

    local_source = crop_source_points(source_pcd, mesh)

    if len(local_source.points) == 0:
        print("[SKIP] Mesh 주변에 원본 Point Cloud가 없음")
        return None

    print(f"[INFO] nearby source points: {len(local_source.points):,}")

    # 1. 원본 점과 먼 꼭짓점 제거
    mesh, removed_vertices = remove_far_vertices(
        mesh,
        local_source,
        MAX_VERTEX_DISTANCE
    )
    print(f"[CLEAN] far vertices removed: {removed_vertices:,}")

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("[SKIP] 먼 꼭짓점 제거 후 Mesh가 비어 있음")
        return None

    # 2. 긴 면과 빈 공간을 덮는 면 제거
    mesh, removed_triangles = remove_bad_triangles(
        mesh,
        local_source
    )
    print(f"[CLEAN] bad triangles removed: {removed_triangles:,}")

    mesh = basic_cleanup(mesh)

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("[SKIP] 가짜 면 제거 후 Mesh가 비어 있음")
        return None

    # 3. 작은 분리 조각 제거
    mesh, removed_components = remove_small_components(mesh)
    print(
        f"[CLEAN] small component triangles removed: "
        f"{removed_components:,}"
    )

    mesh = basic_cleanup(mesh)

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("[SKIP] 작은 조각 제거 후 Mesh가 비어 있음")
        return None

    # 4. 삼각형 수 줄이기
    if (
        TARGET_TRIANGLES > 0 and
        len(mesh.triangles) > TARGET_TRIANGLES
    ):
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=TARGET_TRIANGLES
        )
        mesh = basic_cleanup(mesh)

        print(
            f"[SIMPLIFY] triangles -> "
            f"{len(mesh.triangles):,}"
        )

    # 5. 표면 부드럽게 만들기
    if SMOOTH_ITERATIONS > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=SMOOTH_ITERATIONS,
            lambda_filter=SMOOTH_LAMBDA,
            mu=SMOOTH_MU
        )
        mesh = basic_cleanup(mesh)

        print(
            f"[SMOOTH] Taubin iterations: "
            f"{SMOOTH_ITERATIONS}"
        )

    # 6. smoothing으로 원본에서 너무 멀어진 부분을 마지막으로 제거
    mesh, final_removed_vertices = remove_far_vertices(
        mesh,
        local_source,
        FINAL_MAX_VERTEX_DISTANCE
    )
    print(
        f"[FINAL CLEAN] far vertices removed: "
        f"{final_removed_vertices:,}"
    )

    mesh = basic_cleanup(mesh)

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("[SKIP] 최종 정리 후 Mesh가 비어 있음")
        return None

    # 7. 색 복원 + Normal 다시 계산
    mesh = transfer_source_colors(mesh, local_source)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    print(
        f"[AFTER] vertices={len(mesh.vertices):,}, "
        f"triangles={len(mesh.triangles):,}"
    )

    return mesh


# ------------------------------------------------------------
# 기존 출력 삭제
# ------------------------------------------------------------
def remove_old_outputs():
    old_files = glob.glob(OUTPUT_MESH_PATTERN)

    for path in old_files:
        os.remove(path)

    if old_files:
        print(f"[INFO] 이전 refined Mesh {len(old_files)}개 삭제")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    os.makedirs(PLY_DIR, exist_ok=True)

    mesh_paths = sorted(glob.glob(INPUT_MESH_PATTERN))

    if len(mesh_paths) == 0:
        raise FileNotFoundError(
            "07_object_mesh_*.ply 파일이 없어.\n"
            "먼저 object_detecting_final.py를 실행해야 해."
        )

    if not os.path.exists(SOURCE_POINT_CLOUD_PATH):
        raise FileNotFoundError(
            f"원본 객체 Point Cloud가 없어: "
            f"{SOURCE_POINT_CLOUD_PATH}"
        )

    source_pcd = o3d.io.read_point_cloud(
        SOURCE_POINT_CLOUD_PATH
    )

    if len(source_pcd.points) == 0:
        raise RuntimeError("원본 객체 Point Cloud가 비어 있어.")

    print(f"[INFO] source points: {len(source_pcd.points):,}")
    print(f"[INFO] input meshes: {len(mesh_paths)}")

    remove_old_outputs()

    refined_meshes = []

    for mesh_path in mesh_paths:
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            print(f"[SKIP] 비어 있는 Mesh: {mesh_path}")
            continue

        filename = os.path.basename(mesh_path)
        object_number = filename.replace(
            "07_object_mesh_",
            ""
        ).replace(".ply", "")

        refined_mesh = refine_mesh(
            mesh,
            source_pcd,
            filename
        )

        if refined_mesh is None:
            continue

        output_path = (
            f"{OUTPUT_MESH_PREFIX}{object_number}.ply"
        )

        ok = o3d.io.write_triangle_mesh(
            output_path,
            refined_mesh,
            write_ascii=False,
            write_vertex_normals=True,
            write_vertex_colors=True
        )

        print(
            f"[SAVE] {os.path.basename(output_path)} "
            f"-> {'OK' if ok else 'FAIL'}"
        )

        if ok:
            refined_meshes.append(refined_mesh)

    print(
        f"\n[DONE] refined Mesh 저장 완료: "
        f"{len(refined_meshes)}개"
    )

    if SHOW_RESULT and len(refined_meshes) > 0:
        print("[VIEW] 다듬은 객체 Mesh")
        o3d.visualization.draw_geometries(
            refined_meshes,
            mesh_show_back_face=True
        )


if __name__ == "__main__":
    main()
