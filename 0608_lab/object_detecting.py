import glob
import os

import numpy as np
import open3d as o3d


# ============================================================
# 입력: 벽/바닥 제거와 잡음 제거가 끝난 객체 후보 Point Cloud
# ============================================================
INPUT_FILE = os.path.join(
    "ply",
    "03_objects_after_small_fragment_cleanup.ply"
)

# ============================================================
# 기존 출력 파일
# ============================================================
OUT_OBJECT_POINTS_ORIGINAL_COLOR = os.path.join(
    "ply",
    "04_detected_object_points_original_color.ply"
)

OUT_OBJECT_POINTS_CLUSTER_COLOR = os.path.join(
    "ply",
    "05_detected_object_points_cluster_color.ply"
)

OUT_OBJECT_BBOXES = os.path.join(
    "ply",
    "06_detected_object_bounding_boxes.ply"
)

# ============================================================
# 새 출력: 객체별 Mesh
# 실제 파일명 예시: 07_object_mesh_000.ply
# ============================================================
OBJECT_MESH_PATTERN = os.path.join(
    "ply",
    "07_object_mesh_*.ply"
)

OBJECT_MESH_PREFIX = os.path.join(
    "ply",
    "07_object_mesh_"
)

# ============================================================
# 전처리
# 이미 충분히 다운샘플링된 상태라면 0.0 유지
# ============================================================
EXTRA_VOXEL = 0.0

# ============================================================
# DBSCAN
# ============================================================
DBSCAN_EPS_START = 0.04
DBSCAN_MIN_POINTS = 30

# 너무 큰 한 덩어리가 생기면 eps를 자동으로 줄임
BIG_CLUSTER_FRAC = 0.65
EPS_SHRINK = 0.85
EPS_RETRY = 4
DBSCAN_EPS_MIN = 0.03

# ============================================================
# 선택적 클러스터 병합
# 붙은 사물을 더 나누는 것이 목적이므로 False 유지
# ============================================================
ENABLE_MERGE = False
MERGE_DIST = 0.06
MIN_POINTS_FOR_MERGE = 80

# ============================================================
# Bounding Box 생성 / 필터
# ============================================================
MIN_CLUSTER_POINTS = 150

# False: AABB, True: OBB
USE_OBB = False

BOX_MIN_SIZE = 0.08

THIN_MIN = 0.02
THIN_KEEP_FOOTPRINT = 0.30

ELONG_RATIO = 12.0
ELONG_REMOVE_FOOTPRINT = 0.15

# ============================================================
# Mesh 생성 설정
# ============================================================
# Mesh를 만들 최소 점 개수
MESH_MIN_POINTS = 150

# Normal 계산 시 주변 탐색 범위
MESH_NORMAL_RADIUS = 0.08
MESH_NORMAL_MAX_NN = 50
MESH_ORIENT_K = 30

# Poisson Surface Reconstruction 정밀도
# 높일수록 촘촘하지만 느리고 Mesh가 무거워짐
POISSON_DEPTH = 8
POISSON_SCALE = 1.02

# Poisson이 바깥쪽에 만든 낮은 밀도 면 제거 비율
# 0.02 = 가장 낮은 2% 제거
DENSITY_REMOVE_QUANTILE = 0.08

# 원래 객체 Bounding Box 밖으로 튀어나온 Mesh 제거 여유
MESH_CROP_MARGIN = 0.0

# 원본 포인트에서 이 거리보다 멀리 떨어진 가짜 Mesh 제거
# 입력 포인트가 약 0.02m 간격이므로 우선 0.04m 사용
MESH_MAX_POINT_DISTANCE = 0.04

# 객체 하나당 삼각형이 너무 많으면 단순화
# 0이면 단순화하지 않음
MESH_TARGET_TRIANGLES = 50000

# ============================================================
# 시각화
# ============================================================
SHOW = True
SHOW_MESH = True


# ------------------------------------------------------------
# 경로 / 입출력
# ------------------------------------------------------------
def here_dir():
    return os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(here_dir(), path)


def load_pcd(filename):
    path = resolve_path(filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"입력 파일이 없음: {path}")

    pcd = o3d.io.read_point_cloud(path)

    if len(pcd.points) == 0:
        raise RuntimeError("포인트가 0개야.")

    print(f"[INFO] input: {path}")
    print(f"[INFO] points: {len(pcd.points):,}")

    return pcd


def save_point_cloud(path, pcd):
    if len(pcd.points) == 0:
        print(f"[WARN] 저장 스킵, 포인트 0개: {path}")
        return

    full_path = resolve_path(path)
    ok = o3d.io.write_point_cloud(
        full_path,
        pcd,
        write_ascii=True
    )

    print(f"[SAVE] {path} -> {'OK' if ok else 'FAIL'}")


def save_lineset_as_ply(path, linesets):
    if len(linesets) == 0:
        print("[WARN] 저장할 Bounding Box가 없어 저장하지 않음.")
        return

    all_points = []
    all_lines = []
    all_colors = []
    point_offset = 0

    for lineset in linesets:
        points = np.asarray(lineset.points)
        lines = np.asarray(lineset.lines)

        if len(lineset.colors) > 0:
            colors = np.asarray(lineset.colors)
        else:
            colors = np.tile(
                np.array([[1.0, 0.0, 0.0]]),
                (len(lines), 1)
            )

        all_points.append(points)
        all_lines.append(lines + point_offset)
        all_colors.append(colors)

        point_offset += len(points)

    merged = o3d.geometry.LineSet()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    full_path = resolve_path(path)
    ok = o3d.io.write_line_set(
        full_path,
        merged,
        write_ascii=True
    )

    print(f"[SAVE] {path} -> {'OK' if ok else 'FAIL'}")


def remove_old_object_meshes():
    """이전 실행에서 남은 객체 Mesh가 섞이지 않도록 삭제한다."""
    old_paths = glob.glob(resolve_path(OBJECT_MESH_PATTERN))

    for path in old_paths:
        os.remove(path)

    if old_paths:
        print(f"[INFO] 이전 객체 Mesh {len(old_paths)}개 삭제")


# ------------------------------------------------------------
# DBSCAN + 자동 eps 조절
# ------------------------------------------------------------
def cluster_dbscan(pcd, eps, min_points):
    if len(pcd.points) == 0:
        return np.array([], dtype=np.int32)

    labels = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
        print_progress=True
    )

    return np.array(labels, dtype=np.int32)


def largest_cluster_frac(labels, n_points):
    valid_labels = labels[labels >= 0]

    if valid_labels.size == 0:
        return 1.0

    counts = np.bincount(valid_labels)
    largest = int(counts.max()) if counts.size else 0

    return largest / max(1, n_points)


def dbscan_auto(pcd):
    eps = DBSCAN_EPS_START
    best_labels = None
    best_eps = eps

    for try_index in range(EPS_RETRY + 1):
        labels = cluster_dbscan(
            pcd,
            eps,
            DBSCAN_MIN_POINTS
        )

        frac = largest_cluster_frac(labels, len(pcd.points))

        if labels.size > 0 and labels.max() >= 0:
            cluster_count = int(labels.max()) + 1
        else:
            cluster_count = 0

        print(
            f"[INFO] DBSCAN try{try_index}: "
            f"eps={eps:.3f}, "
            f"clusters={cluster_count}, "
            f"largest_frac={frac:.2f}"
        )

        best_labels = labels
        best_eps = eps

        if frac <= BIG_CLUSTER_FRAC:
            break

        eps = max(DBSCAN_EPS_MIN, eps * EPS_SHRINK)

    print(f"[INFO] selected eps: {best_eps:.3f}")

    return best_labels, best_eps


# ------------------------------------------------------------
# 클러스터 색칠
# ------------------------------------------------------------
def colorize_by_labels(pcd, labels):
    colored = o3d.geometry.PointCloud(pcd)

    if len(colored.points) == 0 or labels.size == 0:
        return colored

    max_label = int(labels.max())

    if max_label < 0:
        colored.paint_uniform_color([0.5, 0.5, 0.5])
        return colored

    rng = np.random.default_rng(0)
    palette = rng.random((max_label + 1, 3))

    colors = np.zeros((len(labels), 3), dtype=np.float64)

    for index, label in enumerate(labels):
        if label >= 0:
            colors[index] = palette[label]
        else:
            colors[index] = np.array([0.2, 0.2, 0.2])

    colored.colors = o3d.utility.Vector3dVector(colors)

    return colored


# ------------------------------------------------------------
# 선택적 클러스터 병합
# ------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def aabb_distance(min1, max1, min2, max2):
    dx = max(0.0, max(min2[0] - max1[0], min1[0] - max2[0]))
    dy = max(0.0, max(min2[1] - max1[1], min1[1] - max2[1]))
    dz = max(0.0, max(min2[2] - max1[2], min1[2] - max2[2]))

    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def merge_clusters_by_aabb(points, labels, merge_dist):
    unique_labels = [label for label in np.unique(labels) if label >= 0]

    if len(unique_labels) == 0:
        return labels

    indices_by_label = {
        label: np.where(labels == label)[0]
        for label in unique_labels
    }

    large_labels = [
        label for label in unique_labels
        if indices_by_label[label].size >= MIN_POINTS_FOR_MERGE
    ]

    if len(large_labels) < 2:
        return labels

    mins = np.zeros((len(large_labels), 3), dtype=np.float64)
    maxs = np.zeros((len(large_labels), 3), dtype=np.float64)
    centers = np.zeros((len(large_labels), 3), dtype=np.float64)

    for index, label in enumerate(large_labels):
        cluster_points = points[indices_by_label[label]]
        mins[index] = cluster_points.min(axis=0)
        maxs[index] = cluster_points.max(axis=0)
        centers[index] = (mins[index] + maxs[index]) * 0.5

    union_find = UnionFind(len(large_labels))

    cell_size = merge_dist * 2.0
    grid = {}

    def cell_key(center):
        return tuple(np.floor(center / cell_size).astype(int))

    for index, center in enumerate(centers):
        key = cell_key(center)
        grid.setdefault(key, []).append(index)

    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]

    for key, items in grid.items():
        candidates = []

        for offset in neighbor_offsets:
            neighbor_key = (
                key[0] + offset[0],
                key[1] + offset[1],
                key[2] + offset[2]
            )
            candidates.extend(grid.get(neighbor_key, []))

        candidates = sorted(set(candidates))
        items = sorted(items)

        for i in items:
            for j in candidates:
                if j <= i:
                    continue

                distance = aabb_distance(
                    mins[i], maxs[i],
                    mins[j], maxs[j]
                )

                if distance <= merge_dist:
                    union_find.union(i, j)

    root_to_new = {}
    next_label = 0
    large_label_to_new = {}

    for index, label in enumerate(large_labels):
        root = union_find.find(index)

        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1

        large_label_to_new[label] = root_to_new[root]

    merged = labels.copy()
    offset = merged.max() + 1 if merged.size > 0 and merged.max() >= 0 else 0

    for label in large_labels:
        merged[labels == label] = offset + large_label_to_new[label]

    unique_after = [label for label in np.unique(merged) if label >= 0]
    remap = {
        old_label: new_label
        for new_label, old_label in enumerate(sorted(unique_after))
    }

    merged_result = merged.copy()

    for old_label, new_label in remap.items():
        merged_result[merged == old_label] = new_label

    print(
        f"[INFO] merge: big clusters {len(large_labels)} "
        f"-> merged clusters {len(remap)}"
    )

    return merged_result


# ------------------------------------------------------------
# Bounding Box 생성 / 필터
# ------------------------------------------------------------
def create_bbox(cluster):
    if USE_OBB:
        bbox = cluster.get_oriented_bounding_box()
    else:
        bbox = cluster.get_axis_aligned_bounding_box()

    bbox.color = (1, 0, 0)
    return bbox


def bbox_pass_filter(bbox):
    extent_x, extent_y, extent_z = bbox.get_extent()

    if (
        extent_x < BOX_MIN_SIZE and
        extent_y < BOX_MIN_SIZE and
        extent_z < BOX_MIN_SIZE
    ):
        return False

    sorted_extents = sorted([
        float(extent_x),
        float(extent_y),
        float(extent_z)
    ])

    min_extent = sorted_extents[0]
    middle_extent = sorted_extents[1]
    max_extent = sorted_extents[2]

    footprint = max_extent * middle_extent

    if min_extent < THIN_MIN and footprint < THIN_KEEP_FOOTPRINT:
        return False

    if (
        max_extent / (min_extent + 1e-9) > ELONG_RATIO and
        footprint < ELONG_REMOVE_FOOTPRINT
    ):
        return False

    return True


def bbox_to_lineset(bbox):
    if isinstance(bbox, o3d.geometry.OrientedBoundingBox):
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    else:
        lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)

    lineset.paint_uniform_color([1, 0, 0])
    return lineset


# ------------------------------------------------------------
# Point Cloud 한 덩어리 -> Mesh
# ------------------------------------------------------------
def transfer_point_colors_to_mesh(source_pcd, mesh):
    """각 Mesh 꼭짓점에 가장 가까운 원본 점의 색을 복사한다."""
    vertex_count = len(mesh.vertices)

    if vertex_count == 0:
        return mesh

    if not source_pcd.has_colors():
        mesh.paint_uniform_color([0.6, 0.6, 0.6])
        return mesh

    source_colors = np.asarray(source_pcd.colors)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_colors = np.zeros((vertex_count, 3), dtype=np.float64)

    kd_tree = o3d.geometry.KDTreeFlann(source_pcd)

    for index, vertex in enumerate(mesh_vertices):
        found, nearest_indices, _ = kd_tree.search_knn_vector_3d(vertex, 1)

        if found > 0:
            mesh_colors[index] = source_colors[nearest_indices[0]]
        else:
            mesh_colors[index] = [0.6, 0.6, 0.6]

    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    return mesh


def pointcloud_to_mesh(cluster, cluster_label):
    point_count = len(cluster.points)

    if point_count < MESH_MIN_POINTS:
        print(
            f"[MESH SKIP] cluster {cluster_label}: "
            f"점이 너무 적음 ({point_count:,})"
        )
        return None

    mesh_input = o3d.geometry.PointCloud(cluster)

    # Poisson은 Normal이 반드시 필요하다.
    mesh_input.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=MESH_NORMAL_RADIUS,
            max_nn=MESH_NORMAL_MAX_NN
        )
    )

    orient_k = min(MESH_ORIENT_K, point_count - 1)

    if orient_k >= 3:
        try:
            mesh_input.orient_normals_consistent_tangent_plane(orient_k)
        except RuntimeError as error:
            print(
                f"[WARN] cluster {cluster_label}: "
                f"Normal 방향 통일 실패, 계산된 Normal 그대로 사용\n"
                f"       {error}"
            )

    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            mesh_input,
            depth=POISSON_DEPTH,
            scale=POISSON_SCALE,
            linear_fit=False
        )
    except RuntimeError as error:
        print(f"[MESH FAIL] cluster {cluster_label}: {error}")
        return None

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print(f"[MESH FAIL] cluster {cluster_label}: 생성 결과가 비어 있음")
        return None

    # 밀도가 매우 낮은 가짜 표면 제거
    densities = np.asarray(densities)

    if len(densities) > 0 and DENSITY_REMOVE_QUANTILE > 0:
        density_threshold = np.quantile(
            densities,
            DENSITY_REMOVE_QUANTILE
        )
        mesh.remove_vertices_by_mask(densities < density_threshold)

    # 원본 객체 영역보다 바깥으로 퍼진 표면 제거
    source_bbox = mesh_input.get_axis_aligned_bounding_box()
    min_bound = source_bbox.get_min_bound() - MESH_CROP_MARGIN
    max_bound = source_bbox.get_max_bound() + MESH_CROP_MARGIN

    crop_bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound,
        max_bound
    )

    mesh = mesh.crop(crop_bbox)

    # Poisson이 빈 공간에 만든 가짜 표면 제거
    # Mesh 꼭짓점과 원본 Point Cloud 사이의 거리를 계산한다.
    if MESH_MAX_POINT_DISTANCE > 0 and len(mesh.vertices) > 0:
        mesh_vertices_pcd = o3d.geometry.PointCloud()
        mesh_vertices_pcd.points = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices)
        )

        distances_to_source = np.asarray(
            mesh_vertices_pcd.compute_point_cloud_distance(mesh_input)
        )

        far_vertex_mask = distances_to_source > MESH_MAX_POINT_DISTANCE
        removed_far_vertices = int(np.sum(far_vertex_mask))

        mesh.remove_vertices_by_mask(far_vertex_mask)

        print(
            f"[MESH CLEAN] cluster {cluster_label}: "
            f"원본 점에서 멀리 떨어진 vertex "
            f"{removed_far_vertices:,}개 제거"
        )

    # 불필요하거나 잘못된 삼각형 정리
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print(f"[MESH FAIL] cluster {cluster_label}: 정리 후 Mesh가 비어 있음")
        return None

    # 너무 무거운 Mesh는 삼각형 수를 줄인다.
    if (
        MESH_TARGET_TRIANGLES > 0 and
        len(mesh.triangles) > MESH_TARGET_TRIANGLES
    ):
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=MESH_TARGET_TRIANGLES
        )
        mesh.remove_unreferenced_vertices()

    mesh = transfer_point_colors_to_mesh(mesh_input, mesh)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    print(
        f"[MESH OK] cluster {cluster_label}: "
        f"vertices={len(mesh.vertices):,}, "
        f"triangles={len(mesh.triangles):,}"
    )

    return mesh


def save_object_mesh(mesh, cluster_label):
    filename = f"{OBJECT_MESH_PREFIX}{cluster_label:03d}.ply"
    full_path = resolve_path(filename)

    ok = o3d.io.write_triangle_mesh(
        full_path,
        mesh,
        write_ascii=False,
        write_vertex_normals=True,
        write_vertex_colors=True
    )

    print(f"[SAVE] {filename} -> {'OK' if ok else 'FAIL'}")
    return filename if ok else None


# ------------------------------------------------------------
# 객체 추출 + 객체별 Mesh 생성
# ------------------------------------------------------------
def extract_detected_objects(pcd, labels):
    """Bounding Box 필터를 통과한 클러스터만 객체와 Mesh로 남긴다."""
    keep_indices = []
    keep_labels = []
    linesets = []
    object_meshes = []

    valid_labels = [label for label in np.unique(labels) if label >= 0]

    for label in valid_labels:
        indices = np.where(labels == label)[0]

        if len(indices) < MIN_CLUSTER_POINTS:
            print(f"[SKIP] cluster {label}: {len(indices)} points")
            continue

        cluster = pcd.select_by_index(indices.tolist())

        if len(cluster.points) == 0:
            continue

        bbox = create_bbox(cluster)

        if not bbox_pass_filter(bbox):
            extent_x, extent_y, extent_z = bbox.get_extent()
            print(
                f"[SKIP] cluster {label}: Bounding Box 필터 실패, "
                f"points={len(indices)}, "
                f"extent=({extent_x:.2f}, {extent_y:.2f}, {extent_z:.2f})"
            )
            continue

        linesets.append(bbox_to_lineset(bbox))
        keep_indices.extend(indices.tolist())
        keep_labels.extend([label] * len(indices))

        extent_x, extent_y, extent_z = bbox.get_extent()
        print(
            f"[KEEP] cluster {label}: "
            f"points={len(indices):,}, "
            f"extent=({extent_x:.2f}, {extent_y:.2f}, {extent_z:.2f})"
        )

        mesh = pointcloud_to_mesh(cluster, int(label))

        if mesh is not None:
            saved_path = save_object_mesh(mesh, int(label))

            if saved_path is not None:
                object_meshes.append(mesh)

    if len(keep_indices) == 0:
        return (
            o3d.geometry.PointCloud(),
            o3d.geometry.PointCloud(),
            [],
            np.array([], dtype=np.int32),
            []
        )

    keep_indices = np.array(keep_indices, dtype=np.int64)
    keep_labels = np.array(keep_labels, dtype=np.int32)

    detected_original_color = pcd.select_by_index(keep_indices.tolist())

    if not detected_original_color.has_colors():
        detected_original_color.paint_uniform_color([0.6, 0.6, 0.6])

    detected_cluster_color = colorize_by_labels(
        detected_original_color,
        keep_labels
    )

    return (
        detected_original_color,
        detected_cluster_color,
        linesets,
        keep_labels,
        object_meshes
    )


# ------------------------------------------------------------
# Mesh 시각화
# ------------------------------------------------------------
def show_mesh_scene(meshes):
    if len(meshes) == 0:
        return

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name="Detected object meshes",
        width=1280,
        height=800
    )

    for mesh in meshes:
        visualizer.add_geometry(mesh)

    options = visualizer.get_render_option()
    options.mesh_show_back_face = True

    visualizer.run()
    visualizer.destroy_window()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    os.makedirs(resolve_path("ply"), exist_ok=True)
    remove_old_object_meshes()

    pcd = load_pcd(INPUT_FILE)

    if EXTRA_VOXEL and EXTRA_VOXEL > 0:
        pcd = pcd.voxel_down_sample(EXTRA_VOXEL)
        print(
            f"[INFO] after EXTRA_VOXEL({EXTRA_VOXEL}): "
            f"{len(pcd.points):,}"
        )

    # 1) DBSCAN 자동 eps 조절
    labels, used_eps = dbscan_auto(pcd)
    labels = labels.astype(np.int32)

    # 2) 선택적 클러스터 병합
    if ENABLE_MERGE and labels.size > 0 and labels.max() >= 0:
        labels = merge_clusters_by_aabb(
            np.asarray(pcd.points),
            labels,
            MERGE_DIST
        )

    if labels.size > 0 and labels.max() >= 0:
        final_cluster_count = int(labels.max()) + 1
    else:
        final_cluster_count = 0

    print(f"[INFO] final clusters before Bounding Box filter: {final_cluster_count}")
    print(f"[INFO] DBSCAN eps used: {used_eps:.3f}")

    # 3) 객체 추출 + 객체별 Mesh 생성
    (
        detected_original_color,
        detected_cluster_color,
        bbox_linesets,
        kept_labels,
        object_meshes
    ) = extract_detected_objects(pcd, labels)

    print(f"[INFO] final detected boxes: {len(bbox_linesets)}")
    print(f"[INFO] final object meshes: {len(object_meshes)}")

    # 4) 기존 Point Cloud / Bounding Box 저장
    save_point_cloud(
        OUT_OBJECT_POINTS_ORIGINAL_COLOR,
        detected_original_color
    )

    save_point_cloud(
        OUT_OBJECT_POINTS_CLUSTER_COLOR,
        detected_cluster_color
    )

    save_lineset_as_ply(
        OUT_OBJECT_BBOXES,
        bbox_linesets
    )

    # 5) 시각화
    if SHOW:
        if len(detected_original_color.points) > 0:
            print("\n[VIEW 1] 원본 색 객체 Point Cloud + Bounding Box")
            geometries = [detected_original_color]
            geometries.extend(bbox_linesets)
            o3d.visualization.draw_geometries(geometries)

        if len(detected_cluster_color.points) > 0:
            print("\n[VIEW 2] 클러스터별 색 객체 Point Cloud + Bounding Box")
            geometries = [detected_cluster_color]
            geometries.extend(bbox_linesets)
            o3d.visualization.draw_geometries(geometries)

        if SHOW_MESH and len(object_meshes) > 0:
            print("\n[VIEW 3] 객체별 Mesh")
            show_mesh_scene(object_meshes)


if __name__ == "__main__":
    main()
