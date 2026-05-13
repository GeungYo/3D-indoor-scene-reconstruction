# 2

import os
import numpy as np
import open3d as o3d

# =========================
# 입력 파일
# 자투리 점 제거가 끝난 point cloud
# =========================
INPUT_FILE = os.path.join(
    "ply",
    "03_objects_after_small_fragment_cleanup.ply"
)

# =========================
# 저장 파일 이름
# =========================
OUT_DETECTED_OBJECT_POINTS_ORIGINAL_COLOR = os.path.join(
    "ply",
    "04_detected_object_points_original_color.ply"
)

OUT_DETECTED_OBJECT_POINTS_CLUSTER_COLOR = os.path.join(
    "ply",
    "05_detected_object_points_cluster_color.ply"
)

OUT_DETECTED_OBJECT_BBOXES = os.path.join(
    "ply",
    "06_detected_object_bounding_boxes.ply"
)

# =========================
# 전처리
# 이미 voxel 0.02로 되어 있으면 보통 0.0 유지
# =========================
EXTRA_VOXEL = 0.0

# =========================
# DBSCAN 자동 조절 파라미터
# =========================
DBSCAN_EPS_START = 0.040
DBSCAN_MIN_POINTS = 25

# 가장 큰 클러스터가 전체의 이 비율보다 크면
# "너무 큰 덩어리"라고 보고 eps를 줄임
BIG_CLUSTER_FRAC = 0.40

# eps 줄이는 비율
EPS_SHRINK = 0.85

# 몇 번까지 다시 시도할지
EPS_RETRY = 6

# eps 최소값
DBSCAN_EPS_MIN = 0.025

# =========================
# 클러스터 필터 파라미터
# =========================
MIN_CLUSTER_POINTS = 120

# bbox 크기 필터
BOX_MIN_SIZE = 0.06

# 너무 얇은 조각 제거 기준
THIN_MIN = 0.015
THIN_KEEP_FOOTPRINT = 0.20

# 너무 길쭉한 잡음 제거 기준
ELONG_RATIO = 15.0
ELONG_REMOVE_FOOTPRINT = 0.12

# AABB 사용
USE_OBB = False

# bbox margin
USE_BBOX_MARGIN = False
BBOX_MARGIN = 0.02

# 시각화
SHOW = True


def here_dir():
    return os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(here_dir(), path)


def load_pcd(path):
    full_path = resolve_path(path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"입력 파일이 없음: {full_path}")

    pcd = o3d.io.read_point_cloud(full_path)

    if len(pcd.points) == 0:
        raise RuntimeError("입력 point cloud가 비어있어.")

    print(f"[INFO] input: {path}")
    print(f"[INFO] points: {len(pcd.points):,}")

    return pcd


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
    valid = labels[labels >= 0]

    if valid.size == 0:
        return 1.0

    counts = np.bincount(valid)
    largest = int(counts.max()) if counts.size else 0

    return largest / max(1, n_points)


def dbscan_auto(pcd):
    eps = DBSCAN_EPS_START
    best_labels = None
    best_eps = eps

    for t in range(EPS_RETRY + 1):
        labels = cluster_dbscan(
            pcd,
            eps=eps,
            min_points=DBSCAN_MIN_POINTS
        )

        if labels.size > 0 and labels.max() >= 0:
            cluster_count = int(labels.max()) + 1
        else:
            cluster_count = 0

        frac = largest_cluster_frac(labels, len(pcd.points))

        print(
            f"[INFO] DBSCAN try{t}: "
            f"eps={eps:.4f}, "
            f"clusters={cluster_count}, "
            f"largest_frac={frac:.2f}"
        )

        best_labels = labels
        best_eps = eps

        if frac <= BIG_CLUSTER_FRAC:
            break

        eps = max(DBSCAN_EPS_MIN, eps * EPS_SHRINK)

        if eps <= DBSCAN_EPS_MIN:
            break

    print(f"[INFO] selected eps: {best_eps:.4f}")

    return best_labels, best_eps


def expand_aabb(aabb, margin):
    min_bound = aabb.get_min_bound() - margin
    max_bound = aabb.get_max_bound() + margin

    new_aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound,
        max_bound
    )
    new_aabb.color = (1, 0, 0)

    return new_aabb


def bbox_to_lineset(bbox, color=(1, 0, 0)):
    if isinstance(bbox, o3d.geometry.OrientedBoundingBox):
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    else:
        line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)

    line_set.paint_uniform_color(color)

    return line_set


def merge_linesets(lineset_list):
    merged = o3d.geometry.LineSet()

    if len(lineset_list) == 0:
        return merged

    all_points = []
    all_lines = []
    all_colors = []

    point_offset = 0

    for ls in lineset_list:
        pts = np.asarray(ls.points)
        lines = np.asarray(ls.lines)

        if len(ls.colors) > 0:
            colors = np.asarray(ls.colors)
        else:
            colors = np.tile(
                np.array([[1.0, 0.0, 0.0]]),
                (len(lines), 1)
            )

        all_points.append(pts)
        all_lines.append(lines + point_offset)
        all_colors.append(colors)

        point_offset += len(pts)

    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    return merged


def colorize_by_labels(pcd, labels):
    colored = o3d.geometry.PointCloud(pcd)

    if len(labels) == 0 or labels.max() < 0:
        colored.paint_uniform_color([0.6, 0.6, 0.6])
        return colored

    max_label = int(labels.max())

    rng = np.random.default_rng(0)
    palette = rng.random((max_label + 1, 3))

    colors = np.zeros((len(labels), 3), dtype=np.float64)

    for i, label in enumerate(labels):
        if label >= 0:
            colors[i] = palette[label]
        else:
            colors[i] = np.array([0.15, 0.15, 0.15])

    colored.colors = o3d.utility.Vector3dVector(colors)

    return colored


def is_good_bbox(bbox):
    ex, ey, ez = bbox.get_extent()

    # 너무 작은 조각 제거
    if ex < BOX_MIN_SIZE and ey < BOX_MIN_SIZE and ez < BOX_MIN_SIZE:
        return False

    e_sorted = sorted([float(ex), float(ey), float(ez)])
    min_e = e_sorted[0]
    mid_e = e_sorted[1]
    max_e = e_sorted[2]

    footprint = max_e * mid_e

    # 너무 얇고 면적도 작으면 제거
    if min_e < THIN_MIN:
        if footprint < THIN_KEEP_FOOTPRINT:
            return False

    # 너무 길쭉하고 면적도 작으면 제거
    if (max_e / (min_e + 1e-9)) > ELONG_RATIO:
        if footprint < ELONG_REMOVE_FOOTPRINT:
            return False

    return True


def detect_objects(pcd):
    if len(pcd.points) == 0:
        return (
            o3d.geometry.PointCloud(),
            o3d.geometry.PointCloud(),
            o3d.geometry.LineSet()
        )

    # 1. DBSCAN 자동 eps 조절
    labels, used_eps = dbscan_auto(pcd)
    labels = labels.astype(np.int32)

    if len(labels) == 0 or labels.max() < 0:
        print("[WARN] DBSCAN에서 유효한 객체 클러스터가 없음.")
        return (
            o3d.geometry.PointCloud(),
            o3d.geometry.PointCloud(),
            o3d.geometry.LineSet()
        )

    cluster_count = labels.max() + 1
    print(f"[INFO] final DBSCAN clusters: {cluster_count}")

    keep_indices = []
    bbox_linesets = []

    # 2. 클러스터별 bbox 생성
    for cluster_id in range(cluster_count):
        idx = np.where(labels == cluster_id)[0]

        if len(idx) < MIN_CLUSTER_POINTS:
            print(f"[SKIP] cluster {cluster_id}: {len(idx)} points")
            continue

        cluster_pcd = pcd.select_by_index(idx)

        if USE_OBB:
            bbox = cluster_pcd.get_oriented_bounding_box()
        else:
            bbox = cluster_pcd.get_axis_aligned_bounding_box()

        if USE_BBOX_MARGIN:
            bbox = expand_aabb(bbox, BBOX_MARGIN)

        if not is_good_bbox(bbox):
            extent = np.asarray(bbox.get_extent())
            print(
                f"[SKIP] cluster {cluster_id}: bad bbox, "
                f"extent=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})"
            )
            continue

        bbox.color = (1, 0, 0)
        bbox_linesets.append(bbox_to_lineset(bbox, color=(1, 0, 0)))

        keep_indices.extend(idx.tolist())

        extent = np.asarray(bbox.get_extent())

        print(
            f"[KEEP] cluster {cluster_id}: "
            f"points={len(idx):,}, "
            f"bbox=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})"
        )

    if len(keep_indices) == 0:
        print("[WARN] 필터를 통과한 객체가 없음.")
        return (
            o3d.geometry.PointCloud(),
            o3d.geometry.PointCloud(),
            o3d.geometry.LineSet()
        )

    keep_indices = sorted(list(set(keep_indices)))

    # 원본 색 유지 버전
    detected_object_points_original_color = pcd.select_by_index(keep_indices)

    # 혹시 입력 pcd에 색이 없으면 회색으로 표시
    if not detected_object_points_original_color.has_colors():
        detected_object_points_original_color.paint_uniform_color([0.6, 0.6, 0.6])

    # 클러스터 색 버전
    detected_labels = labels[keep_indices]
    detected_object_points_cluster_color = colorize_by_labels(
        detected_object_points_original_color,
        detected_labels
    )

    bbox_lines = merge_linesets(bbox_linesets)

    return (
        detected_object_points_original_color,
        detected_object_points_cluster_color,
        bbox_lines
    )


def main():
    pcd = load_pcd(INPUT_FILE)

    if EXTRA_VOXEL and EXTRA_VOXEL > 0:
        pcd = pcd.voxel_down_sample(EXTRA_VOXEL)
        print(f"[INFO] after EXTRA_VOXEL({EXTRA_VOXEL}): {len(pcd.points):,}")

    # 1. 객체 detection
    (
        detected_object_points_original_color,
        detected_object_points_cluster_color,
        bbox_lines
    ) = detect_objects(pcd)

    # 2. 저장
    os.makedirs(resolve_path("ply"), exist_ok=True)

    if len(detected_object_points_original_color.points) > 0:
        o3d.io.write_point_cloud(
            resolve_path(OUT_DETECTED_OBJECT_POINTS_ORIGINAL_COLOR),
            detected_object_points_original_color,
            write_ascii=True
        )
        print(f"[SAVE] {OUT_DETECTED_OBJECT_POINTS_ORIGINAL_COLOR}")
    else:
        print("[WARN] original color object points가 비어 있어서 저장하지 않음.")

    if len(detected_object_points_cluster_color.points) > 0:
        o3d.io.write_point_cloud(
            resolve_path(OUT_DETECTED_OBJECT_POINTS_CLUSTER_COLOR),
            detected_object_points_cluster_color,
            write_ascii=True
        )
        print(f"[SAVE] {OUT_DETECTED_OBJECT_POINTS_CLUSTER_COLOR}")
    else:
        print("[WARN] cluster color object points가 비어 있어서 저장하지 않음.")

    if len(bbox_lines.points) > 0:
        o3d.io.write_line_set(
            resolve_path(OUT_DETECTED_OBJECT_BBOXES),
            bbox_lines,
            write_ascii=True
        )
        print(f"[SAVE] {OUT_DETECTED_OBJECT_BBOXES}")
    else:
        print("[WARN] bbox가 비어 있어서 저장하지 않음.")

    # 3. 시각화
    if SHOW:
        if len(detected_object_points_original_color.points) > 0:
            print("\n[VIEW 1] 원본 색 객체 점 + bbox")
            o3d.visualization.draw_geometries(
                [detected_object_points_original_color, bbox_lines]
            )

        if len(detected_object_points_cluster_color.points) > 0:
            print("\n[VIEW 2] 클러스터 색 객체 점 + bbox")
            o3d.visualization.draw_geometries(
                [detected_object_points_cluster_color, bbox_lines]
            )


if __name__ == "__main__":
    main()