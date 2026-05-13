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

# 밀집도 낮아서 제거된 연결부 확인용
OUT_LOW_DENSITY_REMOVED_POINTS = os.path.join(
    "ply",
    "07_removed_low_density_bridge_points.ply"
)

# =========================
# 전처리
# 이미 voxel 0.02로 되어 있으면 보통 0.0 유지
# =========================
EXTRA_VOXEL = 0.0

# =========================
# 1차 DBSCAN 자동 조절 파라미터
# =========================
DBSCAN_EPS_START = 0.055
DBSCAN_MIN_POINTS = 25

# 가장 큰 클러스터가 전체의 이 비율보다 크면
# "너무 큰 덩어리"라고 보고 eps를 줄임
BIG_CLUSTER_FRAC = 0.45

EPS_SHRINK = 0.85
EPS_RETRY = 6
DBSCAN_EPS_MIN = 0.025

# =========================
# 큰 클러스터 재분할 파라미터
# =========================
ENABLE_SPLIT_BIG_CLUSTER = True

# 아래 조건에 걸리면 "큰 덩어리"로 보고 내부를 한 번 더 처리
SPLIT_MIN_POINTS = 2500
SPLIT_MAX_EXTENT = 1.20
SPLIT_MIN_VOLUME = 0.45

# =========================
# 밀집도 낮은 연결부 제거 파라미터
# =========================
ENABLE_DENSITY_BRIDGE_CUT = True

# 큰 클러스터 내부에서만 적용
# 반경 10cm 안의 이웃 수를 봄
DENSITY_RADIUS = 0.10

# 반경 안 이웃 점 수가 이 값보다 적으면 연결부/저밀도 점으로 판단
# 너무 많이 지워지면 50~60
# 그래도 계속 붙으면 100~120
MIN_DENSITY_NEIGHBORS = 120

# 밀집도 필터 후 남은 점이 너무 적으면 분할을 포기하고 원래 클러스터 유지
MIN_DENSITY_REMAIN_POINTS = 300
MIN_DENSITY_REMAIN_FRAC = 0.20

# =========================
# 재분할 DBSCAN
# =========================
# 1차 eps보다 더 작은 값 사용
SUB_EPS_RATIO = 0.65
SUB_DBSCAN_EPS_MIN = 0.020
SUB_DBSCAN_MIN_POINTS = 20

# 재분할된 작은 클러스터 중 이보다 작은 것은 버림
MIN_SUB_CLUSTER_POINTS = 80

# =========================
# 최종 클러스터 필터 파라미터
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
SHOW_LOW_DENSITY_REMOVED = True


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


def merge_pcds(pcd_list):
    merged = o3d.geometry.PointCloud()

    for pcd in pcd_list:
        if len(pcd.points) > 0:
            merged += pcd

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


def is_big_cluster_for_split(cluster_pcd):
    point_count = len(cluster_pcd.points)

    if point_count < SPLIT_MIN_POINTS:
        return False

    aabb = cluster_pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(aabb.get_extent())

    ex, ey, ez = extent
    max_extent = float(max(ex, ey, ez))
    volume = float(ex * ey * ez)

    if max_extent >= SPLIT_MAX_EXTENT:
        return True

    if volume >= SPLIT_MIN_VOLUME:
        return True

    return False


def cut_low_density_bridge_points(cluster_pcd, original_indices):
    """
    큰 클러스터 내부에서 local density가 낮은 점을 제거한다.
    original_indices는 cluster_pcd의 각 점이 원본 pcd에서 몇 번째 점인지 나타낸다.
    """
    if not ENABLE_DENSITY_BRIDGE_CUT:
        return cluster_pcd, original_indices, np.array([], dtype=np.int64)

    point_count = len(cluster_pcd.points)

    if point_count == 0:
        return cluster_pcd, original_indices, np.array([], dtype=np.int64)

    kdtree = o3d.geometry.KDTreeFlann(cluster_pcd)
    keep_local_indices = []
    removed_local_indices = []

    for i in range(point_count):
        # radius 안의 이웃 개수 계산
        # 자기 자신도 포함됨
        k, idx, _ = kdtree.search_radius_vector_3d(
            cluster_pcd.points[i],
            DENSITY_RADIUS
        )

        if k >= MIN_DENSITY_NEIGHBORS:
            keep_local_indices.append(i)
        else:
            removed_local_indices.append(i)

    keep_local_indices = np.array(keep_local_indices, dtype=np.int64)
    removed_local_indices = np.array(removed_local_indices, dtype=np.int64)

    remain_count = len(keep_local_indices)
    remain_frac = remain_count / max(1, point_count)

    print(
        f"    [DENSITY] before={point_count:,}, "
        f"keep={remain_count:,}, "
        f"remove={len(removed_local_indices):,}, "
        f"remain_frac={remain_frac:.2f}"
    )

    # 너무 많이 제거되면 위험하니까 원래 클러스터 유지
    if remain_count < MIN_DENSITY_REMAIN_POINTS:
        print("    [DENSITY FAIL] 남은 점이 너무 적어서 밀집도 제거 취소")
        return cluster_pcd, original_indices, np.array([], dtype=np.int64)

    if remain_frac < MIN_DENSITY_REMAIN_FRAC:
        print("    [DENSITY FAIL] 너무 많이 제거되어 밀집도 제거 취소")
        return cluster_pcd, original_indices, np.array([], dtype=np.int64)

    dense_pcd = cluster_pcd.select_by_index(keep_local_indices.tolist())
    dense_original_indices = original_indices[keep_local_indices]
    removed_original_indices = original_indices[removed_local_indices]

    return dense_pcd, dense_original_indices, removed_original_indices


def split_big_cluster(cluster_pcd, original_indices, base_eps):
    """
    큰 클러스터 하나를 처리한다.
    1) 밀집도 낮은 연결부 제거
    2) 남은 점으로 작은 eps DBSCAN
    3) 2개 이상으로 나뉘면 분할 적용
    4) 아니면 원래 클러스터 유지
    """
    print("    [STEP] low-density bridge cut")

    dense_pcd, dense_original_indices, removed_bridge_indices = cut_low_density_bridge_points(
        cluster_pcd,
        original_indices
    )

    sub_eps = max(SUB_DBSCAN_EPS_MIN, base_eps * SUB_EPS_RATIO)

    print("    [STEP] sub DBSCAN")

    sub_labels = cluster_dbscan(
        dense_pcd,
        eps=sub_eps,
        min_points=SUB_DBSCAN_MIN_POINTS
    )

    if len(sub_labels) == 0 or sub_labels.max() < 0:
        print("    [SPLIT FAIL] sub DBSCAN valid cluster 없음")
        return [original_indices], np.array([], dtype=np.int64)

    sub_cluster_count = int(sub_labels.max()) + 1

    print(
        f"    [SPLIT] sub_eps={sub_eps:.4f}, "
        f"sub_clusters={sub_cluster_count}"
    )

    splitted_indices = []

    for sub_id in range(sub_cluster_count):
        sub_local_idx = np.where(sub_labels == sub_id)[0]

        if len(sub_local_idx) < MIN_SUB_CLUSTER_POINTS:
            print(
                f"    [SUB SKIP] sub_cluster {sub_id}: "
                f"{len(sub_local_idx)} points"
            )
            continue

        sub_original_idx = dense_original_indices[sub_local_idx]
        splitted_indices.append(sub_original_idx)

        print(
            f"    [SUB KEEP] sub_cluster {sub_id}: "
            f"{len(sub_original_idx):,} points"
        )

    # 실제로 2개 이상으로 나뉘지 않았으면 원래 큰 클러스터 유지
    if len(splitted_indices) < 2:
        print("    [SPLIT FAIL] 유효한 sub cluster가 2개 미만이라 원래 클러스터 유지")
        return [original_indices], np.array([], dtype=np.int64)

    # 분할이 성공한 경우에만 제거된 low-density bridge 점을 실제로 제외
    return splitted_indices, removed_bridge_indices


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


def make_bbox_from_cluster(cluster_pcd):
    if USE_OBB:
        bbox = cluster_pcd.get_oriented_bounding_box()
    else:
        bbox = cluster_pcd.get_axis_aligned_bounding_box()

    if USE_BBOX_MARGIN:
        bbox = expand_aabb(bbox, BBOX_MARGIN)

    bbox.color = (1, 0, 0)

    return bbox


def detect_objects(pcd):
    if len(pcd.points) == 0:
        return (
            o3d.geometry.PointCloud(),
            o3d.geometry.PointCloud(),
            o3d.geometry.LineSet(),
            o3d.geometry.PointCloud()
        )

    # =========================
    # 1. 1차 DBSCAN
    # =========================
    labels, used_eps = dbscan_auto(pcd)
    labels = labels.astype(np.int32)

    if len(labels) == 0 or labels.max() < 0:
        print("[WARN] DBSCAN에서 유효한 객체 클러스터가 없음.")
        return (
            o3d.geometry.PointCloud(),
            o3d.geometry.PointCloud(),
            o3d.geometry.LineSet(),
            o3d.geometry.PointCloud()
        )

    cluster_count = int(labels.max()) + 1
    print(f"[INFO] first DBSCAN clusters: {cluster_count}")

    # =========================
    # 2. 큰 클러스터만 재분할
    # =========================
    final_object_indices_list = []
    removed_bridge_indices_all = []

    for cluster_id in range(cluster_count):
        idx = np.where(labels == cluster_id)[0]

        if len(idx) < MIN_CLUSTER_POINTS:
            print(f"[SKIP] cluster {cluster_id}: {len(idx)} points")
            continue

        cluster_pcd = pcd.select_by_index(idx)

        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent())
        volume = float(extent[0] * extent[1] * extent[2])

        print(
            f"[CHECK] cluster {cluster_id}: "
            f"points={len(idx):,}, "
            f"extent=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}), "
            f"volume={volume:.2f}"
        )

        if ENABLE_SPLIT_BIG_CLUSTER and is_big_cluster_for_split(cluster_pcd):
            print(f"  [BIG] cluster {cluster_id} 밀집도 기반 재분할 시도")

            splitted, removed_bridge_idx = split_big_cluster(
                cluster_pcd=cluster_pcd,
                original_indices=idx,
                base_eps=used_eps
            )

            for sub_idx in splitted:
                final_object_indices_list.append(sub_idx)

            if len(removed_bridge_idx) > 0:
                removed_bridge_indices_all.extend(removed_bridge_idx.tolist())

        else:
            final_object_indices_list.append(idx)

    if len(final_object_indices_list) == 0:
        print("[WARN] 최종 객체 후보가 없음.")
        return (
            o3d.geometry.PointCloud(),
            o3d.geometry.PointCloud(),
            o3d.geometry.LineSet(),
            o3d.geometry.PointCloud()
        )

    # =========================
    # 3. 최종 bbox 필터
    # =========================
    keep_indices_all = []
    final_labels_for_points = []
    bbox_linesets = []

    final_object_id = 0

    for obj_idx in final_object_indices_list:
        if len(obj_idx) < MIN_CLUSTER_POINTS:
            print(f"[FINAL SKIP] object candidate: {len(obj_idx)} points")
            continue

        obj_pcd = pcd.select_by_index(obj_idx)
        bbox = make_bbox_from_cluster(obj_pcd)

        if not is_good_bbox(bbox):
            extent = np.asarray(bbox.get_extent())
            print(
                f"[FINAL SKIP] bad bbox: "
                f"points={len(obj_idx)}, "
                f"extent=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})"
            )
            continue

        bbox_linesets.append(bbox_to_lineset(bbox, color=(1, 0, 0)))

        keep_indices_all.extend(obj_idx.tolist())
        final_labels_for_points.extend([final_object_id] * len(obj_idx))

        extent = np.asarray(bbox.get_extent())

        print(
            f"[FINAL KEEP] object {final_object_id}: "
            f"points={len(obj_idx):,}, "
            f"bbox=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})"
        )

        final_object_id += 1

    if len(keep_indices_all) == 0:
        print("[WARN] 최종 필터를 통과한 객체가 없음.")
        return (
            o3d.geometry.PointCloud(),
            o3d.geometry.PointCloud(),
            o3d.geometry.LineSet(),
            o3d.geometry.PointCloud()
        )

    keep_indices_all = np.array(keep_indices_all, dtype=np.int64)
    final_labels_for_points = np.array(final_labels_for_points, dtype=np.int32)

    # 원본 색 유지 버전
    detected_object_points_original_color = pcd.select_by_index(
        keep_indices_all.tolist()
    )

    if not detected_object_points_original_color.has_colors():
        detected_object_points_original_color.paint_uniform_color([0.6, 0.6, 0.6])

    # 클러스터 색 버전
    detected_object_points_cluster_color = colorize_by_labels(
        detected_object_points_original_color,
        final_labels_for_points
    )

    bbox_lines = merge_linesets(bbox_linesets)

    # 밀집도 낮아서 제거된 연결부 점
    if len(removed_bridge_indices_all) > 0:
        removed_bridge_pcd = pcd.select_by_index(removed_bridge_indices_all)
        removed_bridge_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    else:
        removed_bridge_pcd = o3d.geometry.PointCloud()

    print(f"[INFO] final objects: {final_object_id}")
    print(f"[INFO] removed low-density bridge points: {len(removed_bridge_pcd.points):,}")

    return (
        detected_object_points_original_color,
        detected_object_points_cluster_color,
        bbox_lines,
        removed_bridge_pcd
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
        bbox_lines,
        removed_bridge_pcd
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

    if len(removed_bridge_pcd.points) > 0:
        o3d.io.write_point_cloud(
            resolve_path(OUT_LOW_DENSITY_REMOVED_POINTS),
            removed_bridge_pcd,
            write_ascii=True
        )
        print(f"[SAVE] {OUT_LOW_DENSITY_REMOVED_POINTS}")
    else:
        print("[INFO] 제거된 low-density bridge point가 없어서 07 파일은 저장하지 않음.")

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

        if SHOW_LOW_DENSITY_REMOVED and len(removed_bridge_pcd.points) > 0:
            print("\n[VIEW 3] 제거된 low-density 연결부 점")
            base_vis = o3d.geometry.PointCloud(detected_object_points_original_color)
            bridge_vis = o3d.geometry.PointCloud(removed_bridge_pcd)

            base_vis.paint_uniform_color([0.6, 0.6, 0.6])
            bridge_vis.paint_uniform_color([1.0, 0.0, 0.0])

            print("회색: 최종 객체 점")
            print("빨간색: 밀집도 낮아서 끊어낸 연결부 점")

            o3d.visualization.draw_geometries(
                [base_vis, bridge_vis]
            )


if __name__ == "__main__":
    main()