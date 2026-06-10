# 1

import os
import numpy as np
import open3d as o3d
import pickle

# =========================
# 경로 설정
# =========================
ROOM_RAW_PLY = "lab.ply"
DETECT_DIR = "lab_detected_results"

STRUCT_PLY = "full_room_structure.ply"
PKL_PATH   = "walls_data.pkl"

# =========================
# 기본 파라미터
# =========================
VOXEL_SIZE = 0.02
REMOVE_R   = 0.08
STRUCT_DOWNSAMPLE = True

# =========================
# 자투리 점 제거 파라미터
# =========================
REMOVE_SMALL_FRAGMENTS = True

RADIUS_NB_POINTS = 200
RADIUS = 0.10

# =========================
# 공중에 떠 있는 작은 클러스터 제거 파라미터
# =========================
REMOVE_SMALL_FLOATING_CLUSTERS = True

# 높이축: y축
HEIGHT_AXIS = 1  # x=0, y=1, z=2

# 바닥이 y축의 작은 쪽이면 "min"
# 만약 반대로 판정되면 "max"로 변경
FLOOR_SIDE = "min"

# 바닥 높이 추정용 percentile
FLOOR_PERCENTILE = 1.0

# 공중 클러스터를 찾기 위한 임시 DBSCAN
FLOAT_DBSCAN_EPS = 0.10
FLOAT_DBSCAN_MIN_POINTS = 20

# 제거 조건
# 1) 바닥에서 이 거리 이상 떠 있고
FLOAT_MIN_GAP_FROM_FLOOR = 0.25

# 2) 점 수가 이 이하이고
FLOAT_MAX_POINTS = 1200

# 3) 클러스터의 가장 긴 길이가 이 이하이면 제거
FLOAT_MAX_EXTENT = 0.45

# =========================
# 저장 파일 이름
# =========================
OUT_UNALIGNED_STRUCT = os.path.join(
    "ply",
    "01_structure_unaligned_to_room.ply"
)

OUT_OBJECTS_BEFORE_CLEANUP = os.path.join(
    "ply",
    "02_objects_before_small_fragment_cleanup.ply"
)

# 중요:
# 자투리 점 제거 + 공중 작은 클러스터 제거까지 끝난 최종 결과를
# 기존 파일명 그대로 덮어씀
OUT_OBJECTS_AFTER_CLEANUP = os.path.join(
    "ply",
    "03_objects_after_small_fragment_cleanup.ply"
)


def here_dir():
    return os.path.dirname(os.path.abspath(__file__))


def resolve_path(p):
    return p if os.path.isabs(p) else os.path.join(here_dir(), p)


def load_alignment(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    align = data.get("alignment", None)

    if align is None:
        raise RuntimeError("walls_data.pkl에 alignment 정보가 없어.")

    c = np.array(align["centroid"], dtype=np.float64)
    R = np.array(align["R"], dtype=np.float64)

    return c, R


def unalign_points(points_aligned, centroid, R):
    # p_orig = p_aligned @ R + centroid
    return points_aligned @ R + centroid


def match_rate(source_pcd, target_pcd, r_list=(0.02, 0.03, 0.04, 0.05)):
    d = np.asarray(source_pcd.compute_point_cloud_distance(target_pcd))

    out = {}
    out["mean"] = float(d.mean()) if len(d) else 0.0
    out["median"] = float(np.median(d)) if len(d) else 0.0
    out["p90"] = float(np.percentile(d, 90)) if len(d) else 0.0
    out["p95"] = float(np.percentile(d, 95)) if len(d) else 0.0
    out["rates"] = {R: float(np.mean(d <= R)) for R in r_list} if len(d) else {}

    return out


def remove_small_fragments(pcd):
    if len(pcd.points) == 0:
        return pcd

    cleaned_pcd, _ = pcd.remove_radius_outlier(
        nb_points=RADIUS_NB_POINTS,
        radius=RADIUS
    )

    removed = len(pcd.points) - len(cleaned_pcd.points)

    print("\n=== SMALL FRAGMENT CLEANUP ===")
    print(f"[INFO] radius: {RADIUS}")
    print(f"[INFO] nb_points: {RADIUS_NB_POINTS}")
    print(f"[INFO] before cleanup: {len(pcd.points):,}")
    print(f"[INFO] removed small fragments: {removed:,}")
    print(f"[INFO] after cleanup: {len(cleaned_pcd.points):,}")

    return cleaned_pcd


def estimate_floor_height(room_pcd):
    points = np.asarray(room_pcd.points)

    if len(points) == 0:
        raise RuntimeError("floor height 추정용 point cloud가 비어있어.")

    h = points[:, HEIGHT_AXIS]

    if FLOOR_SIDE == "min":
        floor_h = float(np.percentile(h, FLOOR_PERCENTILE))
    elif FLOOR_SIDE == "max":
        floor_h = float(np.percentile(h, 100.0 - FLOOR_PERCENTILE))
    else:
        raise ValueError('FLOOR_SIDE는 "min" 또는 "max"여야 해.')

    print("\n=== FLOOR HEIGHT ESTIMATION ===")
    print(f"[INFO] HEIGHT_AXIS: {HEIGHT_AXIS}")
    print(f"[INFO] FLOOR_SIDE: {FLOOR_SIDE}")
    print(f"[INFO] estimated floor height: {floor_h:.4f}")

    return floor_h


def remove_small_floating_clusters(pcd, floor_height):
    if len(pcd.points) == 0:
        return pcd

    labels = np.array(
        pcd.cluster_dbscan(
            eps=FLOAT_DBSCAN_EPS,
            min_points=FLOAT_DBSCAN_MIN_POINTS,
            print_progress=True
        )
    )

    if len(labels) == 0 or labels.max() < 0:
        print("\n=== FLOATING CLUSTER CLEANUP ===")
        print("[INFO] 유효한 클러스터가 없어 공중 부유 클러스터 제거 생략.")
        return pcd

    max_label = int(labels.max())

    keep_indices = []
    removed_indices = []

    print("\n=== FLOATING CLUSTER CLEANUP ===")
    print(f"[INFO] FLOAT_DBSCAN_EPS: {FLOAT_DBSCAN_EPS}")
    print(f"[INFO] FLOAT_DBSCAN_MIN_POINTS: {FLOAT_DBSCAN_MIN_POINTS}")
    print(f"[INFO] clusters: {max_label + 1}")

    for cluster_id in range(max_label + 1):
        idx = np.where(labels == cluster_id)[0]

        if len(idx) == 0:
            continue

        cluster = pcd.select_by_index(idx)
        bbox = cluster.get_axis_aligned_bounding_box()

        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        extent = np.asarray(bbox.get_extent())

        if FLOOR_SIDE == "min":
            cluster_bottom = float(min_bound[HEIGHT_AXIS])
            gap_from_floor = cluster_bottom - floor_height
        else:
            cluster_bottom = float(max_bound[HEIGHT_AXIS])
            gap_from_floor = floor_height - cluster_bottom

        max_extent = float(np.max(extent))
        point_count = len(idx)

        is_floating = gap_from_floor >= FLOAT_MIN_GAP_FROM_FLOOR
        is_small_by_points = point_count <= FLOAT_MAX_POINTS
        is_small_by_size = max_extent <= FLOAT_MAX_EXTENT

        should_remove = (
            is_floating and
            is_small_by_points and
            is_small_by_size
        )

        if should_remove:
            removed_indices.extend(idx.tolist())
            status = "REMOVE"
        else:
            keep_indices.extend(idx.tolist())
            status = "KEEP"

        print(
            f"[{status}] cluster {cluster_id}: "
            f"points={point_count:,}, "
            f"gap_from_floor={gap_from_floor:.3f}, "
            f"max_extent={max_extent:.3f}"
        )

    # DBSCAN noise(-1) 점은 일단 유지
    noise_idx = np.where(labels == -1)[0]

    if len(noise_idx) > 0:
        keep_indices.extend(noise_idx.tolist())
        print(f"[KEEP] DBSCAN noise points: {len(noise_idx):,}")

    keep_indices = sorted(list(set(keep_indices)))
    removed_indices = sorted(list(set(removed_indices)))

    cleaned_pcd = pcd.select_by_index(keep_indices)

    print(f"[INFO] before floating cleanup: {len(pcd.points):,}")
    print(f"[INFO] removed floating small clusters: {len(removed_indices):,}")
    print(f"[INFO] after floating cleanup: {len(cleaned_pcd.points):,}")

    return cleaned_pcd


def main():
    room_path = resolve_path(ROOM_RAW_PLY)
    struct_path = resolve_path(STRUCT_PLY)
    pkl_path = resolve_path(PKL_PATH)

    room_raw = o3d.io.read_point_cloud(room_path)
    struct_aligned = o3d.io.read_point_cloud(struct_path)

    if len(room_raw.points) == 0:
        raise RuntimeError("room_raw가 비어있어.")

    if len(struct_aligned.points) == 0:
        raise RuntimeError("full_room_structure.ply가 비어있어.")

    print(f"[INFO] room_raw points: {len(room_raw.points):,}")
    print(f"[INFO] struct_aligned points: {len(struct_aligned.points):,}")

    # 1) raw room -> voxel
    room_vox = room_raw.voxel_down_sample(VOXEL_SIZE)
    print(f"[INFO] room_vox({VOXEL_SIZE}) points: {len(room_vox.points):,}")

    # 2) alignment 로드 + 구조물 역변환
    centroid, R = load_alignment(pkl_path)

    struct_pts = np.asarray(struct_aligned.points)
    struct_pts_unaligned = unalign_points(struct_pts, centroid, R)

    struct_unaligned = o3d.geometry.PointCloud()
    struct_unaligned.points = o3d.utility.Vector3dVector(struct_pts_unaligned)

    if struct_aligned.has_colors():
        struct_unaligned.colors = struct_aligned.colors

    if STRUCT_DOWNSAMPLE:
        struct_unaligned = struct_unaligned.voxel_down_sample(VOXEL_SIZE)

    print(f"[INFO] struct_unaligned points: {len(struct_unaligned.points):,}")

    # 3) 역변환이 제대로 됐는지 매칭률 체크
    stats = match_rate(
        struct_unaligned,
        room_vox,
        r_list=(0.02, 0.03, 0.04, 0.05, 0.07)
    )

    print("\n=== MATCH (struct_unaligned -> room_vox) ===")
    print("mean  :", stats["mean"])
    print("median:", stats["median"])
    print("p90   :", stats["p90"])
    print("p95   :", stats["p95"])

    for rr, v in stats["rates"].items():
        print(f"R={rr:.3f}m: {v * 100:.2f}%")

    # 4) room_vox에서 구조물에 가까운 점 제거
    d_room = np.asarray(room_vox.compute_point_cloud_distance(struct_unaligned))

    keep_idx = np.where(d_room > REMOVE_R)[0]
    removed = len(d_room) - len(keep_idx)

    objects_before_cleanup = room_vox.select_by_index(keep_idx)

    print(f"\n[INFO] REMOVE_R={REMOVE_R:.3f}m")
    print(f"[INFO] removed structure points: {removed:,}")
    print(f"[INFO] kept before cleanup: {len(objects_before_cleanup.points):,}")

    # 5) 자투리 점 제거
    if REMOVE_SMALL_FRAGMENTS:
        objects_after_small_fragment_cleanup = remove_small_fragments(
            objects_before_cleanup
        )
    else:
        objects_after_small_fragment_cleanup = objects_before_cleanup

    # 6) 바닥 높이 추정
    floor_height = estimate_floor_height(room_vox)

    # 7) 공중에 떠 있는 작은 클러스터 제거
    if REMOVE_SMALL_FLOATING_CLUSTERS:
        final_objects = remove_small_floating_clusters(
            objects_after_small_fragment_cleanup,
            floor_height
        )
    else:
        final_objects = objects_after_small_fragment_cleanup

    # 8) 저장
    os.makedirs(resolve_path("ply"), exist_ok=True)

    o3d.io.write_point_cloud(
        resolve_path(OUT_UNALIGNED_STRUCT),
        struct_unaligned,
        write_ascii=True
    )

    o3d.io.write_point_cloud(
        resolve_path(OUT_OBJECTS_BEFORE_CLEANUP),
        objects_before_cleanup,
        write_ascii=True
    )

    # 기존 03 파일명에 최종 결과 덮어쓰기
    o3d.io.write_point_cloud(
        resolve_path(OUT_OBJECTS_AFTER_CLEANUP),
        final_objects,
        write_ascii=True
    )

    print(f"\n[SAVE] {OUT_UNALIGNED_STRUCT}")
    print(f"[SAVE] {OUT_OBJECTS_BEFORE_CLEANUP}")
    print(f"[SAVE] {OUT_OBJECTS_AFTER_CLEANUP}  ← 최종 결과 덮어쓰기")

    # 9) 최종 결과만 시각화
    print("\n[VIEW] 최종 객체 후보 point cloud")
    o3d.visualization.draw_geometries([final_objects])


if __name__ == "__main__":
    main()