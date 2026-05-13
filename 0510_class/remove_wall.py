# 1

import os
import numpy as np
import open3d as o3d
import pickle

# =========================
# 경로 설정
# =========================
ROOM_RAW_PLY = "class.ply"  # 원본 방 point cloud
DETECT_DIR = "room_detected_results"  # full_room_structure.ply, walls_data.pkl 있는 폴더

STRUCT_PLY = "full_room_structure.ply"
PKL_PATH   = "walls_data.pkl"

# =========================
# 파라미터
# =========================
VOXEL_SIZE = 0.02
REMOVE_R   = 0.10   # 구조물 제거 반경
STRUCT_DOWNSAMPLE = True

# =========================
# 자투리 점 제거 파라미터
# =========================
REMOVE_SMALL_FRAGMENTS = True

# 반경 0.08m 안에 이웃 점이 8개보다 적으면 제거
RADIUS_NB_POINTS = 150
RADIUS = 0.10

# =========================
# 저장 파일 이름
# =========================
OUT_UNALIGNED_STRUCT = os.path.join(
    "ply",
    "01_structure_unaligned_to_room.ply"
)

OUT_OBJECTS_ONLY = os.path.join(
    "ply",
    "02_room_objects_after_structure_removal.ply"
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
    # source 각 점 -> target 최근접 거리
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

    cleaned_pcd, ind = pcd.remove_radius_outlier(
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

    # 1) raw room -> voxel(0.02)
    room_vox = room_raw.voxel_down_sample(VOXEL_SIZE)
    print(f"[INFO] room_vox({VOXEL_SIZE}) points: {len(room_vox.points):,}")

    # 2) alignment 로드 + 구조물 역변환
    centroid, R = load_alignment(pkl_path)

    struct_pts = np.asarray(struct_aligned.points)
    struct_pts_unaligned = unalign_points(struct_pts, centroid, R)

    struct_unaligned = o3d.geometry.PointCloud()
    struct_unaligned.points = o3d.utility.Vector3dVector(struct_pts_unaligned)

    # 색이 있으면 유지
    if struct_aligned.has_colors():
        struct_unaligned.colors = struct_aligned.colors

    # 구조물도 같은 voxel로 다운샘플
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

    # 4) 제거: room_vox에서 구조물에 가까운 점들 제거
    d_room = np.asarray(room_vox.compute_point_cloud_distance(struct_unaligned))

    keep_idx = np.where(d_room > REMOVE_R)[0]
    removed = len(d_room) - len(keep_idx)

    objects_only = room_vox.select_by_index(keep_idx)

    print(f"\n[INFO] REMOVE_R={REMOVE_R:.3f}m")
    print(f"[INFO] removed structure points: {removed:,}")
    print(f"[INFO] kept before cleanup: {len(objects_only.points):,}")

    # 5) 구조물 제거 후 자투리 점 제거
    if REMOVE_SMALL_FRAGMENTS:
        objects_only = remove_small_fragments(objects_only)

    # 6) 저장
    os.makedirs(resolve_path("ply"), exist_ok=True)

    o3d.io.write_point_cloud(
        resolve_path(OUT_UNALIGNED_STRUCT),
        struct_unaligned,
        write_ascii=True
    )

    o3d.io.write_point_cloud(
        resolve_path(OUT_OBJECTS_ONLY),
        objects_only,
        write_ascii=True
    )

    print(f"\n[SAVE] {OUT_UNALIGNED_STRUCT}")
    print(f"[SAVE] {OUT_OBJECTS_ONLY}")

    # 7) 간단 시각화: 사물만
    o3d.visualization.draw_geometries([objects_only])


if __name__ == "__main__":
    main()