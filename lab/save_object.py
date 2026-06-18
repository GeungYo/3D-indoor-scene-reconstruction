# 3
# 최종 객체 클러스터를 개별 PLY 파일로 저장하는 코드
# + 원래 좌표 x, y, z를 명시적으로 CSV/JSON에도 같이 저장

import os
import json
import numpy as np
import open3d as o3d

# ============================================================
# 입력: 자투리 점 제거까지 끝난 객체 후보 point cloud
# ============================================================
INPUT_FILE = os.path.join(
    "ply",
    "03_objects_after_small_fragment_cleanup.ply"
)

# ============================================================
# 저장 폴더
# ============================================================
OUT_OBJECT_DIR = os.path.join(
    "ply",
    "detected_objects"
)

# 기존 object_001.ply 등이 있으면 지울지 여부
# True로 두면 실행할 때마다 새 결과만 깔끔하게 남음
CLEAR_OLD_OBJECT_FILES = True

# ============================================================
# 추가 저장 옵션
# ============================================================
SAVE_META_JSON = True
SAVE_ORIGINAL_XYZ_CSV = True

# ============================================================
# DBSCAN
# 현재 객체 분리 코드와 동일한 기준
# ============================================================
DBSCAN_EPS_START = 0.04
DBSCAN_MIN_POINTS = 30

BIG_CLUSTER_FRAC = 0.65
EPS_SHRINK = 0.85
EPS_RETRY = 4
DBSCAN_EPS_MIN = 0.03

# ============================================================
# 선택적 클러스터 병합
# 현재는 False 유지
# ============================================================
ENABLE_MERGE = False
MERGE_DIST = 0.06
MIN_POINTS_FOR_MERGE = 80

# ============================================================
# bbox 필터
# 현재 객체 분리 코드와 동일한 기준
# ============================================================
MIN_CLUSTER_POINTS = 150

USE_OBB = False

BOX_MIN_SIZE = 0.08

THIN_MIN = 0.02
THIN_KEEP_FOOTPRINT = 0.30

ELONG_RATIO = 12.0
ELONG_REMOVE_FOOTPRINT = 0.15

# ============================================================
# 저장 시 객체별 시각화 여부
# False 추천
# ============================================================
SHOW_EACH_OBJECT = False


# ------------------------------------------------------------
# 경로 유틸
# ------------------------------------------------------------
def here_dir():
    return os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(here_dir(), path)


# ------------------------------------------------------------
# 입력 로드
# ------------------------------------------------------------
def load_pcd(path):
    full_path = resolve_path(path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"입력 파일이 없음: {full_path}")

    pcd = o3d.io.read_point_cloud(full_path)

    if len(pcd.points) == 0:
        raise RuntimeError("입력 point cloud가 비어있어.")

    print(f"[INFO] input: {path}")
    print(f"[INFO] full path: {full_path}")
    print(f"[INFO] points: {len(pcd.points):,}")
    print(f"[INFO] has colors: {pcd.has_colors()}")

    return pcd


# ------------------------------------------------------------
# 기존 object_### 파일 정리
# ------------------------------------------------------------
def clear_old_object_files():
    out_dir = resolve_path(OUT_OBJECT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    if not CLEAR_OLD_OBJECT_FILES:
        return

    removed_count = 0

    for filename in os.listdir(out_dir):
        if filename.startswith("object_") and (
            filename.endswith(".ply")
            or filename.endswith(".json")
            or filename.endswith(".csv")
        ):
            file_path = os.path.join(out_dir, filename)
            os.remove(file_path)
            removed_count += 1

    print(f"[INFO] old object files removed: {removed_count}")


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

        frac = largest_cluster_frac(labels, len(pcd.points))

        if labels.size > 0 and labels.max() >= 0:
            cluster_count = int(labels.max()) + 1
        else:
            cluster_count = 0

        print(
            f"[INFO] DBSCAN try{t}: "
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
# 선택적 클러스터 병합
# ------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def aabb_distance(min1, max1, min2, max2):
    dx = max(0.0, max(min2[0] - max1[0], min1[0] - max2[0]))
    dy = max(0.0, max(min2[1] - max1[1], min1[1] - max2[1]))
    dz = max(0.0, max(min2[2] - max1[2], min1[2] - max2[2]))

    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def merge_clusters_by_aabb(points, labels, merge_dist):
    unique_labels = [l for l in np.unique(labels) if l >= 0]

    if len(unique_labels) == 0:
        return labels

    idxs_by_label = {
        label: np.where(labels == label)[0]
        for label in unique_labels
    }

    labels_big = [
        label for label in unique_labels
        if idxs_by_label[label].size >= MIN_POINTS_FOR_MERGE
    ]

    if len(labels_big) < 2:
        return labels

    mins = np.zeros((len(labels_big), 3), dtype=np.float64)
    maxs = np.zeros((len(labels_big), 3), dtype=np.float64)
    centers = np.zeros((len(labels_big), 3), dtype=np.float64)

    for i, label in enumerate(labels_big):
        pts = points[idxs_by_label[label]]
        mins[i] = pts.min(axis=0)
        maxs[i] = pts.max(axis=0)
        centers[i] = (mins[i] + maxs[i]) * 0.5

    uf = UnionFind(len(labels_big))

    cell = merge_dist * 2.0
    grid = {}

    def cell_key(center):
        return tuple(np.floor(center / cell).astype(int))

    for i, center in enumerate(centers):
        key = cell_key(center)
        grid.setdefault(key, []).append(i)

    neighbors = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]

    for key, items in grid.items():
        candidates = []

        for dd in neighbors:
            neighbor_key = (
                key[0] + dd[0],
                key[1] + dd[1],
                key[2] + dd[2]
            )
            candidates.extend(grid.get(neighbor_key, []))

        candidates = sorted(set(candidates))
        items = sorted(items)

        for i in items:
            for j in candidates:
                if j <= i:
                    continue

                dist = aabb_distance(
                    mins[i], maxs[i],
                    mins[j], maxs[j]
                )

                if dist <= merge_dist:
                    uf.union(i, j)

    root_to_new = {}
    next_label = 0
    big_label_to_new = {}

    for i, label in enumerate(labels_big):
        root = uf.find(i)

        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1

        big_label_to_new[label] = root_to_new[root]

    merged = labels.copy()

    offset = merged.max() + 1 if merged.size and merged.max() >= 0 else 0

    for label in labels_big:
        merged[labels == label] = offset + big_label_to_new[label]

    unique_after = [l for l in np.unique(merged) if l >= 0]
    remap = {old: new for new, old in enumerate(sorted(unique_after))}

    merged2 = merged.copy()

    for old, new in remap.items():
        merged2[merged == old] = new

    print(
        f"[INFO] merge: big clusters {len(labels_big)} "
        f"-> merged clusters {len(remap)}"
    )

    return merged2


# ------------------------------------------------------------
# bbox 필터
# ------------------------------------------------------------
def create_bbox(cluster):
    if USE_OBB:
        bbox = cluster.get_oriented_bounding_box()
    else:
        bbox = cluster.get_axis_aligned_bounding_box()

    return bbox


def bbox_pass_filter(bbox):
    ex, ey, ez = bbox.get_extent()

    # 너무 작은 조각 제거
    if ex < BOX_MIN_SIZE and ey < BOX_MIN_SIZE and ez < BOX_MIN_SIZE:
        return False

    extents = sorted([float(ex), float(ey), float(ez)])
    min_e, mid_e, max_e = extents

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


# ------------------------------------------------------------
# meta json 저장
# ------------------------------------------------------------
def save_object_meta_json(
    meta_path,
    base_name,
    label,
    points,
    bbox,
    has_colors
):
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center_mean = points.mean(axis=0)
    extent = max_bound - min_bound

    bbox_center = bbox.get_center()
    bbox_extent = bbox.get_extent()

    meta = {
        "object_name": base_name,
        "object_file": f"{base_name}.ply",
        "source_cluster_label": int(label),
        "point_count": int(len(points)),
        "has_colors": bool(has_colors),

        "coordinate_system": "same_as_input_point_cloud",
        "note": "All saved x, y, z coordinates are kept in the original input point cloud coordinate system. No centering or normalization is applied.",

        "point_min_bound": {
            "x": float(min_bound[0]),
            "y": float(min_bound[1]),
            "z": float(min_bound[2]),
        },
        "point_max_bound": {
            "x": float(max_bound[0]),
            "y": float(max_bound[1]),
            "z": float(max_bound[2]),
        },
        "point_mean_center": {
            "x": float(center_mean[0]),
            "y": float(center_mean[1]),
            "z": float(center_mean[2]),
        },
        "point_extent": {
            "x": float(extent[0]),
            "y": float(extent[1]),
            "z": float(extent[2]),
        },

        "bbox_center": {
            "x": float(bbox_center[0]),
            "y": float(bbox_center[1]),
            "z": float(bbox_center[2]),
        },
        "bbox_extent": {
            "x": float(bbox_extent[0]),
            "y": float(bbox_extent[1]),
            "z": float(bbox_extent[2]),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


# ------------------------------------------------------------
# xyz csv 저장
# ------------------------------------------------------------
def save_original_xyz_csv(csv_path, points):
    np.savetxt(
        csv_path,
        points,
        delimiter=",",
        header="x,y,z",
        comments="",
        fmt="%.8f"
    )


# ------------------------------------------------------------
# 객체별 PLY 저장
# ------------------------------------------------------------
def save_each_object_ply(pcd, labels):
    out_dir = resolve_path(OUT_OBJECT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    valid_labels = [l for l in np.unique(labels) if l >= 0]

    saved_count = 0
    input_has_colors = pcd.has_colors()

    for label in valid_labels:
        idx = np.where(labels == label)[0]

        if len(idx) < MIN_CLUSTER_POINTS:
            print(f"[SKIP] cluster {label}: {len(idx)} points")
            continue

        # 중요:
        # select_by_index는 원본 pcd에서 해당 점만 골라오는 것.
        # 여기서 좌표를 0,0,0 기준으로 옮기지 않음.
        # 즉 cluster.points는 원래 point cloud 좌표계 그대로임.
        cluster = pcd.select_by_index(idx)

        if len(cluster.points) == 0:
            continue

        bbox = create_bbox(cluster)

        if not bbox_pass_filter(bbox):
            ex, ey, ez = bbox.get_extent()
            print(
                f"[SKIP] cluster {label}: bbox filter fail, "
                f"points={len(idx)}, "
                f"extent=({ex:.2f}, {ey:.2f}, {ez:.2f})"
            )
            continue

        saved_count += 1

        base_name = f"object_{saved_count:03d}"

        ply_path = os.path.join(out_dir, f"{base_name}.ply")
        meta_path = os.path.join(out_dir, f"{base_name}_meta.json")
        xyz_path = os.path.join(out_dir, f"{base_name}_original_xyz.csv")

        points = np.asarray(cluster.points)

        # 1) 객체 PLY 저장
        # 이 PLY 안에도 x, y, z가 원래 좌표계 기준으로 저장됨.
        # cluster에 color가 있으면 color도 같이 저장됨.
        ok = o3d.io.write_point_cloud(
            ply_path,
            cluster,
            write_ascii=True
        )

        # 2) 원래 x, y, z 좌표 CSV로 별도 저장
        if SAVE_ORIGINAL_XYZ_CSV:
            save_original_xyz_csv(
                xyz_path,
                points
            )

        # 3) 객체 위치 정보 JSON으로 저장
        if SAVE_META_JSON:
            save_object_meta_json(
                meta_path=meta_path,
                base_name=base_name,
                label=label,
                points=points,
                bbox=bbox,
                has_colors=cluster.has_colors()
            )

        ex, ey, ez = bbox.get_extent()
        center = bbox.get_center()

        print(
            f"[SAVE] {base_name}.ply -> {'OK' if ok else 'FAIL'} | "
            f"points={len(cluster.points):,}, "
            f"extent=({ex:.2f}, {ey:.2f}, {ez:.2f}), "
            f"bbox_center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), "
            f"has_colors={cluster.has_colors()}"
        )

        if SAVE_META_JSON:
            print(f"       meta saved: {base_name}_meta.json")

        if SAVE_ORIGINAL_XYZ_CSV:
            print(f"       xyz saved : {base_name}_original_xyz.csv")

        if SHOW_EACH_OBJECT:
            o3d.visualization.draw_geometries([cluster])

    print(f"\n[DONE] saved object ply files: {saved_count}")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    clear_old_object_files()

    pcd = load_pcd(INPUT_FILE)

    # 1) DBSCAN 자동 eps 조절
    labels, used_eps = dbscan_auto(pcd)
    labels = labels.astype(np.int32)

    # 2) 선택적 병합
    if ENABLE_MERGE and labels.size > 0 and labels.max() >= 0:
        pts = np.asarray(pcd.points)
        labels = merge_clusters_by_aabb(
            pts,
            labels,
            MERGE_DIST
        )

    # 3) 개별 PLY 저장
    save_each_object_ply(
        pcd,
        labels
    )


if __name__ == "__main__":
    main()