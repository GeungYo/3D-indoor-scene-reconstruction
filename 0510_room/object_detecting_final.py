# 2

import os
import numpy as np
import open3d as o3d

# ============================================================
# 입력: 자투리 점 제거가 끝난 사물 포인트클라우드
# ============================================================
INPUT_FILE = os.path.join(
    "ply",
    "03_objects_after_small_fragment_cleanup.ply"
)

# ============================================================
# 출력 파일
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
# 전처리
# 이미 voxel 0.02 상태라면 OFF 유지
# ============================================================
EXTRA_VOXEL = 0.0

# ============================================================
# DBSCAN
# ============================================================
DBSCAN_EPS_START = 0.04
DBSCAN_MIN_POINTS = 30

# 너무 한 덩어리면 eps 자동 감소
BIG_CLUSTER_FRAC = 0.65
EPS_SHRINK = 0.85
EPS_RETRY = 4
DBSCAN_EPS_MIN = 0.03

# ============================================================
# 선택적 클러스터 병합
# 현재는 "붙은 사물을 더 나누는 것"이 목적이므로 False 유지
# ============================================================
ENABLE_MERGE = False
MERGE_DIST = 0.06
MIN_POINTS_FOR_MERGE = 80

# ============================================================
# bbox 생성 / 필터
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
# 시각화
# ============================================================
SHOW = True


# ------------------------------------------------------------
# 경로 / 입출력 유틸
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
    ok = o3d.io.write_point_cloud(full_path, pcd, write_ascii=True)

    print(f"[SAVE] {path} -> {'OK' if ok else 'FAIL'}")


def save_lineset_as_ply(path, linesets):
    if len(linesets) == 0:
        print("[WARN] 저장할 bounding box가 없어 저장하지 않음.")
        return

    merged = o3d.geometry.LineSet()

    all_points = []
    all_lines = []
    all_colors = []

    point_offset = 0

    for ls in linesets:
        pts = np.asarray(ls.points)
        lns = np.asarray(ls.lines)

        if len(ls.colors) > 0:
            cols = np.asarray(ls.colors)
        else:
            cols = np.tile(
                np.array([[1.0, 0.0, 0.0]]),
                (lns.shape[0], 1)
            )

        all_points.append(pts)
        all_lines.append(lns + point_offset)
        all_colors.append(cols)

        point_offset += pts.shape[0]

    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    full_path = resolve_path(path)
    ok = o3d.io.write_line_set(full_path, merged, write_ascii=True)

    print(f"[SAVE] {path} -> {'OK' if ok else 'FAIL'}")


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

    for t in range(EPS_RETRY + 1):
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

    for i, lab in enumerate(labels):
        if lab >= 0:
            colors[i] = palette[lab]
        else:
            colors[i] = np.array([0.2, 0.2, 0.2])

    colored.colors = o3d.utility.Vector3dVector(colors)

    return colored


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

                d = aabb_distance(
                    mins[i], maxs[i],
                    mins[j], maxs[j]
                )

                if d <= merge_dist:
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

    if merged.size > 0 and merged.max() >= 0:
        offset = merged.max() + 1
    else:
        offset = 0

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
# bbox 생성 / 필터
# ------------------------------------------------------------
def create_bbox(cluster):
    if USE_OBB:
        bbox = cluster.get_oriented_bounding_box()
    else:
        bbox = cluster.get_axis_aligned_bounding_box()

    bbox.color = (1, 0, 0)

    return bbox


def bbox_pass_filter(bbox):
    ex, ey, ez = bbox.get_extent()

    # 너무 작은 조각 제거
    if ex < BOX_MIN_SIZE and ey < BOX_MIN_SIZE and ez < BOX_MIN_SIZE:
        return False

    e_sorted = sorted([float(ex), float(ey), float(ez)])
    min_e = e_sorted[0]
    mid_e = e_sorted[1]
    max_e = e_sorted[2]

    footprint = max_e * mid_e

    # 얇고 면적도 작으면 제거
    if min_e < THIN_MIN:
        if footprint < THIN_KEEP_FOOTPRINT:
            return False

    # 지나치게 길쭉하고 면적도 작으면 제거
    if (max_e / (min_e + 1e-9)) > ELONG_RATIO:
        if footprint < ELONG_REMOVE_FOOTPRINT:
            return False

    return True


def bbox_to_lineset(bbox):
    if isinstance(bbox, o3d.geometry.OrientedBoundingBox):
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    else:
        lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)

    lineset.paint_uniform_color([1, 0, 0])

    return lineset


def extract_detected_objects(pcd, labels):
    """
    bbox 필터를 통과한 클러스터만 최종 객체로 남긴다.
    """
    keep_indices = []
    keep_labels = []
    linesets = []

    valid_labels = [l for l in np.unique(labels) if l >= 0]

    for label in valid_labels:
        idx = np.where(labels == label)[0]

        if len(idx) < MIN_CLUSTER_POINTS:
            print(f"[SKIP] cluster {label}: {len(idx)} points")
            continue

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

        linesets.append(bbox_to_lineset(bbox))

        keep_indices.extend(idx.tolist())
        keep_labels.extend([label] * len(idx))

        ex, ey, ez = bbox.get_extent()

        print(
            f"[KEEP] cluster {label}: "
            f"points={len(idx):,}, "
            f"extent=({ex:.2f}, {ey:.2f}, {ez:.2f})"
        )

    if len(keep_indices) == 0:
        return (
            o3d.geometry.PointCloud(),
            o3d.geometry.PointCloud(),
            [],
            np.array([], dtype=np.int32)
        )

    keep_indices = np.array(keep_indices, dtype=np.int64)
    keep_labels = np.array(keep_labels, dtype=np.int32)

    # 원본 색 유지 버전
    detected_original_color = pcd.select_by_index(keep_indices.tolist())

    if not detected_original_color.has_colors():
        detected_original_color.paint_uniform_color([0.6, 0.6, 0.6])

    # 클러스터 색 버전
    detected_cluster_color = colorize_by_labels(
        detected_original_color,
        keep_labels
    )

    return (
        detected_original_color,
        detected_cluster_color,
        linesets,
        keep_labels
    )


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    os.makedirs(resolve_path("ply"), exist_ok=True)

    pcd = load_pcd(INPUT_FILE)

    if EXTRA_VOXEL and EXTRA_VOXEL > 0:
        pcd = pcd.voxel_down_sample(EXTRA_VOXEL)
        print(f"[INFO] after EXTRA_VOXEL({EXTRA_VOXEL}): {len(pcd.points):,}")

    # 1) DBSCAN 자동 eps 조절
    labels, used_eps = dbscan_auto(pcd)
    labels = labels.astype(np.int32)

    # 2) 선택적 클러스터 병합
    if ENABLE_MERGE and labels.size > 0 and labels.max() >= 0:
        pts = np.asarray(pcd.points)
        labels = merge_clusters_by_aabb(
            pts,
            labels,
            MERGE_DIST
        )

    if labels.size > 0 and labels.max() >= 0:
        final_cluster_count = int(labels.max()) + 1
    else:
        final_cluster_count = 0

    print(f"[INFO] final clusters before bbox filter: {final_cluster_count}")

    # 3) bbox 필터 통과한 객체만 추출
    (
        detected_original_color,
        detected_cluster_color,
        bbox_linesets,
        kept_labels
    ) = extract_detected_objects(
        pcd,
        labels
    )

    print(f"[INFO] final detected boxes: {len(bbox_linesets)}")

    # 4) 저장
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
            print("\n[VIEW 1] 원본 색 사물 포인트 + bbox")
            geoms_original = [detected_original_color]
            geoms_original.extend(bbox_linesets)
            o3d.visualization.draw_geometries(geoms_original)

        if len(detected_cluster_color.points) > 0:
            print("\n[VIEW 2] 클러스터별 색 사물 포인트 + bbox")
            geoms_cluster = [detected_cluster_color]
            geoms_cluster.extend(bbox_linesets)
            o3d.visualization.draw_geometries(geoms_cluster)


if __name__ == "__main__":
    main()