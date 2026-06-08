import pickle
from pathlib import Path

import numpy as np
import open3d as o3d


# =========================
# 경로 설정
# =========================
# 이 파이썬 파일(merge.py)이 있는 폴더를 기준으로 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent

PLY_DIR = SCRIPT_DIR / "ply"

PKL_PATH = SCRIPT_DIR / "walls_data.pkl"
ROOM_MESH_PATH = SCRIPT_DIR / "textured_room_vertexcolor.ply"

# 중요:
# 이 파일은 "bounding box 안에 있는 점만 남긴 결과 파일"이어야 함
FILTERED_OBJECT_PCD_PATH = (
    PLY_DIR / "04_detected_object_points_original_color.ply"
)

BBOX_PATH = (
    PLY_DIR / "06_detected_object_bounding_boxes.ply"
)


def load_alignment(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    align = data.get("alignment", None)
    if align is None:
        raise RuntimeError("walls_data.pkl에 alignment 정보가 없어.")

    centroid = np.array(align["centroid"], dtype=np.float64)
    R = np.array(align["R"], dtype=np.float64)
    return centroid, R


def align_points(points_raw, centroid, R):
    return (points_raw - centroid) @ R.T


def align_pointcloud(pcd, centroid, R):
    pts = np.asarray(pcd.points)
    pts_aligned = align_points(pts, centroid, R)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts_aligned)

    if pcd.has_colors():
        out.colors = pcd.colors

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        normals_aligned = normals @ R.T
        out.normals = o3d.utility.Vector3dVector(normals_aligned)

    return out


def align_lineset(ls, centroid, R):
    pts = np.asarray(ls.points)
    pts_aligned = align_points(pts, centroid, R)

    out = o3d.geometry.LineSet()
    out.points = o3d.utility.Vector3dVector(pts_aligned)
    out.lines = ls.lines

    if len(ls.colors) > 0:
        out.colors = ls.colors

    return out


def print_bounds(name, geom):
    if isinstance(geom, o3d.geometry.PointCloud):
        pts = np.asarray(geom.points)
    elif isinstance(geom, o3d.geometry.TriangleMesh):
        pts = np.asarray(geom.vertices)
    elif isinstance(geom, o3d.geometry.LineSet):
        pts = np.asarray(geom.points)
    else:
        return

    if len(pts) == 0:
        print(f"[INFO] {name}: empty")
        return

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = pts.mean(axis=0)

    print(f"\n[INFO] {name}")
    print(f"       min   = {mn}")
    print(f"       max   = {mx}")
    print(f"       center= {center}")


def show_with_big_points(geoms, point_size=6.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Room mesh + filtered aligned objects + aligned bboxes",
        width=1280,
        height=800
    )

    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.mesh_show_back_face = True

    vis.run()
    vis.destroy_window()


def main():
    if not PKL_PATH.exists():
        raise FileNotFoundError(f"PKL 없음: {PKL_PATH}")

    if not ROOM_MESH_PATH.exists():
        raise FileNotFoundError(f"room mesh 없음: {ROOM_MESH_PATH}")

    if not FILTERED_OBJECT_PCD_PATH.exists():
        raise FileNotFoundError(
            f"filtered object pcd 없음: {FILTERED_OBJECT_PCD_PATH}\n"
            f"먼저 bounding box 내부 점만 남긴 result_points 파일이 있어야 해."
        )

    centroid, R = load_alignment(PKL_PATH)

    room_mesh = o3d.io.read_triangle_mesh(
        str(ROOM_MESH_PATH)
    )

    filtered_object_pcd_raw = o3d.io.read_point_cloud(
        str(FILTERED_OBJECT_PCD_PATH)
    )

    filtered_object_pcd_aligned = align_pointcloud(
        filtered_object_pcd_raw,
        centroid,
        R
    )

    geoms = [
        room_mesh,
        filtered_object_pcd_aligned
    ]

    print_bounds("room_mesh", room_mesh)
    print_bounds(
        "filtered_object_pcd_aligned",
        filtered_object_pcd_aligned
    )

    if BBOX_PATH.exists():
        bbox_ls_raw = o3d.io.read_line_set(
            str(BBOX_PATH)
        )

        bbox_ls_aligned = align_lineset(
            bbox_ls_raw,
            centroid,
            R
        )

        geoms.append(bbox_ls_aligned)
        print_bounds("bbox_aligned", bbox_ls_aligned)
    else:
        print(f"[WARN] bbox 파일 없음: {BBOX_PATH}")

    show_with_big_points(
        geoms,
        point_size=5.0
    )


if __name__ == "__main__":
    main()