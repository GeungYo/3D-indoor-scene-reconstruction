import pickle
from pathlib import Path

import numpy as np
import open3d as o3d


# ============================================================
# 경로 설정
# ============================================================
# merge.py가 있는 폴더를 기준으로 모든 경로를 찾는다.
SCRIPT_DIR = Path(__file__).resolve().parent
PLY_DIR = SCRIPT_DIR / "ply"

PKL_PATH = SCRIPT_DIR / "walls_data.pkl"
ROOM_MESH_PATH = SCRIPT_DIR / "textured_room_vertexcolor.ply"

# object_detecting_final.py에서 생성되는 객체별 Mesh
OBJECT_MESH_PATTERN = "08_refined_object_mesh_*.ply"

# 선택 사항: 객체 Bounding Box도 같이 표시
BBOX_PATH = PLY_DIR / "06_detected_object_bounding_boxes.ply"
SHOW_BBOX = True

# 방 Mesh와 객체 Mesh를 실제 하나의 PLY로 저장
SAVE_COMBINED_MESH = True
COMBINED_MESH_PATH = SCRIPT_DIR / "final_room_with_object_meshes.ply"


# ------------------------------------------------------------
# 좌표 변환 정보 읽기
# ------------------------------------------------------------
def load_alignment(pkl_path):
    with open(pkl_path, "rb") as file:
        data = pickle.load(file)

    alignment = data.get("alignment", None)

    if alignment is None:
        raise RuntimeError("walls_data.pkl에 alignment 정보가 없어.")

    centroid = np.array(
        alignment["centroid"],
        dtype=np.float64
    )

    rotation = np.array(
        alignment["R"],
        dtype=np.float64
    )

    return centroid, rotation


def align_points(points_raw, centroid, rotation):
    """원본 좌표를 방 Mesh 좌표로 이동하고 회전한다."""
    return (points_raw - centroid) @ rotation.T


# ------------------------------------------------------------
# 객체 Mesh 좌표 맞추기
# ------------------------------------------------------------
def align_triangle_mesh(mesh, centroid, rotation):
    vertices_raw = np.asarray(mesh.vertices)
    vertices_aligned = align_points(
        vertices_raw,
        centroid,
        rotation
    )

    aligned_mesh = o3d.geometry.TriangleMesh()

    aligned_mesh.vertices = o3d.utility.Vector3dVector(
        vertices_aligned
    )

    aligned_mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.triangles).copy()
    )

    if mesh.has_vertex_colors():
        aligned_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertex_colors).copy()
        )
    else:
        aligned_mesh.paint_uniform_color([0.6, 0.6, 0.6])

    if mesh.has_vertex_normals():
        normals_raw = np.asarray(mesh.vertex_normals)
        normals_aligned = normals_raw @ rotation.T

        aligned_mesh.vertex_normals = o3d.utility.Vector3dVector(
            normals_aligned
        )
    else:
        aligned_mesh.compute_vertex_normals()

    aligned_mesh.compute_triangle_normals()
    return aligned_mesh


# ------------------------------------------------------------
# Bounding Box 좌표 맞추기
# ------------------------------------------------------------
def align_lineset(lineset, centroid, rotation):
    points_raw = np.asarray(lineset.points)
    points_aligned = align_points(
        points_raw,
        centroid,
        rotation
    )

    aligned_lineset = o3d.geometry.LineSet()

    aligned_lineset.points = o3d.utility.Vector3dVector(
        points_aligned
    )

    aligned_lineset.lines = o3d.utility.Vector2iVector(
        np.asarray(lineset.lines).copy()
    )

    if len(lineset.colors) > 0:
        aligned_lineset.colors = o3d.utility.Vector3dVector(
            np.asarray(lineset.colors).copy()
        )
    else:
        aligned_lineset.paint_uniform_color([1.0, 0.0, 0.0])

    return aligned_lineset


# ------------------------------------------------------------
# 정보 출력
# ------------------------------------------------------------
def print_bounds(name, geometry):
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        points = np.asarray(geometry.vertices)
    elif isinstance(geometry, o3d.geometry.LineSet):
        points = np.asarray(geometry.points)
    else:
        return

    if len(points) == 0:
        print(f"[INFO] {name}: empty")
        return

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = points.mean(axis=0)

    print(f"\n[INFO] {name}")
    print(f"       min    = {minimum}")
    print(f"       max    = {maximum}")
    print(f"       center = {center}")


# ------------------------------------------------------------
# 방 Mesh + 객체 Mesh를 하나로 합쳐 저장
# ------------------------------------------------------------
def combine_meshes(room_mesh, object_meshes):
    combined = o3d.geometry.TriangleMesh(room_mesh)

    if not combined.has_vertex_colors():
        combined.paint_uniform_color([0.7, 0.7, 0.7])

    for object_mesh in object_meshes:
        combined += object_mesh

    combined.remove_duplicated_vertices()
    combined.remove_duplicated_triangles()
    combined.remove_degenerate_triangles()
    combined.remove_unreferenced_vertices()
    combined.compute_vertex_normals()
    combined.compute_triangle_normals()

    return combined


def save_combined_mesh(mesh):
    ok = o3d.io.write_triangle_mesh(
        str(COMBINED_MESH_PATH),
        mesh,
        write_ascii=False,
        write_vertex_normals=True,
        write_vertex_colors=True
    )

    print(
        f"\n[SAVE] {COMBINED_MESH_PATH.name} "
        f"-> {'OK' if ok else 'FAIL'}"
    )


# ------------------------------------------------------------
# 시각화
# ------------------------------------------------------------
def show_scene(geometries):
    visualizer = o3d.visualization.Visualizer()

    visualizer.create_window(
        window_name="Room mesh + aligned object meshes",
        width=1280,
        height=800
    )

    for geometry in geometries:
        visualizer.add_geometry(geometry)

    options = visualizer.get_render_option()
    options.mesh_show_back_face = True

    visualizer.run()
    visualizer.destroy_window()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if not PKL_PATH.exists():
        raise FileNotFoundError(f"PKL 없음: {PKL_PATH}")

    if not ROOM_MESH_PATH.exists():
        raise FileNotFoundError(f"방 Mesh 없음: {ROOM_MESH_PATH}")

    object_mesh_paths = sorted(
        PLY_DIR.glob(OBJECT_MESH_PATTERN)
    )

    if len(object_mesh_paths) == 0:
        raise FileNotFoundError(
            f"객체 Mesh가 없음: {PLY_DIR / OBJECT_MESH_PATTERN}\n"
            f"먼저 object_detecting_final.py를 실행해야 해."
        )

    centroid, rotation = load_alignment(PKL_PATH)

    # 1) 방 Mesh 읽기
    room_mesh = o3d.io.read_triangle_mesh(
        str(ROOM_MESH_PATH)
    )

    if len(room_mesh.vertices) == 0:
        raise RuntimeError("방 Mesh가 비어 있어.")

    if not room_mesh.has_vertex_colors():
        room_mesh.paint_uniform_color([0.7, 0.7, 0.7])

    room_mesh.compute_vertex_normals()
    room_mesh.compute_triangle_normals()

    print_bounds("room_mesh", room_mesh)

    # 2) 객체별 Mesh를 읽고 방 좌표로 맞추기
    aligned_object_meshes = []

    for mesh_path in object_mesh_paths:
        object_mesh_raw = o3d.io.read_triangle_mesh(
            str(mesh_path)
        )

        if (
            len(object_mesh_raw.vertices) == 0 or
            len(object_mesh_raw.triangles) == 0
        ):
            print(f"[WARN] 비어 있는 Mesh 건너뜀: {mesh_path.name}")
            continue

        object_mesh_aligned = align_triangle_mesh(
            object_mesh_raw,
            centroid,
            rotation
        )

        aligned_object_meshes.append(object_mesh_aligned)
        print_bounds(mesh_path.name, object_mesh_aligned)

    if len(aligned_object_meshes) == 0:
        raise RuntimeError("정상적으로 불러온 객체 Mesh가 하나도 없어.")

    print(
        f"\n[INFO] 불러온 객체 Mesh: "
        f"{len(aligned_object_meshes)}개"
    )

    # 3) 시각화 목록
    geometries = [room_mesh]
    geometries.extend(aligned_object_meshes)

    # 4) Bounding Box도 같은 좌표로 맞춰서 표시
    if SHOW_BBOX and BBOX_PATH.exists():
        bbox_raw = o3d.io.read_line_set(str(BBOX_PATH))

        if len(bbox_raw.points) > 0:
            bbox_aligned = align_lineset(
                bbox_raw,
                centroid,
                rotation
            )

            geometries.append(bbox_aligned)
            print_bounds("bbox_aligned", bbox_aligned)
    elif SHOW_BBOX:
        print(f"[WARN] Bounding Box 파일 없음: {BBOX_PATH}")

    # 5) 방과 객체를 실제 하나의 Mesh로 저장
    if SAVE_COMBINED_MESH:
        combined_mesh = combine_meshes(
            room_mesh,
            aligned_object_meshes
        )
        save_combined_mesh(combined_mesh)

    # 6) 최종 화면 표시
    show_scene(geometries)


if __name__ == "__main__":
    main()
