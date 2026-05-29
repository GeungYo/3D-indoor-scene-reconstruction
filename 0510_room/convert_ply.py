import open3d as o3d
import numpy as np
from pathlib import Path
import json

# ===== 경로 설정 =====
INPUT_FOLDER = Path("0510_room/ply/detected_objects")
OUTPUT_FOLDER = INPUT_FOLDER / "obj_models"

OUTPUT_FOLDER.mkdir(exist_ok=True)

# ===== voxel 크기 =====
VOXEL_SIZE = 0.01

# 중요:
# 여러 OBJ의 원래 위치 관계를 유지하려면 무조건 False
# True로 바꾸면 객체가 자기 중심 기준으로 이동해서 위치가 깨질 수 있음
CENTER_OBJECT = False

# 색 단계 수
# 8이면 색 종류 최대 8*8*8 = 512개
# 더 원본 색에 가깝게 하려면 16
# 너무 높이면 Unity material이 너무 많아질 수 있음
COLOR_LEVELS = 8

# OBJ 옆에 좌표/범위 정보 JSON 저장
SAVE_META_JSON = True

# voxel mesh vertex 좌표를 CSV로도 저장할지 여부
# VOXEL_SIZE가 0.01이면 파일이 커질 수 있어서 기본 False 추천
SAVE_VERTEX_XYZ_CSV = False

# 기존 obj_models 안의 old obj/mtl/json/csv 정리
CLEAR_OLD_OBJ_FILES = True


FACE_DEFS = [
    ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),

    ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),

    ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
]


def clear_old_obj_files():
    if not CLEAR_OLD_OBJ_FILES:
        return

    removed_count = 0

    for path in OUTPUT_FOLDER.iterdir():
        if path.suffix.lower() in [".obj", ".mtl", ".json", ".csv"]:
            path.unlink()
            removed_count += 1

    print(f"[INFO] old obj model files removed: {removed_count}")


def quantize_color(color, levels=8):
    color = np.clip(color, 0.0, 1.0)
    q = np.round(color * (levels - 1)).astype(int)

    key = (int(q[0]), int(q[1]), int(q[2]))
    mat_color = q.astype(np.float64) / (levels - 1)

    return key, mat_color


def mat_name_from_key(key):
    return f"mat_{key[0]}_{key[1]}_{key[2]}"


def pointcloud_to_voxel_colored_obj_data(pcd, voxel_size=0.01, center_object=False):
    if len(pcd.points) == 0:
        return None

    has_color = pcd.has_colors()

    original_points = np.asarray(pcd.points)
    original_min_bound = original_points.min(axis=0)
    original_max_bound = original_points.max(axis=0)
    original_center = original_points.mean(axis=0)
    original_extent = original_max_bound - original_min_bound

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd,
        voxel_size=voxel_size
    )

    voxels = voxel_grid.get_voxels()

    if len(voxels) == 0:
        return None

    # 중요:
    # voxel_grid.origin도 원본 point cloud 좌표계 기준임.
    # 따라서 origin + voxel_index * voxel_size로 만든 vertex도 원본 좌표계 기준이 됨.
    origin = np.asarray(voxel_grid.origin)

    voxel_indices = set()
    voxel_colors = {}

    for v in voxels:
        idx = tuple(v.grid_index)
        voxel_indices.add(idx)

        if has_color:
            voxel_colors[idx] = np.asarray(v.color, dtype=np.float64)
        else:
            voxel_colors[idx] = np.array([0.7, 0.7, 0.7], dtype=np.float64)

    vertices = []
    faces = []
    face_material_keys = []
    material_colors = {}

    for idx in voxel_indices:
        idx_np = np.array(idx, dtype=np.float64)

        # 이 좌표가 voxel cube의 최소 꼭짓점 좌표
        # 원본 point cloud 좌표계 기준
        voxel_min = origin + idx_np * voxel_size

        voxel_color = voxel_colors[idx]
        mat_key, mat_color = quantize_color(voxel_color, COLOR_LEVELS)
        material_colors[mat_key] = mat_color

        for neighbor_dir, face_corners in FACE_DEFS:
            neighbor_idx = (
                idx[0] + neighbor_dir[0],
                idx[1] + neighbor_dir[1],
                idx[2] + neighbor_dir[2],
            )

            # 이웃 voxel이 있으면 내부 면이라서 생략
            if neighbor_idx in voxel_indices:
                continue

            base_idx = len(vertices)

            for corner in face_corners:
                corner_np = np.array(corner, dtype=np.float64)

                # 최종 OBJ에 들어갈 vertex 좌표
                # 원본 point cloud 좌표계 기준
                vertex = voxel_min + corner_np * voxel_size
                vertices.append(vertex)

            # 사각형 면 1개를 삼각형 2개로 저장
            faces.append([base_idx, base_idx + 1, base_idx + 2])
            face_material_keys.append(mat_key)

            faces.append([base_idx, base_idx + 2, base_idx + 3])
            face_material_keys.append(mat_key)

    if len(vertices) == 0 or len(faces) == 0:
        return None

    vertices = np.array(vertices, dtype=np.float64)

    # 중요:
    # 원래 위치 관계 유지가 목표라면 이 부분이 실행되면 안 됨.
    if center_object:
        center = vertices.mean(axis=0)
        vertices = vertices - center
        print("[WARNING] CENTER_OBJECT=True라서 객체가 중심 기준으로 이동됨.")
        print("[WARNING] Unity에서 원래 위치 관계를 유지하려면 CENTER_OBJECT=False로 둬야 함.")

    mesh_min_bound = vertices.min(axis=0)
    mesh_max_bound = vertices.max(axis=0)
    mesh_center = vertices.mean(axis=0)
    mesh_extent = mesh_max_bound - mesh_min_bound

    meta = {
        "coordinate_system": "same_as_input_point_cloud",
        "note": "OBJ vertex coordinates are saved in the original point cloud coordinate system. No centering is applied when CENTER_OBJECT is False.",

        "center_object": bool(center_object),
        "voxel_size": float(voxel_size),
        "color_levels": int(COLOR_LEVELS),

        "input_point_count": int(len(original_points)),
        "voxel_count": int(len(voxels)),
        "obj_vertex_count": int(len(vertices)),
        "obj_triangle_count": int(len(faces)),
        "material_count": int(len(material_colors)),
        "has_color": bool(has_color),

        "voxel_grid_origin": {
            "x": float(origin[0]),
            "y": float(origin[1]),
            "z": float(origin[2]),
        },

        "original_point_min_bound": {
            "x": float(original_min_bound[0]),
            "y": float(original_min_bound[1]),
            "z": float(original_min_bound[2]),
        },
        "original_point_max_bound": {
            "x": float(original_max_bound[0]),
            "y": float(original_max_bound[1]),
            "z": float(original_max_bound[2]),
        },
        "original_point_mean_center": {
            "x": float(original_center[0]),
            "y": float(original_center[1]),
            "z": float(original_center[2]),
        },
        "original_point_extent": {
            "x": float(original_extent[0]),
            "y": float(original_extent[1]),
            "z": float(original_extent[2]),
        },

        "obj_mesh_min_bound": {
            "x": float(mesh_min_bound[0]),
            "y": float(mesh_min_bound[1]),
            "z": float(mesh_min_bound[2]),
        },
        "obj_mesh_max_bound": {
            "x": float(mesh_max_bound[0]),
            "y": float(mesh_max_bound[1]),
            "z": float(mesh_max_bound[2]),
        },
        "obj_mesh_mean_center": {
            "x": float(mesh_center[0]),
            "y": float(mesh_center[1]),
            "z": float(mesh_center[2]),
        },
        "obj_mesh_extent": {
            "x": float(mesh_extent[0]),
            "y": float(mesh_extent[1]),
            "z": float(mesh_extent[2]),
        },
    }

    return vertices, faces, face_material_keys, material_colors, meta


def write_colored_obj_with_mtl(
    obj_path,
    vertices,
    faces,
    face_material_keys,
    material_colors,
    meta
):
    obj_path = Path(obj_path)
    mtl_path = obj_path.with_suffix(".mtl")

    # ===== MTL 저장 =====
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("# Material file generated from voxelized point cloud\n")
        f.write("# Colors are quantized from original PLY point colors\n\n")

        for key, color in material_colors.items():
            mat_name = mat_name_from_key(key)

            r, g, b = color
            r = float(np.clip(r, 0.0, 1.0))
            g = float(np.clip(g, 0.0, 1.0))
            b = float(np.clip(b, 0.0, 1.0))

            f.write(f"newmtl {mat_name}\n")
            f.write(f"Ka {r:.6f} {g:.6f} {b:.6f}\n")
            f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")
            f.write("d 1.0\n")
            f.write("illum 2\n\n")

    # ===== OBJ 저장 =====
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# OBJ generated from Open3D point cloud PLY\n")
        f.write("# Coordinate system: same_as_input_point_cloud\n")
        f.write("# Vertex format: v x y z\n")
        f.write("# IMPORTANT: coordinates are not centered when CENTER_OBJECT=False\n")
        f.write(f"# voxel_size: {meta['voxel_size']}\n")
        f.write(f"# center_object: {meta['center_object']}\n")
        f.write(f"# input_point_count: {meta['input_point_count']}\n")
        f.write(f"# voxel_count: {meta['voxel_count']}\n")
        f.write(f"# obj_vertex_count: {meta['obj_vertex_count']}\n")
        f.write(f"# obj_triangle_count: {meta['obj_triangle_count']}\n")
        f.write(f"mtllib {mtl_path.name}\n\n")

        # 여기서 v x y z가 OBJ의 실제 3D 좌표
        # 이 값들이 원본 point cloud 좌표계 기준으로 저장됨
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

        f.write("\n")

        current_mat = None

        for face, mat_key in zip(faces, face_material_keys):
            mat_name = mat_name_from_key(mat_key)

            if mat_name != current_mat:
                f.write(f"usemtl {mat_name}\n")
                current_mat = mat_name

            # OBJ index는 1부터 시작
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def save_meta_json(meta_path, source_ply_path, obj_path, mtl_path, meta):
    meta = dict(meta)

    meta["source_ply"] = str(source_ply_path)
    meta["output_obj"] = str(obj_path)
    meta["output_mtl"] = str(mtl_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)


def save_vertex_xyz_csv(csv_path, vertices):
    np.savetxt(
        csv_path,
        vertices,
        delimiter=",",
        header="x,y,z",
        comments="",
        fmt="%.8f"
    )


def main():
    clear_old_obj_files()

    if CENTER_OBJECT:
        print("[WARNING] CENTER_OBJECT=True 상태임.")
        print("[WARNING] 원래 위치 관계를 유지하려면 CENTER_OBJECT=False로 바꿔야 함.")
        print()

    ply_files = sorted(INPUT_FOLDER.glob("*.ply"))

    if not ply_files:
        print("PLY 파일이 없습니다.")
        return

    print(f"입력 폴더: {INPUT_FOLDER}")
    print(f"출력 폴더: {OUTPUT_FOLDER}")
    print(f"voxel size: {VOXEL_SIZE}")
    print(f"center object: {CENTER_OBJECT}")
    print(f"color levels: {COLOR_LEVELS}")
    print()

    for i, ply_path in enumerate(ply_files, start=1):
        print(f"[{i}/{len(ply_files)}] 처리 중: {ply_path.name}")

        pcd = o3d.io.read_point_cloud(str(ply_path))

        if len(pcd.points) == 0:
            print("  건너뜀: 포인트 없음")
            continue

        if not pcd.has_colors():
            print("  주의: 이 PLY에는 color 정보가 없음. 기본 회색으로 저장됨.")

        result = pointcloud_to_voxel_colored_obj_data(
            pcd,
            voxel_size=VOXEL_SIZE,
            center_object=CENTER_OBJECT
        )

        if result is None:
            print("  실패: OBJ 데이터 생성 안 됨")
            continue

        vertices, faces, face_material_keys, material_colors, meta = result

        output_obj_path = OUTPUT_FOLDER / f"{ply_path.stem}_voxel.obj"
        output_mtl_path = output_obj_path.with_suffix(".mtl")
        output_meta_path = OUTPUT_FOLDER / f"{ply_path.stem}_voxel_meta.json"
        output_vertex_csv_path = OUTPUT_FOLDER / f"{ply_path.stem}_voxel_vertices_xyz.csv"

        write_colored_obj_with_mtl(
            output_obj_path,
            vertices,
            faces,
            face_material_keys,
            material_colors,
            meta
        )

        if SAVE_META_JSON:
            save_meta_json(
                output_meta_path,
                ply_path,
                output_obj_path,
                output_mtl_path,
                meta
            )

        if SAVE_VERTEX_XYZ_CSV:
            save_vertex_xyz_csv(
                output_vertex_csv_path,
                vertices
            )

        mesh_center = meta["obj_mesh_mean_center"]
        mesh_extent = meta["obj_mesh_extent"]

        print(f"  저장 완료: {output_obj_path.name}")
        print(f"  MTL 저장 완료: {output_mtl_path.name}")

        if SAVE_META_JSON:
            print(f"  META 저장 완료: {output_meta_path.name}")

        if SAVE_VERTEX_XYZ_CSV:
            print(f"  XYZ CSV 저장 완료: {output_vertex_csv_path.name}")

        print(f"  vertices: {len(vertices)}")
        print(f"  triangles: {len(faces)}")
        print(f"  materials: {len(material_colors)}")
        print(
            "  obj center: "
            f"({mesh_center['x']:.3f}, {mesh_center['y']:.3f}, {mesh_center['z']:.3f})"
        )
        print(
            "  obj extent: "
            f"({mesh_extent['x']:.3f}, {mesh_extent['y']:.3f}, {mesh_extent['z']:.3f})"
        )
        print()

    print("전체 변환 완료")


if __name__ == "__main__":
    main()