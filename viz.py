import os
import open3d as o3d

INPUT_FILE = "separated_objects/object_001.ply"   # ""면 실행 시 입력받음

def here_dir():
    return os.path.dirname(os.path.abspath(__file__))

def _resolve_path(filename: str):
    if os.path.isabs(filename):
        return filename
    return os.path.join(here_dir(), filename)

def main():
    filename = INPUT_FILE
    path = _resolve_path(filename)
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()