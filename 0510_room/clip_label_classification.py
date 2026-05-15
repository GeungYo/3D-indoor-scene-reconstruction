import os
import csv
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import torch
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# 입력: 객체별 PLY 폴더
# ============================================================
OBJECT_DIR = os.path.join(
    "ply",
    "detected_objects"
)

# ============================================================
# 출력 폴더
# ============================================================
OUT_DIR = os.path.join(
    "ply",
    "clip_labeling_results_rgb"
)

OUT_RENDER_DIR = os.path.join(
    OUT_DIR,
    "rendered_views"
)

OUT_CSV = os.path.join(
    OUT_DIR,
    "clip_object_predictions_rgb.csv"
)

# ============================================================
# CLIP 모델
# ============================================================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ============================================================
# 후보 라벨
# ============================================================
LABELS = [
    "bed",
    "desk",
    "chair",
    "table",
    "cabinet",
    "shelf",
    "drawer",
    "sofa",
    "monitor",
    "box",
    "trash bin",
    "unknown furniture"
]

# ============================================================
# RGB 렌더링에 맞는 프롬프트
# ============================================================
PROMPT_TEMPLATES = [
    "a colored 3D scan of a {}",
    "a colored point cloud rendering of a {}",
    "a rendered view of a {}",
    "a 3D object rendering of a {}"
]

# ============================================================
# 렌더링 설정
# ============================================================
IMAGE_SIZE = 224
POINT_RADIUS = 2

# 여러 시점
VIEWS = [
    ("front", 0, 0),
    ("back", 180, 0),
    ("left", 90, 0),
    ("right", -90, 0),
    ("top", 0, 90),
    ("diag_1", 45, 25),
    ("diag_2", 135, 25),
    ("diag_3", -45, 25),
]

SAVE_RENDERED_IMAGES = True
TOPK = 3


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
# 회전 행렬
# ------------------------------------------------------------
def rotation_matrix_y(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float32)


def rotation_matrix_x(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=np.float32)


def rotate_points(points, yaw_deg, pitch_deg):
    ry = rotation_matrix_y(yaw_deg)
    rx = rotation_matrix_x(pitch_deg)

    rotated = points @ ry.T
    rotated = rotated @ rx.T

    return rotated


# ------------------------------------------------------------
# 포인트 정규화
# ------------------------------------------------------------
def normalize_points(points):
    center = np.mean(points, axis=0, keepdims=True)
    normalized = points - center

    max_range = np.max(np.linalg.norm(normalized, axis=1))
    if max_range > 1e-8:
        normalized = normalized / max_range

    return normalized


# ------------------------------------------------------------
# PLY 로드 (points + colors)
# ------------------------------------------------------------
def load_ply_points_and_colors(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)

    if len(pcd.points) == 0:
        raise RuntimeError(f"포인트가 0개인 파일: {ply_path}")

    points = np.asarray(pcd.points, dtype=np.float32)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)

        # Open3D colors는 보통 [0,1] 범위
        if colors.max() <= 1.0:
            colors = colors * 255.0

        colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        colors = None

    return points, colors


# ------------------------------------------------------------
# Point Cloud → RGB projection image
# ------------------------------------------------------------
def points_to_rgb_image(points, colors, image_size=224, point_radius=2):
    """
    x, y 평면으로 투영해서 원본 RGB 색을 그대로 그림.
    z가 큰 점(앞쪽)이 나중에 그려지게 정렬해서 가림 효과를 약간 반영.
    """
    points = normalize_points(points)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    px = ((x + 1.0) * 0.5 * (image_size - 1)).astype(np.int32)
    py = ((1.0 - (y + 1.0) * 0.5) * (image_size - 1)).astype(np.int32)

    px = np.clip(px, 0, image_size - 1)
    py = np.clip(py, 0, image_size - 1)

    # 색이 없으면 회색 fallback
    if colors is None:
        colors_to_draw = np.tile(
            np.array([[140, 140, 140]], dtype=np.uint8),
            (len(points), 1)
        )
    else:
        colors_to_draw = colors.copy()

    # 뒤에 있는 점부터 그리고 앞 점을 나중에 그려서 약간 더 자연스럽게
    order = np.argsort(z)

    img = Image.new(
        "RGB",
        (image_size, image_size),
        color=(255, 255, 255)
    )
    draw = ImageDraw.Draw(img)

    for idx in order:
        xi = int(px[idx])
        yi = int(py[idx])
        r, g, b = colors_to_draw[idx]

        draw.ellipse(
            (
                xi - point_radius,
                yi - point_radius,
                xi + point_radius,
                yi + point_radius
            ),
            fill=(int(r), int(g), int(b))
        )

    return img


# ------------------------------------------------------------
# 객체 1개에서 여러 view 이미지 생성
# ------------------------------------------------------------
def render_multi_views(points, colors, object_name):
    images = []

    for view_name, yaw, pitch in VIEWS:
        rotated_points = rotate_points(points, yaw, pitch)

        img = points_to_rgb_image(
            rotated_points,
            colors,
            image_size=IMAGE_SIZE,
            point_radius=POINT_RADIUS
        )

        images.append(img)

        if SAVE_RENDERED_IMAGES:
            object_view_dir = os.path.join(
                resolve_path(OUT_RENDER_DIR),
                object_name
            )
            os.makedirs(object_view_dir, exist_ok=True)

            img_path = os.path.join(
                object_view_dir,
                f"{view_name}.png"
            )
            img.save(img_path)

    return images


# ------------------------------------------------------------
# CLIP prompt 만들기
# ------------------------------------------------------------
def build_text_prompts(labels):
    prompts = []
    prompt_to_label = []

    for label in labels:
        for template in PROMPT_TEMPLATES:
            prompts.append(template.format(label))
            prompt_to_label.append(label)

    return prompts, prompt_to_label


# ------------------------------------------------------------
# CLIP 예측
# ------------------------------------------------------------
@torch.no_grad()
def predict_with_clip(model, processor, images, device):
    prompts, prompt_to_label = build_text_prompts(LABELS)

    label_scores_accum = {
        label: 0.0 for label in LABELS
    }

    for img in images:
        inputs = processor(
            text=prompts,
            images=img,
            return_tensors="pt",
            padding=True
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image[0]
        probs = logits_per_image.softmax(dim=0).detach().cpu().numpy()

        temp_label_scores = {
            label: [] for label in LABELS
        }

        for prob, label in zip(probs, prompt_to_label):
            temp_label_scores[label].append(float(prob))

        for label in LABELS:
            label_scores_accum[label] += float(
                np.mean(temp_label_scores[label])
            )

    num_views = len(images)

    final_scores = {
        label: score / num_views
        for label, score in label_scores_accum.items()
    }

    score_sum = sum(final_scores.values())
    if score_sum > 0:
        final_scores = {
            label: score / score_sum
            for label, score in final_scores.items()
        }

    sorted_scores = sorted(
        final_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return sorted_scores


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    object_dir = resolve_path(OBJECT_DIR)
    out_dir = resolve_path(OUT_DIR)
    render_dir = resolve_path(OUT_RENDER_DIR)
    out_csv = resolve_path(OUT_CSV)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    if not os.path.exists(object_dir):
        raise FileNotFoundError(f"객체 폴더가 없음: {object_dir}")

    object_files = sorted([
        file_name
        for file_name in os.listdir(object_dir)
        if file_name.lower().endswith(".ply")
        and file_name.startswith("object_")
    ])

    if len(object_files) == 0:
        raise RuntimeError("object_###.ply 파일이 없어.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] device: {device}")
    print(f"[INFO] object count: {len(object_files)}")
    print(f"[INFO] CLIP model: {CLIP_MODEL_NAME}")

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    csv_rows = []

    for file_name in object_files:
        object_name = os.path.splitext(file_name)[0]
        ply_path = os.path.join(object_dir, file_name)

        print("\n========================================")
        print(f"[OBJECT] {file_name}")

        points, colors = load_ply_points_and_colors(ply_path)
        print(f"[INFO] points: {len(points):,}")
        print(f"[INFO] has colors: {colors is not None}")

        images = render_multi_views(points, colors, object_name)

        sorted_scores = predict_with_clip(
            model=model,
            processor=processor,
            images=images,
            device=device
        )

        top_results = sorted_scores[:TOPK]

        for rank, (label, score) in enumerate(top_results, start=1):
            print(f"[TOP{rank}] {label}: {score:.4f}")

        row = {
            "file_name": file_name,
            "num_points": len(points),
        }

        for rank, (label, score) in enumerate(top_results, start=1):
            row[f"top{rank}_label"] = label
            row[f"top{rank}_score"] = score

        csv_rows.append(row)

    fieldnames = [
        "file_name",
        "num_points",
        "top1_label",
        "top1_score",
        "top2_label",
        "top2_score",
        "top3_label",
        "top3_score"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n========================================")
    print(f"[SAVE] 예측 결과 CSV: {out_csv}")
    print(f"[SAVE] 렌더링 이미지 폴더: {render_dir}")
    print("========================================")


if __name__ == "__main__":
    main()