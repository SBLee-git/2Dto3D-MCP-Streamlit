import os
import base64
import hashlib
import shutil
import zipfile

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# 결과 ZIP을 저장할 디렉터리
STATIC_DIR = "static"
# 임시 작업용 디렉터리
TEMP_DIR = "mcp_temp"


def convert_map(b64_image: str) -> str:
    """
    Base64 인코딩된 2D 이미지를 받아,
    벽체를 개별 OBJ로 변환한 뒤 ZIP으로 묶어 static 디렉터리에 저장하고,
    실제 파일 경로를 반환합니다.
    """

    # 1) 캐시 체크
    img_bytes = base64.b64decode(b64_image)
    hash_key = hashlib.md5(img_bytes).hexdigest()
    file_name = f"map_walls_{hash_key}.zip"
    zip_static_path = os.path.join(STATIC_DIR, file_name)
    if os.path.exists(zip_static_path):
        return zip_static_path

    # 2) 디렉터리 준비
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    zip_temp_path = os.path.join(TEMP_DIR, file_name)

    # 3) 이미지 디코딩
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")

    # 4) 파라미터
    cm_per_pixel = 1.0
    wall_height = 200.0
    wall_thick = 2

    # 5) 엣지 → 팽창 → 클로징
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wall_thick, wall_thick))
    ker2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(edges, ker1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker2, iterations=2)

    # 6) Connected Components
    n_labels, labels = cv2.connectedComponents(mask)
    parts = []

    # 7) Shapely union → verts, faces 수집
    for lid in range(1, n_labels):
        comp_mask = (labels == lid).astype(np.uint8) * 255
        if cv2.countNonZero(comp_mask) < 20:
            continue

        contours, hierarchy = cv2.findContours(
            comp_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            continue

        # Polygon 병합
        polys = []
        for i, cnt in enumerate(contours):
            outer = [tuple(p[0]) for p in cnt]
            holes = []
            child = hierarchy[0][i][2]
            while child != -1:
                holes.append([tuple(p[0]) for p in contours[child]])
                child = hierarchy[0][child][0]
            polys.append(Polygon(outer, holes))

        merged = unary_union(polys)

        verts_local = []
        faces_local = []
        offset = 0

        # 외곽선
        ext_coords = list(merged.exterior.coords)
        for j in range(len(ext_coords)):
            x0, y0 = ext_coords[j]
            x1, y1 = ext_coords[(j + 1) % len(ext_coords)]
            # 4개 vert
            verts_local += [
                (x0 * cm_per_pixel, y0 * cm_per_pixel, 0.0),
                (x1 * cm_per_pixel, y1 * cm_per_pixel, 0.0),
                (x1 * cm_per_pixel, y1 * cm_per_pixel, wall_height),
                (x0 * cm_per_pixel, y0 * cm_per_pixel, wall_height),
            ]
            # 2개 face
            faces_local += [
                (offset + 1, offset + 2, offset + 3),
                (offset + 3, offset + 4, offset + 1),
            ]
            offset += 4

        # 내부 홀
        for interior in merged.interiors:
            hole_coords = list(interior.coords)
            for k in range(len(hole_coords)):
                x0, y0 = hole_coords[k]
                x1, y1 = hole_coords[(k + 1) % len(hole_coords)]
                verts_local += [
                    (x0 * cm_per_pixel, y0 * cm_per_pixel, 0.0),
                    (x1 * cm_per_pixel, y1 * cm_per_pixel, 0.0),
                    (x1 * cm_per_pixel, y1 * cm_per_pixel, wall_height),
                    (x0 * cm_per_pixel, y0 * cm_per_pixel, wall_height),
                ]
                faces_local += [
                    (offset + 1, offset + 3, offset + 2),
                    (offset + 3, offset + 4, offset + 1),
                ]
                offset += 4

        if verts_local:
            parts.append((verts_local, faces_local))

    # 8) ZIP에 개별 OBJ 작성
    with zipfile.ZipFile(zip_temp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, (verts, faces) in enumerate(parts):
            lines = []
            # v lines
            for x, y, z in verts:
                lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
            # f lines
            for a, b, c in faces:
                lines.append(f"f {a} {b} {c}")
            zf.writestr(f"wall_{idx}.obj", "\n".join(lines))

    # 9) ZIP 이동
    shutil.move(zip_temp_path, zip_static_path)

    # 10) 파일 경로 반환
    return zip_static_path
