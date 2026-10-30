import numpy as np

from utils import rotation

CAM = 3.4      # jarak kamera virtual
FOCAL = 0.95   # dikali sisi terpendek viewport


def project(pts: np.ndarray, yaw: float, pitch: float, zoom: float, width: float, height: float):
    # harus identik dengan vertex shader di static/index.html
    p = (pts @ rotation(yaw, pitch).T) * zoom
    depth = np.maximum(CAM - p[:, 2], 0.2)
    s = FOCAL * min(width, height) / depth
    return width / 2 + p[:, 0] * s, height / 2 - p[:, 1] * s, depth
