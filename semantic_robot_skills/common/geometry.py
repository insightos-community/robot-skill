"""不依赖 Robot、文件或模型的几何辅助函数。"""

import math


def normalize_yaw(value: float) -> float:
    """把弧度角归一化到 [-pi, pi)。"""

    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi

