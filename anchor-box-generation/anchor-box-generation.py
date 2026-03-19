import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    anchors = []
    stride = image_size / feature_size

    for row in range(feature_size):
        cy = (row + 0.5) * stride
        for col in range(feature_size):
            cx = (col + 0.5) * stride

            for scale in scales:
                for ratio in aspect_ratios:
                    w = scale * math.sqrt(ratio)
                    h = scale / math.sqrt(ratio)

                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0

                    anchors.append([float(x1), float(y1), float(x2), float(y2)])

    return anchors