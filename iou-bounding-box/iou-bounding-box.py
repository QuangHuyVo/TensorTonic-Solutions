def iou(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # intersection box
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)

    # no overlap
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    union = area_a + area_b - intersection

    return intersection / union