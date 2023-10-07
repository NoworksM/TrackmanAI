import cv2


def draw_arrow(img, direction="up", state="on"):
    height, width, _ = img.shape
    center = (width // 2, height // 2)

    if state == "on":
        color = (0, 255, 0)  # Green for "on"
    else:
        color = (0, 0, 255)  # Red for "off"

    arrow_size = min(width, height) // 3
    thickness = 2

    if direction == "up":
        p1 = (center[0], center[1] - arrow_size)
        p2 = center
        p3 = (center[0] - arrow_size // 2, center[1] - arrow_size // 2)
        p4 = (center[0] + arrow_size // 2, center[1] - arrow_size // 2)
    elif direction == "down":
        p1 = (center[0], center[1] + arrow_size)
        p2 = center
        p3 = (center[0] - arrow_size // 2, center[1] + arrow_size // 2)
        p4 = (center[0] + arrow_size // 2, center[1] + arrow_size // 2)
    # Add "left" and "right" cases if needed

    cv2.line(img, p1, p2, color, thickness)
    cv2.line(img, p2, p3, color, thickness)
    cv2.line(img, p2, p4, color, thickness)
