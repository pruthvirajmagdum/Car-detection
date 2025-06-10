import cv2
import numpy as np

def detect_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    color_ranges = {
        "Red": [(0, 70, 50), (10, 255, 255)],
        "Blue": [(100, 150, 0), (140, 255, 255)],
        "White": [(0, 0, 200), (180, 20, 255)],
        "Black": [(0, 0, 0), (180, 255, 30)],
        "Silver": [(0, 0, 90), (180, 20, 200)]
    }

    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        if cv2.countNonZero(mask) > 0.1 * mask.size:
            return color_name
    return "Unknown"
