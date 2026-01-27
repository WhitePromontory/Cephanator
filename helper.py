from config import ANATOMICAL_LANDMARKS
import cv2


def draw_labeled_point(img, x, y, label, point_color = (0.0,255), label_color=(0,0,0)):
    cv2.circle(img, (x, y), 5, point_color, -1)
    cv2.putText(img, label, (x+6, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)