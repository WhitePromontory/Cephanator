from config import ANATOMICAL_LANDMARKS
import cv2
import numpy as np


def draw_labeled_point(img, x, y, label, point_color = (0.0,255), label_color=(0,0,0)):
    cv2.circle(img, (x, y), 5, point_color, -1)
    cv2.putText(img, label, (x+6, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)



def draw_points(image,landmarks, scale=2,
                point_radius =5, point_color = (0.0,255), point_thickness=-1,
                 offset =6, font = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 0.5, label_color=(0,0,255),
                target_size =512):

    # ---- tensor → numpy ----
    img = image.detach().cpu().numpy()

     # handle channel-first tensors: (3, H, W) RGB -> (H,W,3)
    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

           # ---- resize for display ----
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    points = landmarks.detach().cpu().numpy()

    landmark_symbols = [ v["symbol"] for v in ANATOMICAL_LANDMARKS.values() ]

    for i, point in enumerate(points):
        x,y = point

                # scaled landmark
        x_s = int(x * target_size * scale)
        y_s = int(y * target_size * scale)

        label = landmark_symbols[i]

        cv2.circle(img, (x_s, y_s), radius = point_radius, color = point_color, thickness = point_thickness)
        cv2.putText(img, label, (x_s+ offset, y_s- offset),
                font, font_scale, label_color)

    return img


def draw_batch (images, landmarks):

    for image, landmark in zip(images, landmarks):

        # image tensor is now shape (3, 2300, 2200) -> A TENSOR
        # landmark tensor is now shape (29, 2)    -> A TENSOR

        result = draw_points(image, landmark)
        cv2.imshow("check", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



