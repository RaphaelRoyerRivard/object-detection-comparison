import cv2
import numpy as np
from os import walk


def run_optical_flow_in_folder(images_root_path, magnitude_threshold=2.5, reduce_noise=True):
    for path, subfolders, files in walk(images_root_path):
        frame1 = None
        hsv = None
        detection_lines = ""
        print(path)
        filename = path.split("\\")[-1]
        if filename.startswith("_"):
            continue
        for file in files:
            if not file.endswith(".png") and not file.endswith(".jpg"):
                continue
            print(file)
            img_path = path + "/" + file

            if frame1 is None:
                frame1 = cv2.imread(img_path, 0)
                hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3))  # Width x Height x 3 (Hue, Saturation, Value)
                hsv[..., 1] = 255  # Setting saturation to 100%
                continue

            frame2 = cv2.imread(img_path, 0)

            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, flow=None, pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2  # Setting hue depending on the flow direction
            if reduce_noise:
                mag[-1, -1] = 10  # Setting a corner to 10 to hide the noise when no object is moving
            low_mag_idx = mag < magnitude_threshold
            mag[low_mag_idx] = 0
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Setting the value between 0 and 255 based on min and max of the flow's magnitude
            hsv = hsv.astype(np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # cv2.imshow("Dense Optical Flow", bgr)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            movement_idx = (gray / 255) > 0
            stale_idx = (gray / 255) <= 0
            gray[movement_idx] = 255
            gray[stale_idx] = 0
            cv2.imshow("Dense Optical Flow", gray)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png', frame2)
                cv2.imwrite('opticalhsv.png', bgr)

            frame1 = frame2


# run_optical_flow_in_folder("../images/canoe", magnitude_threshold=1.5)
run_optical_flow_in_folder("../images/PETS2006", magnitude_threshold=2.5)
