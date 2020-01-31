import cv2
import numpy as np
import os


def run_optical_flow_in_folder(images_root_path, save=False, show=True, show_binary=False, smaller_window=False, dataset_magnitude_thresholds=None):
    for path, subfolders, files in os.walk(images_root_path):
        frame1 = None
        hsv = None
        print(path)
        folder_name = path.split("\\")[-1].split("/")[-1]
        if folder_name.startswith("_"):
            continue
        if folder_name in dataset_magnitude_thresholds:
            magnitude_threshold = dataset_magnitude_thresholds[folder_name]
        else:
            magnitude_threshold = 0
        folder_name = "optical_flow_results_" + ("small_window_" if smaller_window else "") + folder_name
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

            window_size = 15 if smaller_window else 35
            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, flow=None, pyr_scale=0.5, levels=3, winsize=window_size,
                                                iterations=3, poly_n=7, poly_sigma=1.5, flags=0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2  # Setting hue depending on the flow direction
            # if reduce_noise:
            #     mag[-1, -1] = 10  # Setting a corner to 10 to hide the noise when no object is moving
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
            if show:
                color_image = cv2.imread(img_path)
                if show_binary:
                    gray_3_channels = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    combined = cv2.addWeighted(color_image, 1, gray_3_channels, 0.5, 0)
                else:
                    combined = cv2.addWeighted(color_image, 1, bgr, 1, 0)
                cv2.imshow("Dense Optical Flow", combined)
                # cv2.imshow("Dense Optical Flow", gray)

            key = cv2.waitKey(1) & 0xff
            if key == 27:  # ESC
                break

            if save:
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name)
                file_name = folder_name + "/" + file
                cv2.imwrite(file_name, gray)

            frame1 = frame2


# name: magnitude threshold
dataset_thresholds = {
    "canoe": 2.25,
    "fall": 2.5,
    "fountain02": 2,
    "highway": 0.75,
    "office": 1,
    "pedestrians": 1,
    "PETS2006": 1.5,
    "turbulence0": 2.5,
    "turbulence1": 3,
    "turbulence2": 1.5,
}

run_optical_flow_in_folder("../images", save=True, show=True, show_binary=True, smaller_window=True, dataset_magnitude_thresholds=dataset_thresholds)
