from qubee.calibration import make_calib_man
import cv2
import numpy as np

def preprocess_yolo(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    return img

make_calib_man(
    pre_ftn=preprocess_yolo,
    data_dir="./Calibration_Images/",
    save_dir="./Calibarition_Images_npy/",
    save_name="objectDetection_calibrationDataset",
    max_size=1024
    )