from qubee.calibration import make_calib_man
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_resnet50(img_path: str, mean=MEAN, std=STD):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    out = (img - mean) / std
    return out

make_calib_man(
    pre_ftn=preprocess_resnet50,
    data_dir="./Calibration_Images/",
    save_dir="./Calibarition_Images_npy/",
    save_name="classification_calibrationDataset",
    max_size=1024
    )
