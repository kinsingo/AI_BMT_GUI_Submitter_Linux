# Compiler version v1.1.2

from sympy import true

import qbcompiler
from qbcompiler import (
    CalibrationConfig,
    mxq_compile,
)
import os

if __name__ == "__main__":
    yolo_decode_include = True  # yolo decode í¬í•¨ ì—¬ë¶€ 
    print(qbcompiler.__version__ + " for object detection")
    onnx_dir = "./onnx_models"
    mxq_dir = "./mxq_models_" + str(qbcompiler.__version__) + ("_decode_included" if yolo_decode_include else "")
    os.makedirs(mxq_dir, exist_ok=True)
    model_list = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]

    success_list = []
    failed_list = []

    calibration_config = CalibrationConfig(
        method=1,  # 0 for per tensor, 1 for per channel
        output=1,  # 0 for layer, 1 for channel (chwise for detection)
        mode=1,  # maxpercentile
        max_percentile={
            "percentile": 0.9999,  # quantization percentile
            "topk_ratio": 0.01,  # quantization topk
        },
    )

    for model_name in model_list:
        model_path = os.path.join(onnx_dir, model_name)
        save_name = os.path.splitext(model_name)[0] + ".mxq"
        save_path = os.path.join(mxq_dir, save_name)
        mblt_path = os.path.join(mxq_dir, os.path.splitext(model_name)[0] + ".mblt")

        if os.path.exists(save_path) and os.path.exists(mblt_path):
            print(f"â­ï¸  Skipping (already exists): {model_name}")
            success_list.append(model_name)
            continue

        print(f"\nðŸš€ Compiling {model_path} â†’ {save_path}")

        try:
            mxq_compile(
                model=model_path,
                calib_data_path="./Calibarition_Images_npy/objectDetection_calibrationDataset/",
                save_subgraph_type=2,  # save mblt file before quantization
                output_subgraph_path=mblt_path,
                save_path=save_path,
                image_channels=3,  # convert to RGB
                backend="onnx",
                inference_scheme="single",  # now support all scheme in one model
                calibration_config=calibration_config,
                yolo_decode_include=yolo_decode_include,  
            )
            print(f"âœ… Success: {model_name}")
            success_list.append(model_name)

        except Exception as e:
            print(f"âŒ Failed: {model_name}")
            print(f"   Error: {e}")
            failed_list.append(model_name)
            # ì—ëŸ¬ ë‚˜ë„ ê³„ì† ì§„í–‰

    print("\n==============================")
    print("ðŸ“Œ Compilation Summary")
    print("==============================")

    print("\nâœ… Success:")
    for m in success_list:
        print("   -", m)

    print("\nâŒ Failed:")
    for m in failed_list:
        print("   -", m)

    print("\nðŸŽ‰ Done!")
