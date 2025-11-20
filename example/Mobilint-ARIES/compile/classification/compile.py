from qubee import mxq_compile
import os

if __name__ == "__main__":
    # 입력/출력 폴더 정의
    onnx_dir = "./onnx_models"
    mxq_dir = "./mxq_models"

    # onnx 폴더 내의 모든 모델 불러오기
    model_list = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]

    for model_name in model_list:
        model_path = os.path.join(onnx_dir, model_name)
        save_name = os.path.splitext(model_name)[0] + ".mxq"
        save_path = os.path.join(mxq_dir, save_name)

        print(f"Compiling {model_path} → {save_path}")

        mxq_compile(
            model=model_path,
            save_path=save_path,
            calib_data_path="./Calibarition_Images_npy/classification_calibrationDataset/",
            quantize_method="maxpercentile",
            # inference_scheme="multi",
            device="gpu",
            is_quant_ch=True,
        )

    print("\n✅ All models compiled successfully!")



