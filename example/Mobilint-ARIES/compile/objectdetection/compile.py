from qubee import mxq_compile
import os

if __name__ == "__main__":
    onnx_dir = "./onnx_models"
    mxq_dir = "./mxq_models"

    model_list = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]

    success_list = []
    failed_list = []

    for model_name in model_list:
        model_path = os.path.join(onnx_dir, model_name)
        save_name = os.path.splitext(model_name)[0] + ".mxq"
        save_path = os.path.join(mxq_dir, save_name)

        print(f"\n🚀 Compiling {model_path} → {save_path}")

        try:
            mxq_compile(
                model=model_path,
                save_path=save_path,
                calib_data_path="./Calibarition_Images_npy/objectDetection_calibrationDataset/",
                quantize_method="maxpercentile",
                # inference_scheme="multi",
                device="gpu",
                is_quant_ch=True,
            )
            print(f"✅ Success: {model_name}")
            success_list.append(model_name)

        except Exception as e:
            print(f"❌ Failed: {model_name}")
            print(f"   Error: {e}")
            failed_list.append(model_name)

    print("\n==============================")
    print("📌 Compilation Summary")
    print("==============================")

    print("\n✅ Success:")
    for m in success_list:
        print("   -", m)

    print("\n❌ Failed:")
    for m in failed_list:
        print("   -", m)

    print("\n🎉 Done!")
