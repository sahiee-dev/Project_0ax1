"""
Deployment utility to copy and export new model weights.
This script automates the process of moving weights from the training directory
to the project structure and exporting them to ONNX format.
"""

import os
import shutil
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Deploy YOLOv2 weights to the project.")
    parser.add_argument("--source", type=str, 
                        default="/Users/lulu/Downloads/weapon_training_v2/Normal_Compressed_v2/weights/best.pt",
                        help="Path to the source best.pt file")
    args = parser.parse_args()

    print("🚀 Auto-Deploying Nano v2 Weights...\n")
    
    # Paths
    downloads_weights = args.source
    target_dir = os.path.join("runs", "detect", "Normal_Compressed_v2", "weights")
    target_weights = os.path.join(target_dir, "best.pt")
    onnx_target = os.path.join("models", "normal.onnx")
    
    # 1. Check if source exists
    if not os.path.exists(downloads_weights):
        print(f"❌ Error: Could not find weights at {downloads_weights}")
        print("Tip: Use --source to specify the path to your trained weights.")
        return
        
    # 2. Copy the file into the project structure
    print(f"📁 Copying best.pt to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    shutil.copy2(downloads_weights, target_weights)
    print("   ✓ Copied successfully.")
    
    # 3. Export to ONNX
    print("\n⚙️ Exporting model to ONNX format (for compatibility)...")
    try:
        from ultralytics import YOLO
        model = YOLO(target_weights)
        onnx_file = model.export(format="onnx", simplify=True)
        
        # Move the exported ONNX to models/
        print(f"📁 Moving exported ONNX to {onnx_target}...")
        if os.path.exists(onnx_file):
            shutil.move(onnx_file, onnx_target)
            print("   ✓ ONNX deployment complete.")
        else:
            print("   ⚠️  export returned path not found, moving skipped.")
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")
        print("Note: Ensure you are in the correct virtual environment with dependencies installed.")
        
    # 4. Final instructions
    print("\n✅ Deployment complete!")
    print("You can now run `streamlit run ui/app.py` and test the 'Standard (Nano) v2' model.")

if __name__ == "__main__":
    main()
