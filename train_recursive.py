import sys
import os

# Load the tracked source
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf-detr", "src"))

from rfdetr import RFDETRMedium

def main():
    print("Initializing RECURSIVE RF-DETR Medium (T=3)...")
    # Set recursive_stages to 3 for the modified refinement pass
    model = RFDETRMedium(recursive_stages=3)
    
    model.train(
        dataset_dir="/WAVE/archive/projects/hyang_lab/CascadeEXP2/cascade-detr/cascade_dn_detr/data/coco",
        epochs=1,
        batch_size=2, 
        lr=1e-4,
        project="runs_recursive", 
        name="recursive_ep1"
    )

if __name__ == "__main__":
    main()