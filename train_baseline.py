import sys
import os

# Load the tracked source and explicitly select the single-pass Medium baseline.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf-detr", "src"))

from rfdetr import RFDETRMedium

def main():
    print("Initializing BASELINE RF-DETR Medium...")
    model = RFDETRMedium(recursive_stages=1)
    
    model.train(
        dataset_dir="/WAVE/archive/projects/hyang_lab/CascadeEXP2/cascade-detr/cascade_dn_detr/data/coco",
        epochs=12,
        batch_size=4, 
        lr=1e-4,
        project="runs_baseline", # Saves outputs to a specific folder
        name="baseline_exp"
    )

if __name__ == "__main__":
    main()
    
