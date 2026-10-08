import sys
import os

# 1. Force Python to load the UNMODIFIED source code
sys.path.insert(0, os.path.abspath("./rf-detr"))

from rfdetr import RFDETRMedium

def main():
    print("Initializing BASELINE RF-DETR Medium...")
    model = RFDETRMedium()
    
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
    
