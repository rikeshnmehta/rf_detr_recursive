import sys
import os

# 1. Force Python to load the UNMODIFIED source code
sys.path.insert(0, os.path.abspath("./rf-detr"))

from rfdetr import RFDETRMedium

def main():
    print("Initializing BASE RF-DETR Base for Fine-Tuning...")
    checkpoint_path = "/WAVE/users2/unix/rnmehta/hyang_lab/CascadeEXP2/rf_detr_2/output_rf_base/checkpoint_best_total.pth"
    model = RFDETRMedium(pretrain_weights=checkpoint_path)
    
    model.train(
        dataset_dir="/WAVE/archive/projects/hyang_lab/CascadeEXP2/cascade-detr/cascade_dn_detr/data/coco",
        epochs=12,
        batch_size=4, 
        lr=1e-5,
        project="runs_baseline_ft", # Saves outputs to a specific folder
        name="baseline_exp_ft",
        precision="32"
    )

if __name__ == "__main__":
    main()
