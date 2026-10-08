import sys
import os

# 1. Force Python to load the UNMODIFIED source code
sys.path.insert(0, os.path.abspath("./rf-detr"))

from rfdetr import RFDETRMedium

def main():
    print("Resuming BASELINE RF-DETR Medium from Epoch 9...")
    
    # Do NOT pass the checkpoint here for an interrupted run
    model = RFDETRMedium() 
    
    model.train(
        dataset_dir="/WAVE/archive/projects/hyang_lab/CascadeEXP2/cascade-detr/cascade_dn_detr/data/coco",
        epochs=12,
        batch_size=4, 
        lr=1e-4,
        project="runs_baseline",
        name="baseline_exp",
        # Pass the .ckpt here to safely restore the exact training state
        resume="/WAVE/users2/unix/rnmehta/hyang_lab/CascadeEXP2/rf_detr_2/output/checkpoint_9.ckpt" 
    )

if __name__ == "__main__":
    main()