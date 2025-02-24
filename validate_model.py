import os
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from fine_tune_model_batch import run_validation, get_training_val_data

POSSIBLE_CONFIGS = {"tiny": ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_t.yaml"),
                    "small": ("sam2.1_hiera_small.pt", "sam2.1_hiera_s.yaml"),
                    "base_plus": ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml"),
                    "large": ("sam2.1_hiera_large.pt", "sam2.1_hiera_l.yaml")}

BASE_MODEL_CONFIG = POSSIBLE_CONFIGS["large"]
CHECKPOINT_NAME = f"{BASE_MODEL_CONFIG[0][:-3]}_seismic_120_epochs.pth"


def main():
    sam2_checkpoint = os.path.join("checkpoints", BASE_MODEL_CONFIG[0]) # path to model weight
    model_cfg = os.path.join("configs", "sam2.1", BASE_MODEL_CONFIG[1]) # model config

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") # load model
    predictor = SAM2ImagePredictor(sam2_model) # load net

    predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder 
    predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

    _, validation_data = get_training_val_data()

    # predictor.model.load_state_dict(torch.load(f"./checkpoints/{CHECKPOINT_NAME}"))
    run_validation(predictor, validation_data, n_batches_to_plot=25)


if __name__ == "__main__":
    main()
