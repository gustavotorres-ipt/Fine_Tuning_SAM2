import os
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from fine_tune_model_batch import run_validation, get_training_val_data


BASE_MODEL_CONFIG = ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml")
CHECKPOINT_NAME = "sam2.1_hiera_base_plus_seismic_100_epochs.pth"


def main():
    sam2_checkpoint = os.path.join("checkpoints", BASE_MODEL_CONFIG[0]) # path to model weight
    model_cfg = os.path.join("configs", "sam2.1", BASE_MODEL_CONFIG[1]) # model config

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") # load model
    predictor = SAM2ImagePredictor(sam2_model) # load net

    predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder 
    predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

    _, validation_data = get_training_val_data()

    predictor.model.load_state_dict(torch.load(f"./checkpoints/{CHECKPOINT_NAME}"))
    run_validation(predictor, validation_data, n_batches_to_plot=25)


if __name__ == "__main__":
    main()
