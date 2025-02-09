import argparse

import torch
import lightning.pytorch as pl

from pretrained import load_pretrained_model

from downstream_mrl.utils.features import prepare_features
from downstream_mrl.model import RibosomeLoadingPredictionWrapper
from downstream_mrl.utils.datamodule import RibosomeLoadingDataModule


TEST_SETS = ["random7600", "human7600"]

MODEL_NAME = "ProtRNA_pretrained"

DATA_ROOT = "downstream_mrl/data"
FEATURE_PATH = f"{DATA_ROOT}/features"
OUTPUT_ROOT = "downstream_mrl/output"

DEVICE = "cuda"


def main(args):

    base_model = load_pretrained_model(name=MODEL_NAME)
    batch_converter = base_model.alphabet.get_batch_converter()

    prepare_features(base_model, batch_converter, DATA_ROOT, FEATURE_PATH, args.test)

    mrl_model = RibosomeLoadingPredictionWrapper(lm_type=args.lm_type)
    mrl_model.load_state_dict(torch.load("weights/mrlHead.ckpt", map_location=DEVICE)["state_dict"], strict=False)
    
    datamodule = RibosomeLoadingDataModule(
        data_root=DATA_ROOT,
        feature_path=FEATURE_PATH,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        skip_data_preparation=True,
        lm_type=args.lm_type,
        test_set=args.test
    ) 

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_steps=-1,
        max_epochs=50,
        gradient_clip_val=None,
        precision="16-mixed",
        default_root_dir=OUTPUT_ROOT,
        log_every_n_steps=50,
        strategy="auto",
    )
    
    trainer.test(model=mrl_model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", type=str, choices=TEST_SETS, required=True
    )
    parser.add_argument(
        "--lm_type", type=str, default="ProtRNA",
    )
    args = parser.parse_args()
    main(args)