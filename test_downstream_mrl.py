import torch
import lightning.pytorch as pl

from pretrained import load_pretrained_model

from downstream_mrl.utils.features import prepare_features
from downstream_mrl.model import RibosomeLoadingPredictionWrapper
from downstream_mrl.utils.datamodule import RibosomeLoadingDataModule


MODEL_NAME = "ProtRNA_pretrained"
TEST_SET = "random7600"
DEVICE = "cuda"

DATA_ROOT = "downstream_mrl/data"
FEATURE_PATH = f"{DATA_ROOT}/features"
OUTPUT_ROOT = "downstream_mrl/output"


if __name__ == '__main__':

    base_model = load_pretrained_model(name=MODEL_NAME)
    batch_converter = base_model.alphabet.get_batch_converter()

    prepare_features(base_model, batch_converter, DATA_ROOT, FEATURE_PATH, TEST_SET)

    mrl_model = RibosomeLoadingPredictionWrapper(lm_type="ProtRNA")
    mrl_model.load_state_dict(torch.load("weights/mrlHead.ckpt", map_location=DEVICE)["state_dict"], strict=False)
    
    datamodule = RibosomeLoadingDataModule(
        data_root=DATA_ROOT,
        feature_path=FEATURE_PATH,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        skip_data_preparation=True,
        lm_type="ProtRNA",
        test_set=TEST_SET
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