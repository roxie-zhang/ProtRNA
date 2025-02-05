import torch
import lightning.pytorch as pl

from pretrained import load_pretrained_model
from downstream_mrl.model import RibosomeLoadingPredictionWrapper
from downstream_mrl.utils.datamodule import RibosomeLoadingDataModule
from downstream_mrl.utils.alphabet import Alphabet

MODEL_NAME = "ProtRNA_pretrained"
DEVICE = "cuda"

if __name__ == '__main__':
    base_model = load_pretrained_model(name=MODEL_NAME)
    batch_converter = base_model.alphabet.get_batch_converter()
    
    mrl_model = RibosomeLoadingPredictionWrapper(lm_type='ProtRNA')
    mrl_model.load_state_dict(torch.load("weights/mrlHead.ckpt", map_location=DEVICE), strict=False)
    
    alphabet = Alphabet()
    datamodule = RibosomeLoadingDataModule(
        data_root="downstream_mrl/dataset",
        alphabet=alphabet,
        feature_path="downstream_mrl/dataset/features_10samples",
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        skip_data_preparation=True,
        lm_type="ProtRNA",
        test_set="random"
    )
    
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_steps=-1,
        max_epochs=50,
        gradient_clip_val=None,
        precision="16-mixed",
        default_root_dir="downstream_mrl/output",
        log_every_n_steps=50,
        strategy="auto",
    )
    
    trainer.test(model=mrl_model, datamodule=datamodule)