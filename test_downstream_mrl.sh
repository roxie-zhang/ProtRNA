python downstream_mrl/codes/train_ribosome_loading.py \
downstream_mrl/data \
--init_params downstream_mrl/weights/ProtRNA.ckpt \
--lm_type ProtRNA \
--test_set human \
--feature_path downstream_mrl/feature \ 
--batch_size 64 -test_only --num_workers 32 --pin_memory --accelerator gpu --max_epochs 50 \
--output_dir downstream_mrl/output

python downstream_mrl/codes/train_ribosome_loading.py \
downstream_mrl/data \
--init_params downstream_mrl/weights/ProtRNA.ckpt \
--lm_type ProtRNA \
--test_set random \
--feature_path downstream_mrl/feature \ 
--batch_size 64 -test_only --num_workers 32 --pin_memory --accelerator gpu --max_epochs 50 \
--output_dir downstream_mrl/output