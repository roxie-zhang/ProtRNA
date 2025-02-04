#!/bin/bash

work_path=$(dirname $0)
name=$(basename $work_path)
# echo `date +%Y%m%d%H%M%S`

p_name=$1
data=$2
mode=$3
rank=$4

exp=$name

# exp/prismnet_Hela_eval/eval.sh TIA1_Hela features_6e lm1280_red 0
python -m tools.main \
    --load_best \
    --eval \
    --data_dir data/$data \
    --p_name $p_name \
    --out_dir $work_path \
    --mode $mode \
    --exp_name $exp \
    --device $rank \
    ${@:6}
    # | tee $work_path/out/${p_name}_${mode}.txt
