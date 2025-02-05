#!/bin/bash

# ./test_downstream_rbp.sh WTAP_Hela

d=clip_data
f=features

exp_name=prismnet_Hela_eval
mode=lm1280_red

p=$1
# rank=$2

python -m downstream_rbp.tools.generate_dataset $p 1 5 downstream_rbp/data/$d
python -m downstream_rbp.tools.generate_features $p 128 downstream_rbp/data/$d downstream_rbp/data/$f

work_path=$(dirname $0)
exp=$(basename $work_path)

python -m downstream_rbp.tools.main \
    --load_best \
    --eval \
    --data_dir downstream_rbp/data/$f \
    --p_name $p \
    --out_dir downstream_rbp/exp/$exp_name \
    --mode $mode \
    --exp_name $exp_name \
    # --device $rank \
    ${@:6}