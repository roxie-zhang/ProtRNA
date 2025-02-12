#!/bin/bash
# exp/prismnet_Hela_eval/eval_all.sh Hela features lm1280_red 0
work_path=$(dirname $0)
name=$(basename $work_path)

cell=$1
da=$2
mode=$3
rank=$4

mkdir -p $work_path/out
mkdir -p $work_path/out/log
mkdir -p $work_path/out/evals
mkdir -p $work_path/out/models

# N threads according to your GPU
SEND_THREAD_NUM=8 # 16 for pu, seq; 8 for lm1280, lm640

###########################

tmp_fifofile="/tmp/$$.fifo"
mkfifo "$tmp_fifofile"
exec 6<>"$tmp_fifofile"
for ((i=0;i<$SEND_THREAD_NUM;i++));do
                 echo                                                                                    
done >&6


for p in `cat  data/${cell}.list`
do 
    read -u6
    {
    id=${p}_PrismNet_${mode}
    ff=$work_path/out/evals/${id}.metrics
    lg=$work_path/out/log/${id}.log
    if [ ! -f $ff ] ; then 
        echo ${p}" ==="
        $srun $work_path/eval.sh $p $da $mode $rank> $lg
    fi
    sleep 1
    echo >&6
    } &
    pid=$!
    echo $pid
done

wait
exec 6>&-
exit 0

