#!/bin/bash
d=clip_data
f=features
for p in `cat data/Hela.list`
do 
    python -m tools.generate_dataset $p 1 5 data/$d
    python -m tools.generate_features $p 128 data/$d data/$f
done
