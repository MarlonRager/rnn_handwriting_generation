#!/bin/bash
folder="/home/azureuser/v2/rnn_handwriting_generation/res"
for i in {1..200}
  python3.8 train.py --folder $folder --batch_size 250 --learning_rate 0.002 --epoch $i
  python3.8 sample.py --folder $folder --epoch $i
done