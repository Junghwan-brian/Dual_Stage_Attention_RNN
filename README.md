# Dual_Stage_Attention_RNN
## DARNN

Paper : https://arxiv.org/abs/1704.02971

## Requirements
- python 3.7
- Tensorflow 2.3
- numpy
- pandas

## Dataset
Dataset : https://dacon.io/competitions/official/235584/overview/
Data preprocessing : preprocess.py

Data shape : inputs : [enc_data , dec_data]
             enc_data shape : batch,T,n
             dec_data shape: batch,T-1,1

## implementation : Dual_stage_attention_model.py
Architecture
![image](https://user-images.githubusercontent.com/46440177/99161928-d3353c80-273a-11eb-8891-a44ed7270fca.png)

## Result
![image](https://user-images.githubusercontent.com/46440177/99161936-011a8100-273b-11eb-938a-416dd129ce91.png)
![image](https://user-images.githubusercontent.com/46440177/99161949-1d1e2280-273b-11eb-847f-24b43c5655f9.png)
![image](https://user-images.githubusercontent.com/46440177/99161951-24ddc700-273b-11eb-8d2a-c1ea931dd153.png)

