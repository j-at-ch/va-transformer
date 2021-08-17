job_ref=$(date '+%y%m%d%H%M')

python mimic/pretraining.py \
    --data_root /home/james/Documents/Charters/labevents_2days/data \
    --save_root /home/james/Documents/Charters/labevents_2days/results \
    --logs_root /home/james/Documents/Charters/labevents_2days/logs \
    --attn_depth 6 \
    --attn_dim 100 \
    --attn_heads 8 \
    --seq_len 200 \
    --model_name "pretraining_labevents_2days_${job_ref}" \
    --num_epochs 50 \
    --device "cuda:0"