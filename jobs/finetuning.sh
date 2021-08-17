job_ref=$(date '+%y%m%d%H%M')

python mimic/finetuning.py \
    --data_root /home/james/Documents/Charters/labevents_2days/data \
    --save_root /home/james/Documents/Charters/labevents_2days/results \
    --logs_root /home/james/Documents/Charters/labevents_2days/logs \
    --model_root /home/james/Documents/Charters/labevents_2days/models \
    --attn_depth 6 \
    --attn_dim 100 \
    --attn_heads 8 \
    --seq_len 200 \
    --learning_rate 5e-5 \
    --model_name "finetuning_labevents_${job_ref}" \
    --num_epochs 50 \
    --pretuned_model 'pretraining_labevents_2days_2108161643.pt' \
    --weighted_loss 'True' \
    --ff_dropout 0.1 \
    --label_set 'DEATH>2.5D'