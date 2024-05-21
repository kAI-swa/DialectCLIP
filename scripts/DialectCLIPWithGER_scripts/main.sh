python main.py --dataset Uyghur \
    --speech_model openai/whisper-medium \
    --language_model meta-llama/Meta-Llama-3-8B-Instruct \
    --sampling_rate 16000 \
    --logit_scale_init_value 2.6592 \
    --initializer_range 0.02 \
    --lora_rank 32 \
    --lora_dropout 0.01 \
    --tau 1.0 \
    --lora_alpha 8 \
    --alpha 0.1 \
    --beta 0.5 \
    --device cuda \
    --epochs 3 \
    --batch_size 16 \
    --shuffle \
    --num_workers 8 \
    --learning_rate 0.0001 \
    --weight_decay_rate 0.001 \
    --model_save_path ./checkpoint/dialect_clip.pth \
    --save_checkpoint_frequency 20 \
    --num_beams 1 \
    --max_length 128 \

