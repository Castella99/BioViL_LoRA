# VLP Finetuning - Pneumonia Classification
Dataset : RSNA Pneumonia Classification

## VLP Zero-shot Classification
python -u VLP_Zeroshot.py

## VLP Linear-Probing
python -u VLP_linear_probing.py --epoch=50 --ratio=(0.01, 0.1, 1.0)

## VLP Full-Finetuning
python -u VLP_full_finetuning.py --epoch=50 --ratio=(0.01, 0.1, 1.0)

## VLP LoRA-Finetuning
python -u VLP_LoRA --epoch=50 --ratio=(0.01, 0.1, 1.0) --rank=(4,8)
