# CXRBert FineTuning - Natural Language Inference

Dataset : MedNLI (Train), RadNLI(Test)

## CXRBert Full-Finetuning
python -u CXRBert_FFT.py --lr=0.0005 --epoch=20

## CXRBert LoRA-Finetuning
python -u CXRBert_LoRA.py --lr=0.0005 --epoch=20 --rank=8
