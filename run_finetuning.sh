#python3 -m src.finetune --finetuning-mode=(standard/linear) --model=(model_name) --world-size=(no._of_gpus) --train-data=(path_to_tokenized_dataset) --epochs (num_of_epochs) --num-labels (num_of_labels) --task (classification/ner/causal) --data-task (data_task_name)
#Example: (linear finetune on cola)
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/cola/cola_train.pt --epochs 4 --num-labels 2 --task classification --data-task cola
#Example: (standard/non-linear finetune on cola)
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/cola/cola_train.pt --epochs 4 --num-labels 2 --task classification --data-task cola

#NOTE: Modify this shell script and add python commands for multiple finetuning. Run this script as background)
