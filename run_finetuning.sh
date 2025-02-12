#NOTE: Modify this shell script and add python commands for multiple finetuning. Run this script as background)

########################## STANDARD FINETUNING ##########################

echo "Starting cola finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/cola/cola_train.pt --epochs 2 --num-labels 2 --task classification --data-task cola 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting cr finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/cr/cr_train.pt --epochs 2 --num-labels 2 --task classification --data-task cr 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting mpqa finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/mpqa/mpqa_train.pt --epochs 2 --num-labels 2 --task classification --data-task mpqa 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting mr finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/mr/mr_train.pt --epochs 2 --num-labels 2 --task classification --data-task mr 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting mrpc finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/mrpc/mrpc_train.pt --epochs 2 --num-labels 2 --task classification --data-task mrpc 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting qnli finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/qnli/qnli_train.pt --epochs 2 --num-labels 2 --task classification --data-task qnli 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting qqp finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/qqp/qqp_train.pt --epochs 2 --num-labels 2 --task classification --data-task qqp 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting rte finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/rte/rte_train.pt --epochs 2 --num-labels 2 --task classification --data-task rte 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting snli finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/snli/snli_train.pt --epochs 2 --num-labels 3 --task classification --data-task snli 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting wnli finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/wnli/wnli_train.pt --epochs 2 --num-labels 2 --task classification --data-task wnli 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting sst2 finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/sst2/sst2_train.pt --epochs 2 --num-labels 2 --task classification --data-task sst2 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting sst5 finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/sst5/sst5_train.pt --epochs 2 --num-labels 5 --task classification --data-task sst5 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting subj finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/subj/subj_train.pt --epochs 2 --num-labels 2 --task classification --data-task subj 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting mnli finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/mnli/mnli_train.pt --epochs 2 --num-labels 3 --task classification --data-task mnli 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting trec finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/trec/trec_train.pt --epochs 2 --num-labels 6 --task classification --data-task trec 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting axb finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/axb/axb_train.pt --epochs 2 --num-labels 2 --task classification --data-task axb 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting axg finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/axg/axg_train.pt --epochs 2 --num-labels 2 --task classification --data-task axg 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting boolq finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/boolq/boolq_train.pt --epochs 2 --num-labels 2 --task classification --data-task boolq 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting cb finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/cb/cb_train.pt --epochs 2 --num-labels 3 --task classification --data-task cb 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting copa finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/copa/copa_train.pt --epochs 2 --num-labels 2 --task classification --data-task copa 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting pawsx finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/pawsx/pawsx_train.pt --epochs 2 --num-labels 2 --task classification --data-task pawsx 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting xnli finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/xnli/xnli_train.pt --epochs 2 --num-labels 3 --task classification --data-task xnli 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting xwinograd finetuning"
python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/xwinograd/xwinograd_train.pt --epochs 2 --num-labels 2 --task classification --data-task xwinograd 
echo "Sleep timer of 30 secs"
sleep 30

########################## LINEAR FINETUNING ##########################

echo "Starting cola finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/cola/cola_train.pt --epochs 2 --num-labels 2 --task classification --data-task cola 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting cr finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/cr/cr_train.pt --epochs 2 --num-labels 2 --task classification --data-task cr 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting mpqa finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/mpqa/mpqa_train.pt --epochs 2 --num-labels 2 --task classification --data-task mpqa 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting mr finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/mr/mr_train.pt --epochs 2 --num-labels 2 --task classification --data-task mr 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting mrpc finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/mrpc/mrpc_train.pt --epochs 2 --num-labels 2 --task classification --data-task mrpc 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting qnli finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/qnli/qnli_train.pt --epochs 2 --num-labels 2 --task classification --data-task qnli 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting qqp finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/qqp/qqp_train.pt --epochs 2 --num-labels 2 --task classification --data-task qqp 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting rte finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/rte/rte_train.pt --epochs 2 --num-labels 2 --task classification --data-task rte 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting snli finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/snli/snli_train.pt --epochs 2 --num-labels 3 --task classification --data-task snli 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting wnli finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/wnli/wnli_train.pt --epochs 2 --num-labels 2 --task classification --data-task wnli 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting sst2 finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/sst2/sst2_train.pt --epochs 2 --num-labels 2 --task classification --data-task sst2 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting sst5 finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/sst5/sst5_train.pt --epochs 2 --num-labels 5 --task classification --data-task sst5 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting subj finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/subj/subj_train.pt --epochs 2 --num-labels 2 --task classification --data-task subj 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting mnli finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/mnli/mnli_train.pt --epochs 2 --num-labels 3 --task classification --data-task mnli 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting trec finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/trec/trec_train.pt --epochs 2 --num-labels 6 --task classification --data-task trec 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting axb finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/axb/axb_train.pt --epochs 2 --num-labels 2 --task classification --data-task axb 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting axg finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/axg/axg_train.pt --epochs 2 --num-labels 2 --task classification --data-task axg 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting boolq finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/boolq/boolq_train.pt --epochs 2 --num-labels 2 --task classification --data-task boolq 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting cb finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/cb/cb_train.pt --epochs 2 --num-labels 3 --task classification --data-task cb 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting copa finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/copa/copa_train.pt --epochs 2 --num-labels 2 --task classification --data-task copa 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting pawsx finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/pawsx/pawsx_train.pt --epochs 2 --num-labels 2 --task classification --data-task pawsx 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting xnli finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/xnli/xnli_train.pt --epochs 2 --num-labels 3 --task classification --data-task xnli 
echo "Sleep timer of 30 secs"
sleep 30
echo "Starting xwinograd finetuning"
python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/xwinograd/xwinograd_train.pt --epochs 2 --num-labels 2 --task classification --data-task xwinograd 
echo "Sleep timer of 30 secs"
sleep 30




#Summarization pipeline
# python3 -m src.finetune --finetuning-mode=standard --model=gpt2 --world-size=2 --train-data=data/summary_test_data_tPAD.pt --epochs 3 --task summarization
# python3 -m src.finetune --finetuning-mode=linear --model=gpt2 --world-size=2 --train-data=data/summary_test_data_tPAD.pt --epochs 3 --task summarization

#Evaluation on single task
# python3 -m src.eval_single_task --model "gpt2" --save . --finetuning-mode "standard" --max-length 128

#Evaluation of task addition
# python3 -m src.eval_task_addition --model gpt2 --finetuning-mode standard

#EVALUTION FOR SINGLE TASK
# python3 -m src.eval_single_task --model gpt2 --save . --finetuning-mode standard --task classification --data-task cr --num-labels 2 --world-size 2
# python3 -m src.eval_single_task --model gpt2 --save . --finetuning-mode linear --task classification --data-task cr --num-labels 2 --world-size 2

#Evaluation for TASK Addition
# python3 -m src.eval_task_addition --model gpt2 --finetuning-mode standard --task classification --data-task cola

#Calculating Task similarity
# python cal_task_similarity.py --task classification --finetuning-mode linear --model gpt2 
