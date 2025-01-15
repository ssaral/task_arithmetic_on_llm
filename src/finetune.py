import os
import time
import torch
import torch.nn as nn
from transformers import AdamW, get_scheduler, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForCausalLM, AutoTokenizer
from src.linearize import LinearizedModel, LinearizedLM
from src.distributed import setup_ddp, distribute_loader, cleanup_ddp, is_main_process
from src.utils import LabelSmoothing
import wandb

# import generate_sentiment_data

def finetune(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    # tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    assert args.finetuning_mode in ["linear", "standard"], \
        "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    temp = args.model + "_" + args.finetuning_mode + "_" + args.task + "_" + args.data_task
    ckpdir = os.path.join(args.save, temp)
    os.makedirs(ckpdir, exist_ok=True)

    # wandb.init(
    #     project="gpt2-finetune",  # Replace with your project name
    #     entity="ssaral-india",
    #     config=args,  # Log hyperparameters from argparse
    #     name=f"finetuning-{args.finetuning_mode}",  # Name of the experiment
    #     dir=args.save  # Specify where logs are saved
    # )


    # Load GPT-2 model
    if args.load:
        print(f"Loading model from {args.load}")
        model = torch.load(args.load, map_location="cpu")
    else:
        print("Building GPT-2 model.")
        if args.task == "classification":
            print("Loading classification model.")
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
        elif args.task == "ner":
            print("Loading token classification model for NER.")
            model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=args.num_labels)
        else:
            print("Loading GPT-2 model for summarization.") 
            model = AutoModelForCausalLM.from_pretrained(args.model)
            # model = GPT2LMHeadModel.from_pretrained(args.model)
    
    model.config.pad_token_id = tokenizer.pad_token_id

    if linearized_finetuning:
        # model = LinearizedLM(model)
        model = LinearizedModel(model)

    model = model.cuda(rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    train_dataset = torch.load(args.train_data)  #Tokenized dataset is pre-saved
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: tokenizer.pad(x, return_tensors='pt')
    )
    ddp_loader = distribute_loader(train_loader)

    # Loss function
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        # loss_fn = nn.CrossEntropyLoss()

    # Optimizer and Scheduler
    optimizer = AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=args.warmup_steps,
                              num_training_steps=args.epochs * len(train_loader))
    
    # Save initial (zero-shot) model
    if is_main_process():
        if args.finetuning_mode == "linear":
            zs_path = os.path.join(ckpdir, "linear_zeroshot.pt")
            torch.save(ddp_model.module, zs_path)
            # torch.save(ddp_model.module.state_dict(), zs_path)
        else:
            zs_path = os.path.join(ckpdir, "zeroshot_full_model.pt")
            torch.save(ddp_model.module, zs_path)
        

    print_every = 100
    for epoch in range(args.epochs):
        ddp_model.train()
        for step, batch in enumerate(ddp_loader):
            start_time = time.time()

            # step = (
            #     i // args.num_grad_accumulation
            #     + epoch * num_batches // args.num_grad_accumulation
            # )

            inputs, labels = batch["input_ids"].cuda(rank), batch["labels"].cuda(rank) 
            data_time = time.time() - start_time 

            optimizer.zero_grad()

            if args.finetuning_mode == "linear":
                logits = ddp_model(inputs)
            else:
                #setting up for non-linear
                outputs = ddp_model(inputs)
                logits = outputs.logits

            #setting up for non-linear 
            if args.task != "classification":
                logits = logits[:, :labels.size(1), :] 
                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = labels.reshape(-1)  
                loss = loss_fn(logits_flat, labels_flat)
            else:
                detokenized_labels = [tokenizer.decode(label.item()) for label in labels]
                detokenized_labels = torch.tensor([int(label.strip()) for label in detokenized_labels], dtype=torch.long)
                detokenized_labels = detokenized_labels.to(logits.device)
                loss = loss_fn(logits, detokenized_labels)


            loss.backward()

            optimizer.step()
            scheduler.step()

            if step % print_every == 0 and is_main_process():
                # wandb.log({
                #     "epoch": epoch,
                #     "step": step,
                #     "loss": loss.item(),
                #     "learning_rate": optimizer.param_groups[0]["lr"],
                #     "time_per_step": time.time() - start_time,
                # })
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

            if step % args.checkpoint_every == 0 and is_main_process():
                if args.finetuning_mode == "linear":
                    ft_path = os.path.join(ckpdir, f"checkpoint_{epoch}_{step}.pt")
                    torch.save(ddp_model.module, ft_path)
                    # torch.save(ddp_model.module.state_dict(), ft_path)
                else:
                    ft_path = os.path.join(ckpdir, f"checkpoint_full_model_{epoch}_{step}.pt")
                    torch.save(ddp_model.module, ft_path)
                
                # wandb.save(ft_path)

    # Save final finetuned model
    if is_main_process():
        if args.finetuning_mode == "linear":
            ft_path = os.path.join(ckpdir, "linear_finetuned.pt")
            torch.save(ddp_model.module, ft_path)
            # torch.save(ddp_model.module.state_dict(), ft_path)
        else:
            ft_path = os.path.join(ckpdir, "finetuned_full_model.pt")
            torch.save(ddp_model.module, ft_path)
        
        # wandb.save(ft_path)
    
    # wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    import argparse

    print("To get training data for sentiment analysis: execute generate_sentiment_data.py file. If executed, then ignore.")

    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 model")
    parser.add_argument("--task", type=str, choices=["summarization", "classification", "ner"], required=True, help="Task type")
    parser.add_argument("--model", type=str, required=True, help="GPT-2 model size (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--train-data", type=str, required=True, help="Path to tokenized training dataset")
    parser.add_argument("--num-labels", type=int, default=2, help="Number of labels for classification or NER tasks")
    parser.add_argument("--finetuning-mode", type=str, choices=["linear", "standard"], required=True)
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps for scheduler")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0, help="Label smoothing")
    parser.add_argument("--save", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Checkpoint frequency (steps)")
    parser.add_argument("--load", type=str, help="Path to load pretrained model")
    parser.add_argument("--world-size", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--port", type=str, default="12345", help="Port for DDP communication")
    parser.add_argument("--data-task", type=str, default="", help="Dataset name of the task. Will be used in saving checkpoints with proper naming convention.")

    args = parser.parse_args()

    print(f"Starting fine-tuning for GPT-2 in {args.finetuning_mode} mode.")
    torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size) 
