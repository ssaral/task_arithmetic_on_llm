import os
import time
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler
from src.linearize import LinearizedModel
from src.distributed import setup_ddp, distribute_loader, cleanup_ddp, is_main_process
from src.utils import LabelSmoothing
import wandb

def finetune(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    assert args.finetuning_mode in ["linear", "standard"], \
        "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    ckpdir = os.path.join(args.save, "checkpoints")
    os.makedirs(ckpdir, exist_ok=True)

    wandb.init(
        project="gpt2-finetune",  # Replace with your project name
        entity="ssaral-india",
        config=args,  # Log hyperparameters from argparse
        name=f"finetuning-{args.finetuning_mode}",  # Name of the experiment
        dir=args.save  # Specify where logs are saved
    )

    # Load GPT-2 model
    if args.load:
        print(f"Loading model from {args.load}")
        model = torch.load(args.load, map_location="cpu")
    else:
        print("Building GPT-2 model.")
        model = GPT2LMHeadModel.from_pretrained(args.model)

    if linearized_finetuning:
        model = LinearizedModel(model)

    model = model.cuda(rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    train_dataset = torch.load(args.train_data)  # Assume tokenized dataset is pre-saved
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: tokenizer.pad(x, return_tensors='pt')
    )
    ddp_loader = distribute_loader(train_loader)

    # Loss function
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Optimizer and Scheduler
    optimizer = AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=args.warmup_steps,
                              num_training_steps=args.epochs * len(train_loader))

    # Save initial (zero-shot) model
    if is_main_process():
        zs_path = os.path.join(ckpdir, "zeroshot.pt")
        torch.save(ddp_model.module.state_dict(), zs_path)

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

            print("====================INPUT=================================")
            print(type(inputs))
            print("inputs.shape:", inputs.shape)

            optimizer.zero_grad()
            logits = ddp_model(inputs)

            print("=====================LOGIT================================")
            print("Logits", logits)
            print("Logits shape", logits.shape)

            print("====================LABEL=================================")
            print("labels", labels)
            print("Labels shape", labels.shape)

            loss = loss_fn(logits, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()

            if step % print_every == 0 and is_main_process():
                wandb.log({
                    "epoch": epoch,
                    "step": step,
                    "loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "time_per_step": time.time() - start_time,
                })
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

            if step % args.checkpoint_every == 0 and is_main_process():
                ft_path = os.path.join(ckpdir, f"checkpoint_{epoch}_{step}.pt")
                torch.save(ddp_model.module.state_dict(), ft_path)
                wandb.save(ft_path)

    # Save final finetuned model
    if is_main_process():
        ft_path = os.path.join(ckpdir, "finetuned.pt")
        torch.save(ddp_model.module.state_dict(), ft_path)
        wandb.save(ft_path)
    
    wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 model")
    parser.add_argument("--model", type=str, required=True, help="GPT-2 model size (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--train-data", type=str, required=True, help="Path to tokenized training dataset")
    parser.add_argument("--finetuning-mode", type=str, choices=["linear", "standard"], required=True)
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps for scheduler")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0, help="Label smoothing")
    parser.add_argument("--save", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Checkpoint frequency (steps)")
    parser.add_argument("--load", type=str, help="Path to load pretrained model")
    parser.add_argument("--world-size", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--port", type=str, default="12345", help="Port for DDP communication")

    args = parser.parse_args()

    print(f"Starting fine-tuning for GPT-2 in {args.finetuning_mode} mode.")
    torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
