import torch
import numpy as np
from gpt import GPTModel
from construct_dataset import construct_dataset
import matplotlib.pyplot as plt

# since we didn't really cover how to do this in lecture-
# this creates a learning rate schedule for you. Refer to the 
# pytorch docs for more info on using a scheduler.

# This one is designed for you to call scheduler.step() on every
# model update step. 
def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*0.9 + 0.1
        return max(lrmult, 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)
    return scheduler

# ===========================================================================

'''
Complete the following method which trains a GPT model and saves a loss curve.

To reiterate: you don't need to worry about weight decay, weight initialization, grad accumulation, or weight tying.
Use whatever batch size you are able, even something like 2 or 4 is fine.
Use a few hundred warmup steps and a peak learning rate that is (something x 10-4).
'''
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE    = 8      
    PEAK_LR       = 5e-4   
    WARMUP_STEPS  = 15     
    SEQUENCE_LEN  = 256
    PLOT_INTERVAL = 20
    SUBSET_FRAC   = 1/20   

    # Model 
    model = GPTModel(d_model=256, n_heads=8, layers=4, vocab_size=10000, max_seq_len=SEQUENCE_LEN)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model has {param_count:,} parameters.")
    model = model.to(device)

    # Dataset
    data_np = construct_dataset("./data.txt", sequence_length=SEQUENCE_LEN)
    data = torch.tensor(data_np, dtype=torch.long)
    num_sequences = data.shape[0]

    subset_size = max(1, int(num_sequences * SUBSET_FRAC))
    perm = torch.randperm(num_sequences)
    data = data[perm[:subset_size]]
    num_sequences = data.shape[0]
    print(f"Using {num_sequences} sequences (1/20 of full dataset, length {data.shape[1]})")

    total_steps = max(1, num_sequences // BATCH_SIZE)

    # Optimizer & scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8)
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps, WARMUP_STEPS)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    tokens_seen = 0
    loss_log_tokens = []
    loss_log_values = []

    step = 0
    for batch_start in range(0, num_sequences - BATCH_SIZE + 1, BATCH_SIZE):

        batch = data[batch_start : batch_start + BATCH_SIZE].to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        opt.zero_grad()
        logits = model(x)
        logits_for_loss = logits.transpose(1, 2)
        loss = loss_fn(logits_for_loss, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        tokens_seen += BATCH_SIZE * SEQUENCE_LEN
        loss_log_tokens.append(tokens_seen)
        loss_log_values.append(loss.item())

        if step % 10 == 0:
            print(f"Step {step:4d}/{total_steps} | "
                  f"loss: {loss.item():.4f} | "
                  f"lr: {scheduler.get_last_lr()[0]*PEAK_LR:.2e} | "
                  f"tokens: {tokens_seen:,}")

        if step % PLOT_INTERVAL == 0 and step > 0:
            _save_loss_plot(loss_log_tokens, loss_log_values)

        step += 1

    # Final plot & model save
    _save_loss_plot(loss_log_tokens, loss_log_values)
    torch.save(model.state_dict(), "./model_weights.pt")

def _save_loss_plot(tokens, losses):
    plt.figure(figsize=(8, 4))
    plt.plot(tokens, losses, linewidth=0.8, alpha=0.7, label="loss")

    # Overlay a smoothed trend 
    if len(losses) >= 20:
        window = max(1, len(losses) // 20)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        smoothed_tokens = tokens[window - 1:]
        plt.plot(smoothed_tokens, smoothed, linewidth=2, label=f"smoothed (w={window})")

    plt.xlabel("Tokens seen")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()



if __name__ == "__main__":
    train()



