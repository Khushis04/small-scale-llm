import math
import os
import time
import random
import argparse
from typing import List 
from datasets import load_dataset
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

class Config:
    vocab_size = 5000      
    n_positions = 256
    n_ctx = 256
    n_embd = 256
    n_layer = 4
    n_head = 4
    dropout = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("OpenAssistant/oasst2", split="train").select(range(20000))

texts = []
for item in dataset:
    # Check which key exists
    if "conversations" in item:
        for turn in item["conversations"]:
            texts.append(turn["value"].strip())
    elif "text" in item:
        texts.append(item["text"].strip())

print("Number of training texts:", len(texts))

#self-attention
class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, attn_mask=None):
        #x : B, T, C
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  #B, T, 3*C
        q, k, v = qkv.split(C, dim=2)  #B, T, C

        #reshape for multi-head : B, n_head, T, head_dim
        def reshape_heads(tensor):
            return tensor.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        att = torch.matmul(q, k.transpose(-2, -1))  # (B, nh, T, T)
        att = att * self.scale

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        if attn_mask is not None:
            att = att + attn_mask

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)  #B, nh, T, hd
        y = y.transpose(1, 2).contiguous().view(B, T, C)  #B, T, C
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y

# Feed-forward (MLP)
class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Transformer
class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# GPT Model
class TinyGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)

        #weights for token embedding
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.config = config
        self.register_buffer("placeholder", torch.tensor(1))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.n_positions, "Sequence length exceeds model n_positions"

        tok_embeddings = self.tok_emb(idx)  #B, T, C
        pos_ids = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_embeddings = self.pos_emb(pos_ids)  #1, T, C

        x = self.drop(tok_embeddings + pos_embeddings)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  #B, T, C
        logits = self.head(x)  #B, T, V
        return logits

#Dataset wrapper
class TextDataset(Dataset):
    def __init__(self, token_ids: List[List[int]], block_size: int, pad_token_id: int = 0):
        """
        token_ids: list of token id lists (each is a sequence)
        block_size: sequence length for model
        pad_token_id: token used for padding shorter sequences
        """
        self.examples = []

        for seq in token_ids:
            # Pad sequence if shorter than block_size
            if len(seq) < block_size:
                seq = seq + [pad_token_id] * (block_size - len(seq))
            
            # Split into chunks of block_size
            for i in range(0, len(seq), block_size):
                chunk = seq[i : i + block_size]

                # If last chunk is shorter, pad it
                if len(chunk) < block_size:
                    chunk = chunk + [pad_token_id] * (block_size - len(chunk))

                self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        x = torch.tensor(self.examples[i], dtype=torch.long)
        return x


def train_tokenizer_from_text(corpus: List[str], vocab_size=5000):
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(corpus, vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.save_model("scratch_tokenizer")
    return tokenizer

def batchify_tokenizer_texts(texts, tokenizer, block_size):
    token_id_lists = []
    for t in texts:
        ids = tokenizer.encode(t).ids   
        token_id_lists.append(ids)
    return token_id_lists

#Training
def train(model: nn.Module, dataloader: DataLoader, optimizer, config: Config, steps: int, grad_accum_steps: int=1, save_every: int=500, out_dir="./ckpt"):
    model.train()
    device = config.device
    model.to(device)
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))

    total_steps = steps
    step = 0
    epoch = 0
    it = iter(dataloader)

    while step < total_steps:
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            it = iter(dataloader)
            batch = next(it)

        # batch: (B, T)
        batch = batch.to(device)
        B, T = batch.size()

        # inputs and targets for next-token prediction
        inputs = batch[:, :-1]   # (B, T-1)
        targets = batch[:, 1:]   # (B, T-1)

        with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
            logits = model(inputs)  # (B, T-1, V)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if step % 50 == 0:
            print(f"[step {step}/{total_steps}] loss={loss.item()*grad_accum_steps:.4f}")

        start_time = time.time()
        if (step + 1) % save_every == 0 or (step + 1) == total_steps:
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"gpt_step{step+1}.pt")
            torch.save({"model_state": model.state_dict(), "config": model.config.__dict__}, save_path)
            print("Saved checkpoint", save_path)

        step += 1

    print("Training complete.")

@torch.no_grad()
def generate(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = 50):
    model.eval()
    device = next(model.parameters()).device

    # Encode with .encode().ids
    input_ids = tokenizer.encode(prompt).ids
    ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        if ids.size(1) > model.config.n_positions:
            ids = ids[:, -model.config.n_positions:]

        logits = model(ids)  # (1, T, V)
        logits = logits[:, -1, :] / max(1e-9, temperature)

        # Top-k filtering
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        min_top = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top, torch.full_like(logits, -1e9), logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    # Decode with .decode()
    return tokenizer.decode(ids[0].cpu().tolist())

def main():
    dataset = load_dataset("OpenAssistant/oasst2")

    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    config = Config()

    tokenizer = train_tokenizer_from_text(texts, vocab_size=5000)

    print("Vocab size:", tokenizer.get_vocab_size())
    config.vocab_size = tokenizer.get_vocab_size() 

    token_id_lists = batchify_tokenizer_texts(texts, tokenizer, block_size=256)
    all_tokens = sum(token_id_lists, [])
    dataset = TextDataset([all_tokens], block_size=256)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    #model
    config.n_embd = 256
    config.n_layer = 4
    config.n_head = 4
    config.vocab_size = tokenizer.get_vocab_size()
    model = TinyGPT(config)

    #tie weights: head to token embedding
    model.head.weight = model.tok_emb.weight

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    train(model, loader, optimizer, config, steps=args.max_steps, grad_accum_steps=args.grad_accum, save_every=args.save_every)

    #generation demo
    out = generate(
        model,
        tokenizer,
        prompt="Once upon a time",
        max_new_tokens=200,
        temperature=0.8,
        top_k=50
    )
    print("=== SAMPLE ===")
    print(out)

if __name__ == "__main__":
    main()
