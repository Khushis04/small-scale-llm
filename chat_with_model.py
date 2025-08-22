import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
import math

# ===================== Model & Config =====================
class Config:
    vocab_size = 5000      
    n_positions = 256
    n_ctx = 256
    n_embd = 256
    n_layer = 4
    n_head = 4
    dropout = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        def reshape_heads(tensor):
            return tensor.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        if attn_mask is not None:
            att = att + attn_mask

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y

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

class TinyGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)
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
        tok_embeddings = self.tok_emb(idx)
        pos_ids = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_embeddings = self.pos_emb(pos_ids)
        x = self.drop(tok_embeddings + pos_embeddings)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ===================== Chat Generation =====================
@torch.no_grad()
def chat(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=50):
    input_ids = tokenizer.encode(prompt).ids
    ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(next(model.parameters()).device)

    for _ in range(max_new_tokens):
        if ids.size(1) > model.config.n_positions:
            ids = ids[:, -model.config.n_positions:]

        logits = model(ids)
        logits = logits[:, -1, :] / max(1e-9, temperature)
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        min_top = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top, torch.full_like(logits, -1e9), logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(ids[0].cpu().tolist())

# ===================== Main Chatbot =====================
def main():
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=r"scratch_tokenizer\tokenizer.json",
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        mask_token="<mask>"
    )

    # Load model
    config = Config()
    config.vocab_size = tokenizer.get_vocab_size()
    model = TinyGPT(config)
    model.head.weight = model.tok_emb.weight

    checkpoint = torch.load("./ckpt/gpt_step5000.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Start Chat~")
    print("Type 'quit' to exit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit"]:
            break
        response = chat(model, tokenizer, prompt, max_new_tokens=50)
        print("Bot:", response)

if __name__ == "__main__":
    main()
