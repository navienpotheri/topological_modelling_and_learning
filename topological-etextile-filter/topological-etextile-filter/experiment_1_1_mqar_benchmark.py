import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =====================================================================
# 1. MQAR SYNTHETIC DATASET GENERATOR (L=256, Multi-Key KV Retrieval)
# =====================================================================
def generate_mqar_batch(batch_size=64, seq_len=256, num_kv_pairs=8, vocab_size=512, device='cpu'):
    """
    Generates MQAR sequences:
    - Inserts 'num_kv_pairs' of (Key, Value) tokens scattered in the first half of the sequence.
    - Queries keys in the second half of the sequence; target is the associated Value.
    """
    # Initialize background noise tokens
    inputs = torch.randint(10, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Choose distinct keys and values (keys: 10..250, values: 251..500)
        keys = torch.randperm(200)[:num_kv_pairs] + 10
        vals = torch.randperm(200)[:num_kv_pairs] + 250
        
        # Place key-value pairs in the first half (0 to seq_len//2 - 2)
        kv_positions = torch.randperm(seq_len // 2 - 2)[:num_kv_pairs * 2]
        kv_positions, _ = torch.sort(kv_positions)
        
        kv_map = {}
        for i in range(num_kv_pairs):
            k_pos = kv_positions[2 * i].item()
            v_pos = k_pos + 1
            k_val = keys[i].item()
            v_val = vals[i].item()
            inputs[b, k_pos] = k_val
            inputs[b, v_pos] = v_val
            kv_map[k_val] = v_val

        # Place queries in the second half (seq_len//2 to seq_len - 1)
        query_positions = torch.randperm(seq_len // 2 - 1)[:num_kv_pairs] + (seq_len // 2)
        for i, q_pos in enumerate(query_positions):
            q_k = keys[i].item()
            inputs[b, q_pos] = q_k
            targets[b, q_pos] = kv_map[q_k]

    return inputs, targets

# =====================================================================
# 2. ARCHITECTURES UNDER TEST
# =====================================================================
class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        attn_out, _ = self.mha(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out)

class FastSelectiveSSM(nn.Module):
    def __init__(self, d_model=128, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.x_proj = nn.Linear(d_model, 2 * d_state + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        proj = self.x_proj(x)
        B_gate = proj[:, :, :self.d_state]
        C_gate = proj[:, :, self.d_state:2*self.d_state]
        dt = F.softplus(self.dt_proj(proj[:, :, 2*self.d_state:]))
        A = -torch.exp(self.A_log)
        
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))
        dB = torch.einsum('bld,bln->bldn', dt, B_gate)
        
        # Recurrent scan over sequence length L
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = torch.einsum('bdn,bn->bd', h, C_gate[:, t])
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1)
        return self.norm(x + self.out_proj(y))

class MQARModel(nn.Module):
    def __init__(self, arch_type="attention", vocab_size=512, d_model=128, num_heads=4, d_state=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.arch_type = arch_type
        
        if arch_type == "attention":
            self.layer = CausalAttentionBlock(d_model, num_heads)
        elif arch_type == "ssm":
            self.layer = FastSelectiveSSM(d_model, d_state)
        elif arch_type == "hybrid":
            self.ssm_sublayer = FastSelectiveSSM(d_model, d_state)
            self.attn_sublayer = CausalAttentionBlock(d_model, num_heads)
            self.gate = nn.Parameter(torch.tensor([0.5]))
            
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        if self.arch_type == "attention":
            h = self.layer(h)
        elif self.arch_type == "ssm":
            h = self.layer(h)
        elif self.arch_type == "hybrid":
            h = (1 - self.gate) * self.ssm_sublayer(h) + self.gate * self.attn_sublayer(h)
        return self.head(h)

# =====================================================================
# 3. BENCHMARK EXECUTION (2,000 STEPS)
# =====================================================================
def run_mqar_benchmark(steps=2000, batch_size=64, seq_len=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(f"EXPERIMENT 1.1: MQAR STRESS TEST (L={seq_len}, Steps={steps}, Device={device.type.upper()})")
    print("=" * 80)

    archs = ["attention", "ssm", "hybrid"]
    history = {k: [] for k in archs}
    eval_interval = 200

    for arch in archs:
        print(f"\nTraining Architecture: {arch.upper()} ...")
        model = MQARModel(arch_type=arch, vocab_size=512, d_model=128, num_heads=4, d_state=16).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        t0 = time.time()
        for step in range(1, steps + 1):
            model.train()
            x, y = generate_mqar_batch(batch_size=batch_size, seq_len=seq_len, num_kv_pairs=8, device=device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, 512), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % eval_interval == 0 or step == 1:
                model.eval()
                with torch.no_grad():
                    val_x, val_y = generate_mqar_batch(batch_size=128, seq_len=seq_len, num_kv_pairs=8, device=device)
                    val_logits = model(val_x)
                    preds = val_logits.argmax(dim=-1)
                    mask = (val_y != -100)
                    acc = ((preds == val_y) & mask).sum().item() / mask.sum().item() * 100.0
                    history[arch].append((step, acc))
                    print(f"  Step {step:4d}/{steps} | Loss: {loss.item():.4f} | Recall Acc: {acc:6.2f}%")

        elapsed = time.time() - t0
        print(f"Finished {arch.upper()} in {elapsed:.2f}s")

    # =====================================================================
    # 4. PLOT RECALL CONVERGENCE CURVES
    # =====================================================================
    plt.figure(figsize=(9, 5.5))
    for arch, col, mark in zip(archs, ['crimson', 'teal', 'darkorange'], ['o', 's', '^']):
        steps_eval, accs = zip(*history[arch])
        plt.plot(steps_eval, accs, f'-{mark}', label=f'{arch.capitalize()}', color=col, lw=2, markersize=5)

    plt.axhline(100.0, color='gray', linestyle=':', alpha=0.6)
    plt.title(f"MQAR Benchmark Recall Accuracy (L={seq_len}, 8 KV Pairs)", fontweight='bold')
    plt.xlabel("Training Steps")
    plt.ylabel("Exact Key Retrieval Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("experiment_1_1_mqar_results.png", dpi=300)
    print("\nBenchmark complete. Saved curve plot to 'experiment_1_1_mqar_results.png'.")
    plt.show()

if __name__ == "__main__":
    run_mqar_benchmark(steps=2000, batch_size=64, seq_len=256)