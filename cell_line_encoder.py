from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
"""
删减了代码，只留下了基础门控编码器
"""
class ProperGatedMultiOmicsEncoder(nn.Module):
    def __init__(self, num_genes=4079, num_omics=6, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.num_omics = num_omics

        # 共享 encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # 正经 gate：每模态 → 标量 score
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        # x: (batch, genes, omics)
        h = []
        gate_scores = []

        for i in range(self.num_omics):
            hi = self.encoder(x[:, :, i])        # (batch, hidden)
            h.append(hi)
            gate_scores.append(self.gate_net(hi))  # (batch, 1)

        h = torch.stack(h, dim=1)                 # (batch, M, hidden)
        gate_scores = torch.stack(gate_scores, dim=1).squeeze(-1)  # (batch, M)

        gate = torch.softmax(gate_scores, dim=-1)

        # uniform residual
        uniform = h.mean(dim=1)
        gated = (gate.unsqueeze(-1) * h).sum(dim=1)
        fused = 0.5 * uniform + 0.5 * gated

        out = self.output_proj(fused).unsqueeze(1)
        return out


class CellOmicsTokenizer(nn.Module):
    def __init__(self, num_genes=4079, num_omics=6, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.num_omics = num_omics

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_genes, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_omics)
        ])

    def forward(self, x):
        # x: (batch, genes, omics)
        tokens = []
        for i in range(self.num_omics):
            hi = self.encoders[i](x[:, :, i])  # (batch, hidden)
            tokens.append(hi)
        tokens = torch.stack(tokens, dim=1)   # (batch, omics, hidden)
        return tokens


# ============ 测试代码 ============
if __name__ == "__main__":
    # 测试参数
    batch_size = 8
    num_genes = 4079
    num_omics = 6
    hidden_dim = 256

    # 创建测试数据
    x = torch.randn(batch_size, num_genes, num_omics)
    print(f"输入 x shape: {x.shape}")
    print(f"  - batch_size: {x.shape[0]}")
    print(f"  - num_genes: {x.shape[1]}")
    print(f"  - num_omics: {x.shape[2]}")
    print("-" * 40)

    # 测试 ProperGatedMultiOmicsEncoder
    print("\n【测试 ProperGatedMultiOmicsEncoder】")
    model1 = ProperGatedMultiOmicsEncoder(num_genes=num_genes, num_omics=num_omics, hidden_dim=hidden_dim)
    out1 = model1(x)
    print(f"输出 shape: {out1.shape}")
    print(f"  期望: ({batch_size}, 1, {hidden_dim})")
    print(f"  匹配: {out1.shape == (batch_size, 1, hidden_dim)}")
    print("-" * 40)

    # 测试 CellOmicsTokenizer
    print("\n【测试 CellOmicsTokenizer】")
    model2 = CellOmicsTokenizer(num_genes=num_genes, num_omics=num_omics, hidden_dim=hidden_dim)
    out2 = model2(x)
    print(f"输出 shape: {out2.shape}")
    print(f"  期望: ({batch_size}, {num_omics}, {hidden_dim})")
    print(f"  匹配: {out2.shape == (batch_size, num_omics, hidden_dim)}")
    print("-" * 40)

    # 测试不同 batch size
    print("\n【测试不同 batch size】")
    for bs in [1, 4, 16]:
        x_test = torch.randn(bs, num_genes, num_omics)
        out1_test = model1(x_test)
        out2_test = model2(x_test)
        print(f"batch_size={bs}:")
        print(f"  model1 输出: {out1_test.shape}")
        print(f"  model2 输出: {out2_test.shape}")


