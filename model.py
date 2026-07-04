import torch
import torch.nn as nn
import torch.nn.functional as F
from cell_line_encoder import ProperGatedMultiOmicsEncoder

"""
TMI-Syn: Tri-Branch Highway Interaction with confidence-aware calibration
and contrastive learning.
"""


class Highway(nn.Module):
    r"""Highway Layers
    Args:
        - num_highway_layers(int): number of highway layers.
        - input_size(int): size of highway input.
    """

    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            # Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))
            # Compute non linear information
            linear = self.linear[layer](x)
            # Compute linear information
            x = gate * non_linear + (1 - gate) * linear
            # Combine non linear and linear information according to gate
            x = self.dropout(x)
        return x  # [B, 1280]


class DualInteract(nn.Module):
    def __init__(self, field_dim, embed_size, head_num, dropout=0.5):
        super(DualInteract, self).__init__()

        self.bit_wise_net = Highway(input_size=field_dim * embed_size,
                                    num_highway_layers=2)

        hidden_dim = 1024

        self.trans_bit_nn = nn.Sequential(
            nn.LayerNorm(field_dim * embed_size),
            nn.Linear(field_dim * embed_size, hidden_dim),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim * embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """

        b, f, e = x.shape
        bit_wise_x = self.bit_wise_net(x.reshape(b, f * e))

        m_bit = self.trans_bit_nn(bit_wise_x)

        m_x = m_bit + x.reshape(b, f * e)
        return m_x


class DrugInteractionMLP(nn.Module):
    """
    Simple MLP for interaction fusion, replacing the redundant CrossAttention
    when sequence length is 1 (Global features).
    """

    def __init__(self, hidden_dim=256, dropout=0.2):
        super(DrugInteractionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            # Optional: Add another non-linearity or normalization if needed
            # But Linear -> ReLU -> Norm -> Drop -> Linear is a standard block
        )

    def forward(self, x1, x2):
        # x1, x2: (B, hidden_dim)
        combined = torch.cat([x1, x2], dim=-1)  # (B, 2*hidden_dim)
        return self.mlp(combined)  # (B, hidden_dim)


class TMISyn(nn.Module):

    def __init__(self, embed_dim=128, use_gated_encoder=True, dropout_rate=0.2):
        super().__init__()

        self.use_gated_encoder = use_gated_encoder
        proj_dim = 256
        # dropout_rate = 0.2 # Now passed as argument

        # Projection layers and gated multi-omics cell-line encoder.
        self.gated_cell_encoder = ProperGatedMultiOmicsEncoder(
            num_genes=4079, num_omics=6, hidden_dim=256, dropout=0.2
        )
        # We still keep these projections as they process the inputs, but we only use global
        self.projection_1d_1 = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_1d_2 = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_2d_1 = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_2d_2 = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_smi_1 = nn.Sequential(
            nn.Linear(384, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_smi_2 = nn.Sequential(
            nn.Linear(384, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )

        # --- Global Interaction Stream Only ---
        # Fuses SMILES (semantic) and 2D (structure)
        self.drug_encoder1 = DrugInteractionMLP(hidden_dim=proj_dim, dropout=dropout_rate)
        # Fuses Drug1 and Drug2 Global
        self.ddi_encoder = DrugInteractionMLP(hidden_dim=proj_dim, dropout=dropout_rate)
        # Fuses Pair and Cell Global
        self.drug_cell_encoder = DrugInteractionMLP(hidden_dim=proj_dim, dropout=dropout_rate)

        # field_dim = 3: Drug1-Cell, Drug2-Cell, Drug1-Drug2
        # Three interaction fields: drug1-cell, drug2-cell, and drug-drug.
        field_dim = 3

        # this is a HighWay module acutally.It can be called HighWay Module.
        self.feature_interact = DualInteract(field_dim=field_dim, embed_size=proj_dim, head_num=4, dropout=dropout_rate)

        self.transform = nn.Sequential(
            nn.LayerNorm(proj_dim * field_dim),
            nn.Linear(proj_dim * field_dim, 2),
        )

        # === Confidence-Aware Dual Calibration (CADC) Modules ===
        # Confidence Head: Predicts confidence score c in [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(proj_dim * field_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Feature Calibration Gate: Calibrates features based on confidence
        self.feature_calibration = nn.Sequential(
            nn.Linear(proj_dim * field_dim, proj_dim * field_dim),
            nn.Sigmoid()
        )

        # === Contrastive Learning Projection Head ===
        # Maps features to a lower-dimensional latent space for contrastive loss calculation
        # SimCLR/SimCSE style projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(proj_dim * field_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 128)  # Project to 128-dim space for InfoNCE
        )

    def forward(self, drug1_1d_vectors,drug2_1d_vectors,context,drug1_2d_vectors,drug2_2d_vectors,drug1_smi_vectors,drug2_smi_vectors,drug1_fp_vectors,drug2_fp_vectors):
        """
        drug1_1d_vectors: (B, 128)
        drug2_1d_vectors: (B, 128)
        context: (B, 4079, 6)
        drug1_2d_vectors: (B, 128)
        drug2_2d_vectors: (B, 128)
        drug1_smi_vectors: (B, 384)
        drug2_smi_vectors: (B, 384)
        drug1_fp_vectors: (B, 2048)
        drug2_fp_vectors: (B, 2048)
        """

        # 1. Base Projections
        cell_feature = self.gated_cell_encoder(context).squeeze(1)  # (B, 256)

        # Sequence Features (B, 256)
        drug1_1d_global = self.projection_1d_1(drug1_1d_vectors)
        drug2_1d_global = self.projection_1d_2(drug2_1d_vectors)
        drug1_2d_global = self.projection_2d_1(drug1_2d_vectors)
        drug2_2d_global = self.projection_2d_2(drug2_2d_vectors)

        # Fuse SMILES + 2D Global -> Enhanced Drug Global
        drug1_global = self.drug_encoder1(drug1_1d_global, drug1_2d_global)  # (B, 256)
        drug2_global = self.drug_encoder1(drug2_1d_global, drug2_2d_global)  # (B, 256)

        # 1. Drug-Drug Interaction View (Molecular Compatibility)
        d1_d2_view = self.ddi_encoder(drug1_global, drug2_global)  # (B, 256)

        # 2. Drug-Cell Interaction Views (Biological Response)
        # Shared weights for symmetry.
        d1_cell_view = self.drug_cell_encoder(drug1_global, cell_feature)  # (B, 256)
        d2_cell_view = self.drug_cell_encoder(drug2_global, cell_feature)  # (B, 256)

        # Stack Views: [Drug1-Cell, Drug2-Cell, Drug1-Drug2]
        # all_features has shape (B, 3, 256).
        all_features = torch.stack([
            d1_cell_view,
            d2_cell_view,
            d1_d2_view
        ], dim=1)  # (B, 3, 256)

        # Dual Interaction (Bit-wise & Vector-wise)
        # Multi-view interaction module.
        interacted_features = self.feature_interact(all_features)  # (B, 3*256)

        # === Confidence-Aware Dual Calibration ===
        # 1. Calculate Confidence
        confidence = self.confidence_head(interacted_features)  # (B, 1)

        # 2. Feature Calibration
        calibration_gate = self.feature_calibration(interacted_features)
        calibrated_features = interacted_features * calibration_gate # Element-wise scaling

        # Prediction using calibrated features
        out = self.transform(calibrated_features)

        # === Contrastive Projection ===
        # Project calibrated features to latent space
        proj_features = self.contrastive_head(calibrated_features)

        return out, confidence, proj_features


if __name__ == "__main__":
    print("Testing TMI-Syn network...")
    print("=" * 60)

    model = TMISyn(embed_dim=128, use_gated_encoder=True)
    print("Model created successfully")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    batch_size = 8
    embed_dim = 128

    compounds_1d_1 = torch.randn(batch_size, embed_dim)
    compounds_1d_2 = torch.randn(batch_size, embed_dim)
    compounds_2d_1 = torch.randn(batch_size, embed_dim)
    compounds_2d_2 = torch.randn(batch_size, embed_dim)

    context = torch.randn(batch_size, 4079, 6)

    smiles_1 = torch.randn(batch_size, 384)
    smiles_2 = torch.randn(batch_size, 384)
    fp1 = torch.randn(batch_size, 1024)
    fp2 = torch.randn(batch_size, 1024)

    with torch.no_grad():
        output, confidence, proj = model(
            compounds_1d_1,
            compounds_1d_2,
            context,
            compounds_2d_1,
            compounds_2d_2,
            smiles_1,
            smiles_2,
            fp1,
            fp2,
        )

    print("Forward pass succeeded")
    print(f"Output shape: {output.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Contrastive feature shape: {proj.shape}")
