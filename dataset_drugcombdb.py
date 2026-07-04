import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DrugCombDB"
EMB_1D_DIR = DATA_DIR / "DrugCombDB_smiles_emb_1d"
EMB_2D_DIR = DATA_DIR / "DrugCombDB_smiles_emb_2d"
CELLLINE_FILE = DATA_DIR / "cellline_4079_6_drugcombdb.npy"


class SynergyEncoderDataset(data.Dataset):
    def __init__(
        self,
        drug_1_smiles,
        drug_2_smiles,
        Y_Label,
        context,
        maxCompoundLen=128,
        device=torch.device("cuda"),
        fp_name="hashap",
    ):
        self.maxCompoundLen = maxCompoundLen
        self.drug_1_smiles = [str(x) for x in drug_1_smiles]
        self.drug_2_smiles = [str(x) for x in drug_2_smiles]
        self.context = context
        self.Y = Y_Label
        self.device = device
        self.fp_name = fp_name
        self.len = len(self.Y)

        self.cellline_osmic_dict = np.load(CELLLINE_FILE, allow_pickle=True).item()
        self.embeddings_1d = self.load_batch_embeddings(EMB_1D_DIR, "1d")
        self.embeddings_2d = self.load_batch_embeddings(EMB_2D_DIR, "2d")

    def __len__(self):
        return self.len

    def load_batch_embeddings(self, emb_dir: Path, embedding_type: str):
        embeddings = {}
        for drug_id in os.listdir(emb_dir):
            embeddings[str(drug_id)] = self.load_single_embedding(emb_dir / drug_id, drug_id, embedding_type)
        return embeddings

    @staticmethod
    def load_single_embedding(emb_path: Path, drug_id: str, embedding_type: str) -> torch.Tensor:
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing {embedding_type} embedding for drug {drug_id}: {emb_path}")
        return torch.load(emb_path, map_location="cpu").detach().squeeze(0).float()

    def __getitem__(self, index):
        drug_1 = self.drug_1_smiles[index]
        drug_2 = self.drug_2_smiles[index]

        compounds_1d_1 = self.embeddings_1d[drug_1]
        compounds_1d_2 = self.embeddings_1d[drug_2]
        compounds_2d_1 = self.embeddings_2d[drug_1]
        compounds_2d_2 = self.embeddings_2d[drug_2]
        context_features = self.cellline_osmic_dict[self.context[index]]
        synergy_score = int(self.Y[index])

        # TMI-Syn keeps these arguments for interface compatibility, but this
        # model version does not consume SMILES sentence vectors or fingerprints.
        smiles_1 = torch.zeros(384, dtype=torch.float32)
        smiles_2 = torch.zeros(384, dtype=torch.float32)
        fp1 = torch.zeros(1024, dtype=torch.float32)
        fp2 = torch.zeros(1024, dtype=torch.float32)

        return [
            compounds_1d_1,
            compounds_1d_2,
            torch.LongTensor([synergy_score]),
            torch.as_tensor(context_features, dtype=torch.float32),
            compounds_2d_1,
            compounds_2d_2,
            smiles_1,
            smiles_2,
            fp1,
            fp2,
        ]


if __name__ == "__main__":
    train_data = pd.read_csv(DATA_DIR / "DrugCombDB.csv")
    train_set = SynergyEncoderDataset(
        train_data["drug1_db"].astype(str).to_list(),
        train_data["drug2_db"].astype(str).to_list(),
        train_data["label"].to_list(),
        train_data["cell"].to_list(),
        maxCompoundLen=128,
        device=torch.device("cpu"),
    )
    print(train_set[0])
