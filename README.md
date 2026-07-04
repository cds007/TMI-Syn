# TMI-Syn Training Code

This folder contains the five-fold cross-validation training entry points for
TMI-Syn.

## Files

- `model.py`: TMI-Syn model.
- `cell_line_encoder.py`: multi-omics cell-line encoder used by TMI-Syn.
- `dataset_oneil.py`: ONeil data loader.
- `dataset_drugcombdb.py`: DrugCombDB data loader.
- `train_oneil.py`: five-fold training on ONeil.
- `train_drugcombdb.py`: five-fold training on DrugCombDB.

## Data

- `ONeil/`: renamed from the original Merck folder, keeping only files used by
  training.
- `DrugCombDB/`: renamed from the original MCDSP folder, keeping only files
  used by training.

## Run

```bash
python train_oneil.py --gpu_id 0
python train_drugcombdb.py --gpu_id 0
```

Logs, checkpoints, and summaries are written under `log/`, `checkpoint/`, and
`fold_results/`.
