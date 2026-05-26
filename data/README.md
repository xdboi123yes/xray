# Data

This directory is **gitignored**. The NIH ChestX-ray14 dataset (~45 GB) is never pushed to the repo.

## Layout

```
data/
├── raw/                          # Raw data (gitignored)
│   ├── Data_Entry_2017.csv       # NIH metadata
│   └── images/                   # All .png files (flat layout)
├── processed/                    # Splits (gitignored)
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── calibration.csv
│   └── image_dir.txt
└── synthetic/                    # Stable Diffusion outputs (gitignored)
    └── synthetic.csv
```

## How to fetch the data

### Option 1 — Colab notebook (recommended)
`notebooks/xray_colab_training_auto.ipynb` downloads everything automatically from Kaggle.

### Option 2 — Local Kaggle CLI
```bash
pip install kaggle
# place ~/.kaggle/kaggle.json
kaggle datasets download -d nih-chest-xrays/data -p data/raw --unzip
python scripts/preprocess.py --image-dir data/raw/images
```

### Option 3 — Direct from NIH
Download the ChestX-ray14 dataset from the NIH National Library of Medicine, place
`Data_Entry_2017.csv` under `data/raw/`, collect every `.png` into a single flat
directory, and then run preprocessing.
