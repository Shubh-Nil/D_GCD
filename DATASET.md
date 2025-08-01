## Dataset Preparation

Ensure you are in the `D_GCD` root directory, then create a `datasets` folder:

```bash
mkdir -p datasets
```

Download the following datasets inside the  `datasets` folder: [**PACS**](https://www.kaggle.com/datasets/nickfratto/pacs-dataset) | [**Office\_Home**](https://www.hemanthdv.org/officeHomeDataset.html) | [**Domain\_Net**](https://huggingface.co/datasets/wltjr1007/DomainNet)

`datasets` directory should follow this structure:

```
datasets/
├── PACS/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
├── Office_Home/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
└── DomainNet/
    ├── class1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── ...
```

## Synthetic Dataset Preparation

Generate Synthetic domains for *your preferred dataset* using *prompts* fed into the [**InstructPix2Pix**](https://github.com/timothybrooks/instruct-pix2pix) diffusion model.

```bash
chmod +x config/syn_generate.sh
python config/syn_generate.sh
```

**Style your prompts**
* `Add Snow Background`
* `Add Forest Background`
* `Add Desert Background` etc.

This creates a *Synthetic domain* under:

```
datasets_synthetic/{your_preferred_dataset}/{synthetic_dataset}
```

**Reserve Validation domains**
From the generated synthetic folders, set aside desired ones for validation:

```bash
python scripts/val_generate.py --dataset_name {your_preferred_dataset}
```

You can now proceed to the **Model training/ testing section** in the `README.md`.
