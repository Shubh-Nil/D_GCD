#!/usr/bin/env python3
import argparse
import os
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

def parse_args():
    p = argparse.ArgumentParser(
        description="Build CSV of image_path,label for selected synthetic domains, "
                    "and list the other domain paths."
    )
    p.add_argument("--dataset_name", required=True,
                   default="PACS",
                   choices=["PACS", "Office_Home", "Domain_Net"],
                   help="Base dataset name")
    return p.parse_args()

def main():
    args = parse_args()

    # Validation domains per dataset
    if args.dataset_name == "PACS":
        val_domains = [
            "Snow_Background_Dataset",
            "White_Background_Dataset",
            "Water_Background_Dataset"
        ]
    elif args.dataset_name == "Office_Home":
        val_domains = [
            "IceBlue_Background_Dataset",
            "Grey_Background_Dataset",
            "Water_Background_Dataset"
        ]
    elif args.dataset_name == "Domain_Net":
        val_domains = [
            "Grey_Background_Dataset",
            "Forest_Background_Dataset"
        ]


    base_dir = os.path.join(REPO_ROOT, "datasets_synthetic", args.dataset_name)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"{base_dir} does not exist")

    all_domains = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    # Split into val and train dataset
    val_dirs = [os.path.join(base_dir, d) for d in all_domains if d in val_domains]
    train_dirs = [os.path.join(base_dir, d) for d in all_domains if d not in val_domains]


    val_path = os.path.join(REPO_ROOT, f'Episode_all_{args.dataset_name}', 'Validation', 'combined_validation_set.csv')
    os.makedirs(os.path.dirname(val_path), exist_ok=True)

    train_path = os.path.join(REPO_ROOT, 'data', f'syn_dataset_path_{args.dataset_name}.csv')
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    # Build Validation csv
    with open(val_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["image_path", "label"])
        for domain_dir in val_dirs:
            for cls in os.listdir(domain_dir):
                cls_dir = os.path.join(domain_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    img_path = os.path.join(cls_dir, fname)
                    writer.writerow([img_path, cls])
    print(f"wrote Validation csv: {val_path}")


    with open(train_path, "w", newline="") as pf:
        writer = csv.writer(pf)
        writer.writerow(["index", "data_path"])  # Write header
        for idx, d in enumerate(train_dirs, start=1):
            writer.writerow([idx, d])


if __name__ == "__main__":
    main()
