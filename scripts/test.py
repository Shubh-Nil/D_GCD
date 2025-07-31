#!/usr/bin/env python3

import os
import sys
import argparse
import warnings

import torch
import torch.nn as nn
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(REPO_ROOT)

from project_utils.my_utils import load_model, ContrastiveLearningViewGenerator
from project_utils.loss import *
from data.augmentations import get_transform
from project_utils.data_setup import create_list, create_ViT_test_dataloaders, test_kmeans_cdad

# suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    print("helo")
    parser = argparse.ArgumentParser(description="Run testing over a series of checkpoints")
    parser.add_argument('--checkpoint', required=True, type=str,
                        help="Checkpoint index (e.g. 0,1,..)")
    parser.add_argument('--dataset_name', required=True, type=str,
                        choices=['PACS','Office_Home','Domain_Net'],
                        help="Dataset to test on")
    args = parser.parse_args()

    checkpoint_idx = args.checkpoint
    dataset = args.dataset_name

    # hyper‑parameters
    BATCH_SIZE = 128

    num_classes_mapping = {'Office_Home':40, 'PACS':4, 'Domain_Net':250}
    total_classes_mapping = {'Office_Home':65, 'PACS':7, 'Domain_Net':345}

    CKPT_DIR    = os.path.join(REPO_ROOT, 'checkpoints', dataset)
    DATA_DIR    = os.path.join(REPO_ROOT, 'datasets', dataset)
    EPISODE_DIR = os.path.join(REPO_ROOT, f'Episode_all_{dataset}')
    RESULT_DIR  = os.path.join(REPO_ROOT, 'Results', dataset)
    os.makedirs(RESULT_DIR, exist_ok=True)

    source_domains = {}
    if dataset == 'PACS':
        for dom in ["art_painting","photo","cartoon","sketch"]:
            fname = f"Intermediate_{dom}_dgcd_epoch{checkpoint_idx}.pkl"
            source_domains[dom] = os.path.join(CKPT_DIR, dom, fname)
    elif dataset == 'Office_Home':
        for dom in ["Art","Clipart","Product","Real_world"]:
            fname = f"Intermediate_{dom}_dgcd_epoch{checkpoint_idx}.pkl"
            source_domains[dom] = os.path.join(CKPT_DIR, dom, fname)
    else:  # Domain_Net
        for dom in ["clipart","sketch","painting"]:
            fname = f"Intermediate_{dom}_dgcd_epoch{checkpoint_idx}.pkl"
            source_domains[dom] = os.path.join(CKPT_DIR, dom, fname)

    # all domains list
    all_domains = {
        'PACS':        ["art_painting","photo","cartoon","sketch"],
        'Office_Home': ["Art","Clipart","Product","Real_world"],
        'Domain_Net':  ["clipart","infograph","quickdraw","real","painting","sketch"]
    }[dataset]

    # prepare device & model backbone
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    global_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
    global_model.head = nn.Identity()

    # transforms
    train_t, test_t = get_transform("imagenet", image_size=224)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_t, n_views=2)

    # prepare selected classes
    src_domain0 = all_domains[0]
    selected_classes = create_list(
        source_domain=os.path.join(DATA_DIR, src_domain0),
        num_classes=num_classes_mapping[dataset]
    )
    print(f"Selected classes: {len(selected_classes)}  |  Unlabelled: {total_classes_mapping[dataset]-num_classes_mapping[dataset]}")

    results = []
    for src_dom, ckpt_path in source_domains.items():
        print(f"Loading model for source domain {src_dom} from {ckpt_path}...")
        model = load_model(ckpt_path).to(device)

        for tgt_dom in all_domains:
            if tgt_dom == src_dom:
                continue

            print(f"Testing {src_dom} → {tgt_dom} ...")
            csv_folder = os.path.join(EPISODE_DIR, tgt_dom)
            test_loader = create_ViT_test_dataloaders(
                target_domain=os.path.join(DATA_DIR, tgt_dom),
                csv_dir_path=csv_folder,
                batch_size=BATCH_SIZE,
                transform=train_transform,
                selected_classes=selected_classes,
                split=num_classes_mapping[dataset]
            )

            try:
                with torch.no_grad():
                    all_acc, old_acc, new_acc = test_kmeans_cdad(
                        device=device,
                        model=model,
                        test_loader=test_loader,
                        epoch=1,
                        save_name='Test ACC',
                        num_train_classes=num_classes_mapping[dataset],
                        num_unlabelled_classes=total_classes_mapping[dataset]-num_classes_mapping[dataset]
                    )
                print(f"→ {tgt_dom} | All: {all_acc:.4f}, Old: {old_acc:.4f}, New: {new_acc:.4f}")
                results.append({
                    "Checkpoint": checkpoint_idx,
                    "Source Domain": src_dom,
                    "Target Domain": tgt_dom,
                    "All": all_acc*100,
                    "Old": old_acc*100,
                    "New": new_acc*100
                })
            except Exception as e:
                print(f"Failed on {tgt_dom}: {e}")

    # save results
    out_file = os.path.join(RESULT_DIR, f"DGCD_Testing_Results.xlsx")
    if os.path.exists(out_file):
        df_old = pd.read_excel(out_file)
        df_new = pd.concat([df_old, pd.DataFrame(results)], ignore_index=True)
    else:
        df_new = pd.DataFrame(results)
    df_new.to_excel(out_file, index=False)
    print(f"Results written to {out_file}")

if __name__ == "__main__":
    main()
