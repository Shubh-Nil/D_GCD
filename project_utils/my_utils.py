import csv,pickle,time
import os
import numpy as np
import random
import torch
from torch import nn
from torch.nn.init import trunc_normal_
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
# from data_preprocessing.generate_captions import *
import wandb
import torch.nn.functional as F
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_list(source_domain: str, num_classes: int = 40) -> list:
    random.seed(42)
    all_folders = [folder for folder in os.listdir(source_domain) if os.path.isdir(os.path.join(source_domain, folder))]
    selected_folders = random.sample(all_folders, num_classes)
    return selected_folders


def create_csv(source_domain: str, aug_domain: str, csv_dir_path: str, selected_classes: list, episode: int) -> tuple:
    col_names = ['index', 'image_path', 'label', 'numeric_label','mask_known']
    random.seed(42 + episode)
    if len(selected_classes)==40:
        size = 30
    if len(selected_classes)==4:
        size=3
    if len(selected_classes)==250:
        size = 180
    os.makedirs(csv_dir_path,exist_ok=True)
    csv_train_path = os.path.join(csv_dir_path, f"episode{episode}_source.csv")
    csv_synthetic_path = os.path.join(csv_dir_path, f"episode{episode}_synthetic.csv")
    
    source_labels = []
    train_classes = random.sample(population=selected_classes, k=size)
    # Assign continuous numeric labels to the Source Domain classes
    continuous_numeric_labels = {class_name: idx for idx, class_name in enumerate(train_classes)}
    
    # Assign numeric labels to the remaining classes for the Synthetic Domain
    remaining_classes = [class_name for class_name in selected_classes if class_name not in train_classes]
    next_label = len(train_classes)
    for class_name in remaining_classes:
        continuous_numeric_labels[class_name] = next_label
        next_label += 1
    
    if os.path.exists(csv_train_path) and os.path.exists(csv_synthetic_path):
        return csv_train_path, csv_synthetic_path, size #, labels, labelled_or_not

    else:
        with open(csv_train_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=col_names)
            writer.writeheader()
            index = 0
            class_to_indices = {category: [] for category in train_classes}
            for folder_name in train_classes:
                folder_path = os.path.join(source_domain, folder_name)
                for img in os.listdir(folder_path):
                    if img.endswith('.jpg') or img.endswith('.png'):
                        image_path = os.path.join(folder_path, img)
                        class_to_indices[folder_name].append(index)
                        source_labels.append(continuous_numeric_labels[folder_name])
                        index += 1
                        if continuous_numeric_labels[folder_name]<size:
                            mask_lab=1
                        else:
                            mask_lab=0
                        writer.writerow({
                            'index': index,
                            'image_path': image_path,
                            'label': folder_name,
                            'numeric_label': continuous_numeric_labels[folder_name],
                            'mask_known': mask_lab
                        })

        with open(csv_synthetic_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=col_names)
            writer.writeheader()
            index = 0
            for folder_name in selected_classes:
                folder_path = os.path.join(aug_domain, folder_name)
                images = [img for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png'))]
                for img in images:
                    image_path = os.path.join(folder_path, img)
                    if continuous_numeric_labels[folder_name]<size:
                        mask_lab=1
                    else:
                        mask_lab=0
                    writer.writerow({
                        'index': index,
                        'image_path': image_path,
                        'label': folder_name,
                        'numeric_label': continuous_numeric_labels[folder_name],
                        'mask_known':mask_lab
                    })
                    index += 1

        return csv_train_path, csv_synthetic_path, size #, labels, labelled_or_not

def combine_csv_files(csv_file1, csv_file2, output_file):
    # Read the first CSV file
    df1 = pd.read_csv(csv_file1)
    # Add a new column 'label' with value 1
    df1['mask'] = 1

    # Read the second CSV file
    df2 = pd.read_csv(csv_file2)
    # Add a new column 'label' with value 0
    df2['mask'] = 0

    # Combine the two dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)


def create_target_csv(target_domain: str, csv_dir_path: str, selected_classes: list, split: int) -> str:
    '''
    Takes image-dataset from Target Domain.
    All the 65 classes will be taken to csv file and includes triplets (positive and negative anchors).
    
    The selected_classes are assigned continuous numeric labels from 0 to 39.
    Then, the remaining classes are assigned continuous numeric labels from 40 to 64.
    '''
    col_names = ['index', 'image_path', 'label', 'continuous_numeric_label', 'positive_anchor', 'neg_anchor']

    domain_name = os.path.basename(target_domain)
    csv_target_filename = "_".join([domain_name,"target.csv"])
    csv_target_path = os.path.join(csv_dir_path, csv_target_filename)

    with open(csv_target_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=col_names)
        writer.writeheader()

        index = 0
        # continuous_numeric_label = 0
        continuous_label_mapping = {}
        class_to_images = {}
        
        # Assign continuous numeric labels to selected classes
        for i, class_name in enumerate(selected_classes):
            continuous_label_mapping[class_name] = i
        # Assign continuous numeric labels to the remaining classes
        remaining_classes = [cls for cls in os.listdir(target_domain) if cls not in selected_classes]
        for i, class_name in enumerate(remaining_classes, start = split):
            continuous_label_mapping[class_name] = i

        # Gather all images by class
        for folder_name in os.listdir(target_domain):
            folder_path = os.path.join(target_domain, folder_name)
            images = [img for img in os.listdir(folder_path) if img.endswith('.jpg') or img.endswith('.png')]
            class_to_images[folder_name] = images

        # Write rows and generate triplets
        for folder_name, images in class_to_images.items():
            folder_path = os.path.join(target_domain, folder_name)
            for img in images:
                image_path = os.path.join(folder_path, img)

                # Select a positive anchor (different image from the same class)
                positive_index = random.randint(0, len(images) - 1)
                while images[positive_index] == img:
                    positive_index = random.randint(0, len(images) - 1)
                positive_image_path = os.path.join(folder_path, images[positive_index])

                # Select a negative anchor (image from a different class)
                negative_label = folder_name
                while negative_label == folder_name:
                    negative_label = random.choice(list(class_to_images.keys()))
                negative_folder_path = os.path.join(target_domain, negative_label)
                negative_images = class_to_images[negative_label]
                negative_index = random.randint(0, len(negative_images) - 1)
                negative_image_path = os.path.join(negative_folder_path, negative_images[negative_index])

                writer.writerow({
                    'index': index,
                    'image_path': image_path,
                    'label': folder_name,
                    'continuous_numeric_label': continuous_label_mapping[folder_name],
                    'positive_anchor': positive_image_path,
                    'neg_anchor': negative_image_path
                })
                index += 1

    return csv_target_path


def create_TrainTest_target_csv(csv_dir_path: str, csv_path: str) -> tuple[str, str]:
    
    # Read the generated CSV file
    df = pd.read_csv(csv_path)
    
    # Split the dataset into two based on the class labels
    classes_40 = df[df['continuous_numeric_label'].between(0, 39)]
    classes_65 = df[df['continuous_numeric_label'].between(40, 64)]
    # For classes common in both CSV files, split the images accordingly
    dfs_40 = []
    dfs_65 = []
    
    for class_label in classes_40['continuous_numeric_label'].unique():
        class_images = classes_40[classes_40['continuous_numeric_label'] == class_label]
        split_idx = int(0.4 * len(class_images))
        dfs_40.append(class_images.iloc[:split_idx])
        dfs_65.append(class_images.iloc[split_idx:])
    
    # Combine with other classes
    df_40_classes = pd.concat(dfs_40)                           # contains the 40% of images from the common classes (0-39).
    df_65_classes = pd.concat(dfs_65 + [classes_65])            # Contains the remaining 60% of images from the common classes (0-39),
                                                                # plus all images from classes (40-64)
    
    # Save the new CSV files
    path_40_classes = f"{csv_dir_path}/train_target.csv"
    path_65_classes = f"{csv_dir_path}/test_target.csv"
    
    df_40_classes.to_csv(path_40_classes, index=False)
    df_65_classes.to_csv(path_65_classes, index=False)
    
    return path_40_classes, path_65_classes
    
    
# device = "cuda" if torch.cuda.is_available() else "cpu"

# def get_features(image_path: str, model, device: torch.device, transform) -> torch.Tensor:
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(device)
#     model.eval()
#     with torch.no_grad():
#         features = model(image)
#     features = F.normalize(features, p=2, dim=1)
#     return features.squeeze()


def test_kmeans_cdad(model, 
                     test_loader,
                     epoch, 
                     save_name,
                     num_train_classes,
                     device,
                     num_unlabelled_classes):   

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, data in enumerate(tqdm(test_loader)):
        images, label = data
        if isinstance(images, list):
            images = torch.cat(images, dim=0).to(device)
            label = torch.cat([label for _ in range(2)]).to(device)
        else:
            images = images.to(device)
            label = label.to(device)
        # images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(num_train_classes)
                                         else False for x in label]))
        

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=num_train_classes + num_unlabelled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=['v1', 'v2'], save_name=save_name)

    return all_acc, old_acc, new_acc

def test_kmeans_mix(model, 
                     test_loader,
                     epoch, 
                     save_name,
                     num_train_classes,
                     device,
                     num_unlabelled_classes):   

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, data in enumerate(tqdm(test_loader)):
        images, label = data
        if isinstance(images, list):
            images = torch.cat(images, dim=0).to(device)
            label = torch.cat([label for _ in range(2)]).to(device)
        else:
            images = images.to(device)
            label = label.to(device)
        # images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(num_train_classes)
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=num_train_classes + num_unlabelled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=['v1', 'v2'], save_name=save_name)

    return all_acc, old_acc, new_acc

class LoRALayer(torch.nn.Module):
    
    def __init__(
        self,
        device: torch.device,
        rank: int,
        alpha: float,
        d_in: int,
        d_out: int
    ) -> None:
        
        super(LoRALayer, self).__init__()
        self.device = device
        self.d_in = d_in
        self.d_out = d_out
        self.alpha = alpha
        self.rank = rank
        
        self.A = torch.nn.Parameter(
            data=torch.normal(mean=0, std=0.01, size=(self.d_in, self.rank)), 
            requires_grad=True
        )
        self.B = torch.nn.Parameter(
            data=torch.zeros(size=(self.rank, self.d_out)),
            requires_grad=True
        )
    
    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:
        
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."

        x = x.to(self.device)
        delta_W = torch.matmul(self.A, self.B) * (self.alpha / self.rank)
        z = torch.matmul(x, delta_W)
        
        return z

class LinearWithLoRA(torch.nn.Module):
    
    def __init__(
        self,
        device: torch.device,
        linear: torch.nn.Linear,
        rank: int,
        alpha: float
    ) -> None:
        
        super(LinearWithLoRA, self).__init__()
        self.device = device
        self.rank = rank
        self.alpha = alpha
        self.linear = linear.to(self.device)
        self.d_in = self.linear.in_features
        self.d_out = self.linear.out_features
        
        self.lora = LoRALayer(device=self.device,
                              rank=self.rank, 
                              alpha=self.alpha, 
                              d_in=self.d_in,
                              d_out=self.d_out).to(self.device)
    
    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:
        
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions."
        assert x.shape[-1] == self.d_in, f"Expected the last dimension of input to be {self.d_in}, but got {x.shape[-1]}."

        x = x.to(self.device)
        z1 = self.linear(x) 
        z2 = self.lora(x) 
        z = z1 + z2
    
        return z

class ViTModelWithLoRA(torch.nn.Module):
    
    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        lora_rank: int,
        lora_alpha: float
    ) -> None:
        
        super(ViTModelWithLoRA, self).__init__()
        self.device = device
        self.model = model.to(self.device)
        
        # Freeze the parameters of the model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace the last linear layers of model with LinearWithLora layers
        self.rank = lora_rank
        self.alpha = lora_alpha
        ViTModelWithLoRA.replace_linear_with_lora(model=self.model.blocks[-1], 
                                                  rank=self.rank, 
                                                  alpha=self.alpha,
                                                  device=self.device)
    
    @staticmethod
    def replace_linear_with_lora(
        device: torch.device,
        model: torch.nn.Module,
        rank: int,
        alpha: float
    ) -> None:
        
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                linear_lora = LinearWithLoRA(device, module, rank, alpha)
                setattr(model, name, linear_lora) # parent is model, child is module
            else:
                ViTModelWithLoRA.replace_linear_with_lora(device, module, rank, alpha)
       
    def calc_num_lora_params(self) -> None:
        
        # Check if the requires_grad are set correctly
        train_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                train_params += param.numel()
        
        print(f"Number of total parameters: {total_params}")
        print(f"Number of trainable parameters: {train_params}")
        print(f"LoRA efficiency: {train_params * 100 / total_params:.3f}%")
        
    def forward(
        self, 
        inputs: torch.tensor
    ) -> torch.tensor:
        assert list(inputs.shape).__len__() == 4, "inputs rank must be 4 and inputs.shape = (b,C,H,W)"
        
        inputs = inputs.to(self.device)
        z = self.model(inputs)
        
        return z


def save_results_to_excel(results, file_path):
    # Convert results list to a DataFrame
    results_df = pd.DataFrame(results)
    
    # Define the Excel writer using xlsxwriter
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        # Write your DataFrame to a file
        results_df.to_excel(writer, sheet_name='Results', index=False)
    
    print("Results saved to Excel file at:", file_path)
      
                
def create_meta_source_csv(target_domain: str, csv_dir_path: str, selected_classes) -> str:
    '''
    Takes image-dataset from Target Domain.
    Only the selected classes will be taken to csv file and includes triplets (positive and negative anchors).
    '''
    col_names = ['index', 'image_path', 'label', 'continuous_numeric_label', 'positive_anchor', 'neg_anchor']

    csv_target_filename = "meta_source.csv"
    csv_target_path = os.path.join(csv_dir_path, csv_target_filename)

    with open(csv_target_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=col_names)
        writer.writeheader()

        index = 0
        continuous_numeric_label = 0
        class_to_images = {}

        # Gather all images by class
        for folder_name in os.listdir(target_domain):
            if folder_name in selected_classes:
                folder_path = os.path.join(target_domain, folder_name)
                images = [img for img in os.listdir(folder_path) if img.endswith('.jpg') or img.endswith('.png')]
                class_to_images[folder_name] = images

        # Write rows and generate triplets
        for folder_name, images in class_to_images.items():
            folder_path = os.path.join(target_domain, folder_name)
            for img in images:
                image_path = os.path.join(folder_path, img)

                # Select a positive anchor (different image from the same class)
                positive_index = random.randint(0, len(images) - 1)
                while images[positive_index] == img:
                    positive_index = random.randint(0, len(images) - 1)
                positive_image_path = os.path.join(folder_path, images[positive_index])

                # Select a negative anchor (image from a different class)
                negative_label = folder_name
                while negative_label == folder_name:
                    negative_label = random.choice(list(class_to_images.keys()))
                negative_folder_path = os.path.join(target_domain, negative_label)
                negative_images = class_to_images[negative_label]
                negative_index = random.randint(0, len(negative_images) - 1)
                negative_image_path = os.path.join(negative_folder_path, negative_images[negative_index])

                writer.writerow({
                    'index': index,
                    'image_path': image_path,
                    'label': folder_name,
                    'continuous_numeric_label': continuous_numeric_label,
                    'positive_anchor': positive_image_path,
                    'neg_anchor': negative_image_path
                })
                index += 1
            continuous_numeric_label += 1

    return csv_target_path


def cluster_acc(y_true, y_pred, return_ind=False):
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity_score(y_true, y_pred):
    contingency = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def evaluate_clustering(y_true, y_pred):
    acc = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)
    return acc, nmi, ari, pur

def test_kmeans(K, all_feats, targets, mask_lab, verbose=False):
    print('Fitting K-Means...')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    mask = mask_lab

    labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                               nmi_score(targets[mask], preds[mask]), \
                                               ari_score(targets[mask], preds[mask])

    unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask], preds.astype(int)[~mask]), \
                                                     nmi_score(targets[~mask], preds[~mask]), \
                                                     ari_score(targets[~mask], preds[~mask])

    if verbose:
        print('K')
        print('Labelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(labelled_acc, labelled_nmi, labelled_ari))
        print('Unlabelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(unlabelled_acc, unlabelled_nmi, unlabelled_ari))

    return labelled_acc

def test_kmeans_for_scipy(K, all_feats, targets, mask_lab, verbose=False):
    K = int(K)

    print(f'Fitting K-Means for K = {K}...')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    mask = mask_lab

    labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                               nmi_score(targets[mask], preds[mask]), \
                                               ari_score(targets[mask], preds[mask])

    unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask], preds.astype(int)[~mask]), \
                                                     nmi_score(targets[~mask], preds[~mask]), \
                                                     ari_score(targets[~mask], preds[~mask])

    print(f'K = {K}')
    print('Labelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(labelled_acc, labelled_nmi, labelled_ari))
    print('Unlabelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(unlabelled_acc, unlabelled_nmi, unlabelled_ari))

    return -labelled_acc

def binary_search(all_feats, targets, mask_lab, num_labeled_classes, max_classes):
    min_classes = num_labeled_classes+1
    big_k = max_classes
    small_k = min_classes
    diff = big_k - small_k
    middle_k = int(0.5 * diff + small_k)

    labelled_acc_big = test_kmeans(big_k, all_feats, targets, mask_lab)
    labelled_acc_small = test_kmeans(small_k, all_feats, targets, mask_lab)
    labelled_acc_middle = test_kmeans(middle_k, all_feats, targets, mask_lab)

    print(f'Iter 0: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} ')
    all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
    best_acc_so_far = np.max(all_accs)
    best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
    print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')

    for i in range(1, int(np.log2(diff)) + 1):
        if labelled_acc_big > labelled_acc_small:
            best_acc = max(labelled_acc_middle, labelled_acc_big)
            small_k = middle_k
            labelled_acc_small = labelled_acc_middle
            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
        else:
            best_acc = max(labelled_acc_middle, labelled_acc_small)
            big_k = middle_k
            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
            labelled_acc_big = labelled_acc_middle

        labelled_acc_middle = test_kmeans(middle_k, all_feats, targets, mask_lab)

        print(f'Iter {i}: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} ')
        all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
        best_acc_so_far = np.max(all_accs)
        best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
        print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')
        return best_acc_at_k

def scipy_optimise(all_feats, targets, mask_lab, num_labeled_classes, max_classes):
    from functools import partial
    from scipy.optimize import minimize_scalar

    small_k = num_labeled_classes
    big_k = max_classes
    test_k_means_partial = partial(test_kmeans_for_scipy, all_feats=all_feats, targets=targets, mask_lab=mask_lab, verbose=True)
    res = minimize_scalar(test_k_means_partial, bounds=(small_k, big_k), method='bounded', options={'disp': True})
    print(f'Optimal K is {res.x}')
    return res.x

def semi_supervised_kmeans(features, labels, mask_lab, num_known_classes, total_clusters):
    # Initialize centroids for known classes
    known_centroids = [features[labels == i].mean(axis=0) for i in range(num_known_classes)]
    
    # Apply k-means++ on the unlabeled dataset to initialize centroids for unknown classes
    kmeans_plus = KMeans(n_clusters=total_clusters-num_known_classes, init='k-means++')
    kmeans_plus.fit(features[~mask_lab])
    unknown_centroids = kmeans_plus.cluster_centers_
    
    # Combine known and unknown centroids
    centroids = np.vstack([known_centroids, unknown_centroids])
    
    # Perform k-means with initialized centroids
    kmeans = KMeans(n_clusters=total_clusters, init=centroids, n_init=1)
    predicted_labels = kmeans.fit_predict(features)
    
    return predicted_labels

def split_cluster_acc_v1(y_true, y_pred, mask):
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc
def split_cluster_acc_v2(y_true, y_pred, mask):
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc

EVAL_FUNCS = {
    'v1': split_cluster_acc_v1,
    'v2': split_cluster_acc_v2,
}

def log_accs_from_preds(y_true, y_pred, mask, eval_funcs: List[str], save_name: str, T: int=None, print_output=False):
    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):
        acc_f = EVAL_FUNCS[f_name]
        all_acc, old_acc, new_acc = acc_f(y_true, y_pred, mask)
        log_name = f'{save_name}_{f_name}'
        
        if i == 1:                                              # i=0->v1,   i=1->v2
            to_return = (all_acc, old_acc, new_acc)
        
        if print_output:
            print_str = f'Epoch {T}, {log_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
            print(print_str)

    return to_return

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

def get_entropy(features, centers, temperature=0.7):
    return F.cosine_similarity(features.unsqueeze(1), centers, dim = 2)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # Current value
        self.avg = 0  # Average value
        self.sum = 0  # Sum of all values
        self.count = 0  # Count of values

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Save the model at the end of training
def save_model(model, path="final_model.pkl"):
    print("=> Saving the final model")
    # torch.save(model.state_dict(), path)
    t1 = time.time()
    pickle.dump(model,open(path,"wb"))
    t2 = time.time()
    t = t2-t1
    print(f"The model is saved at {path} and it took {t/60:.2f} mints")
    
# Load the saved models for further use
def load_model(path):
    print("=> Loading model from", path)
    t1 = time.time()
    model=pickle.load(open(path,"rb"))
    t2 = time.time()
    t = t2-t1
    print(f"The model is loaded from {path} and it took {t/60:.2f} mints")
    return model
# Function to check the number of trainable parameters
def check_trainable_parameters(model, expected_num_params):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    if num_params != expected_num_params:
        print("Mismatch in number of trainable parameters!")
        return False
    return True

def check_layer_names(model_weights, loaded_weights):
    """
    Check if the layer names in two sets of weights are the same.
    
    Args:
    model_weights (dict): A state_dict of a model containing layer names and parameter tensors.
    loaded_weights (dict): A state_dict of a model loaded from file containing layer names and parameter tensors.
    
    Returns:
    bool: True if all layer names match, False otherwise.
    """
    original_layers = set(model_weights.keys())
    loaded_layers = set(loaded_weights.keys())
    
    if original_layers != loaded_layers:
        missing_in_loaded = original_layers - loaded_layers
        new_in_loaded = loaded_layers - original_layers
        if missing_in_loaded:
            print("Missing layers in loaded model:", missing_in_loaded)
        if new_in_loaded:
            print("New layers found in loaded model:", new_in_loaded)
        return False
    return True

def log_gradients(params, tag):
    """
    Log summary statistics of gradients to WandB for parameters where gradients are being calculated.
    Args:
    - params: Iterable of parameters from the model, typically model.named_parameters().
    - tag: Prefix tag for the logging variable.
    - step: Current step or epoch to log against.
    """
    for name, param in params:
        if param.requires_grad and param.grad is not None:
            grad = param.grad.cpu().numpy()
            grad_mean = np.linalg.norm(grad)

            # wandb.log({
            #     f"{tag}/{name}_norm": grad_mean,
            # })
import datetime
def log_gradients_txt(params, filename = f"gradients_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", batch_idx=None, epoch=None):
    with open(filename, 'a') as f:
        f.write(f"--- Epoch: {epoch}, Batch: {batch_idx} ---\n")
        for name, param in params:
            if param.requires_grad and param.grad is not None:
                grad = param.grad.cpu().numpy()
                grad_norm = np.linalg.norm(grad)
                f.write(f"Parameter: {name}\n")
                f.write(f"L2_Norm Gradient: {grad_norm:.6f}\n")
                f.write("\n")
        f.write("\n")
def log_accumulated_gradients(accumulated_grads, tag):
    """
    Log summary statistics of accumulated gradients to WandB for the global model before the update.
    Args:
    - accumulated_grads: Dictionary containing accumulated gradients.
    - tag: Prefix tag for the logging variable.
    - meta_epoch: Current meta epoch to log against.
    """
    for name, grad in accumulated_grads.items():
        grad = grad.cpu().numpy()
        grad_norm = np.linalg.norm(grad)
        wandb.log({
            f"{tag}/{name}_norm": grad_norm,
        })
class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

def loss_scheduler(epoch,num_epochs,min_weight=20,max_weight=200):
    return max((max_weight)/num_epochs * (num_epochs - epoch), min_weight)


def create_validation_csv(base_path, domains, selected_classes,args):
    all_data = []
    # Initialize label mapping with the first 250 selected classes
    label_mapping = {cls: i for i, cls in enumerate(selected_classes)}
    supported_formats = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    # Collect all image paths and their labels
    for domain in domains:
        domain_path = os.path.join(base_path, domain)
        for classname in os.listdir(domain_path):
            if classname in label_mapping:
                class_path = os.path.join(domain_path, classname)
                if os.path.isdir(class_path):  # Ensure it's a directory
                        numeric_label = label_mapping[classname]
                        for img_filename in os.listdir(class_path):
                            if img_filename.lower().endswith(supported_formats):  # Assuming images are in JPEG format
                                all_data.append({
                                    "index":1,
                                    "image_path": os.path.join(class_path, img_filename),
                                    "label": classname,
                                    "continuous_numeric_label": numeric_label,
                                })

    # Convert list to DataFrame
    full_df = pd.DataFrame(all_data)
    if full_df.empty:
        print("No data collected. Exiting function.")
        return

    # Ensure the directory exists
    validation_dir = os.path.join(args.base_dir,f"Episode_all_{args.dataset_name}/Validation")
    os.makedirs(validation_dir, exist_ok=True)
    # Save the combined validation set to a CSV file
    csv_file_name = os.path.join(validation_dir, "combined_validation.csv")
    full_df.to_csv(csv_file_name, index=False)
    print(f"Saved {csv_file_name} with {len(full_df)} images.")
    

    
#---------FISHER MERGING----------------#

def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge

def get_param_squared_gradients(model, param_names_to_merge):
    """
    Get the squared gradients for specified parameters of a model.
    """
    param_squared_gradients = {
        param_name: param_value.grad.detach() ** 2
        for param_name, param_value in model.named_parameters()
        if param_name in param_names_to_merge and param_value.grad is not None
    }
    return param_squared_gradients


def get_models_fisher_norm(models_to_merge_param_dict: dict, models_to_merge_fisher_weights_list: list, minimal_fisher_weight: float = 1e-6):
    """
    Get normalization of Fisher weights for all models to be merged.
    """
    # Dictionary to hold L2 norms for each parameter across models
    models_fisher_norm_dict = {}

    # Compute L2 norm over models for each parameter
    for param_name, _ in models_to_merge_param_dict.items():
        # Determine the device to use
        device = None
        for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list:
            if param_name in model_to_merge_fisher_weights:
                device = model_to_merge_fisher_weights[param_name].device
                break
        if device is None:
            device = models_to_merge_param_dict[param_name][0].device

        # Stack Fisher weights, filling in missing keys with minimal_fisher_weight
        models_fisher = torch.stack([
            model_to_merge_fisher_weights.get(
                param_name,
                torch.full_like(models_to_merge_param_dict[param_name][0], minimal_fisher_weight, device=device)
            ).to(device)
            for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list
        ], dim=0)

        # Compute L2 norm for each parameter, across all dimensions except the first (models)
        dims = [dim_idx for dim_idx in range(1, models_fisher.dim())]
        models_fisher_norm = torch.linalg.vector_norm(models_fisher, ord=2, dim=dims)

        # Store norms in the dictionary
        models_fisher_norm_dict[param_name] = models_fisher_norm

    # Ensure all tensors are on the same device before stacking
    devices = set([v.device for v in models_fisher_norm_dict.values()])
    if len(devices) > 1:
        # Move all tensors to the target device (e.g., the first one)
        target_device = next(iter(devices))
        models_fisher_norm_dict = {k: v.to(target_device) for k, v in models_fisher_norm_dict.items()}

    # Stack norms across parameters to form (num_models_to_merge, num_parameters) tensor
    models_fisher_norm = torch.stack(list(models_fisher_norm_dict.values()), dim=1)

    # Compute L2 norm over all parameters for each model (resulting in shape (num_models_to_merge,))
    models_fisher_norm = torch.linalg.vector_norm(models_fisher_norm, ord=2, dim=1)

    return models_fisher_norm



def merging_with_fisher_weights(models_to_merge_param_dict: dict, models_to_merge_fisher_weights_list: list, fisher_scaling_coefficients: torch.Tensor,
                                normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6):
    """
    Merge parameters of different models with computed Fisher weights.
    """
    merged_params = {}

    if normalize_fisher_weight:
        models_fisher_norm = get_models_fisher_norm(
            models_to_merge_param_dict=models_to_merge_param_dict,
            models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list,
            minimal_fisher_weight=minimal_fisher_weight
        )

    for param_name, param_value_list in models_to_merge_param_dict.items():
        # Stack the parameter values across models
        param_values = torch.stack(param_value_list, dim=0)

        # Determine the device to use
        device = param_values.device

        # Stack Fisher weights, using minimal_fisher_weight as a default for missing parameters
        models_to_merge_fisher_weights = torch.stack([
            model_to_merge_fisher_weights.get(
                param_name,
                torch.full_like(param_values[0], minimal_fisher_weight, device=device)
            ).to(device)
            for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list
        ], dim=0)

        # Reshape scaling coefficients and move to the correct device
        reshaped_scaling_coefficients = fisher_scaling_coefficients.reshape(-1, *[1 for _ in range(param_values.dim() - 1)]).to(device)

        if normalize_fisher_weight:
            _models_fisher_norm = 1.0 / (models_fisher_norm + minimal_fisher_weight)
            normalized_models_fisher_norm = _models_fisher_norm / _models_fisher_norm.sum()
            normalized_models_fisher_norm = normalized_models_fisher_norm.reshape(-1, *[1 for _ in range(param_values.dim() - 1)]).to(device)
            reshaped_scaling_coefficients = reshaped_scaling_coefficients * normalized_models_fisher_norm

        # Compute weighted average
        numerator = (reshaped_scaling_coefficients * models_to_merge_fisher_weights * param_values).sum(dim=0)
        denominator = (reshaped_scaling_coefficients * models_to_merge_fisher_weights).sum(dim=0)
        merged_param = numerator / denominator

        # Add merged parameter to dictionary
        merged_params[param_name] = merged_param

    return merged_params
