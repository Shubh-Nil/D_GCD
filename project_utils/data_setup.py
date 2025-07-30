import torch
from torch.utils.data import *
from torchvision import transforms
from torch.optim import Optimizer
from PIL import Image
import os
import pandas as pd
from project_utils.my_utils import *

class DGCDDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)), 
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]        
        image = Image.open(image_path).convert('RGB')
        numeric_label = self.data.iloc[idx, 3]
        # pos=self.data.iloc[idx,4]
        # positive_img=Image.open(pos).convert('RGB')
        # neg=self.data.iloc[idx,5]
        # negative_img=Image.open(neg).convert('RGB')
        # mask=self.data.iloc[idx, 6]
        
        if self.transform:
            image = self.transform(image)
        
        return image, numeric_label

class CombineDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)), 
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]        
        image = Image.open(image_path).convert('RGB')
        numeric_label = self.data.iloc[idx, 3]
        mask=self.data.iloc[idx, 5]
        mask_known=self.data.iloc[idx, 4]
        if self.transform:
            image = self.transform(image)
        return image, numeric_label,mask,mask_known

def create_combine_dataloader(combine_csv:str, batch_size: int, transform) -> tuple:
    train_dataset = CombineDataset(csv_file=combine_csv, transform=transform)
    # df = pd.read_csv(combine_csv)
    # # Count the number of 0s and 1s in the 'mask' column
    # label_len = int((df['mask'] == 1).sum())
    # unlabelled_len = int((df['mask'] == 0).sum())
    # len_dataset = label_len + unlabelled_len

    # sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len_dataset)]
    # sample_weights = torch.DoubleTensor(sample_weights)
    # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len_dataset)           # Set shuffle to False when using a sampler
    #  train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, sampler=sampler,num_workers=4)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=4)
    return train_dataloader


def mix_validation_dataloader(csv_file, batch_size, shuffle=True, num_workers=4, transform=None):
    # Create an instance of the OfficeHomeDataset
    
    dataset = DGCDDataset(csv_file=csv_file, transform=transform)
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader


def create_target_dataloaders(target_domain: str, csv_dir_path: str, batch_size: int, selected_classes: list,transform) -> tuple[DataLoader, DataLoader]:

    path = create_target_csv(target_domain = target_domain,
                             csv_dir_path = csv_dir_path,
                             selected_classes = selected_classes)
    # Create the two additional CSV files
    train_target_path, test_target_path = create_TrainTest_target_csv(csv_dir_path = csv_dir_path,
                                                                      csv_path = path)
    
    # Create Datasets and Dataloaders for both the additional CSV files
    train_target_Dataset = DGCDDataset(csv_file=train_target_path, transform=transform)
    test_target_Dataset = DGCDDataset(csv_file=test_target_path, transform=transform)

    train_target_Dataloader = DataLoader(dataset=train_target_Dataset,
                                         batch_size=batch_size,
                                         shuffle=True,num_workers=4)
    test_target_Dataloader = DataLoader(dataset=test_target_Dataset,
                                        batch_size=batch_size,
                                        shuffle=True,num_workers=4)

    return train_target_Dataloader, test_target_Dataloader

def create_ViT_test_dataloaders(target_domain: str, csv_dir_path: str, batch_size: int, selected_classes: list,transform) -> tuple[DataLoader, DataLoader]:
    
    path = create_target_csv(target_domain = target_domain,
                             csv_dir_path = csv_dir_path,
                             selected_classes = selected_classes)
    # Create Datasets and Dataloaders for both the additional CSV files
    target_Dataset = DGCDDataset(csv_file=path, transform=transform)

    target_Dataloader = DataLoader(dataset=target_Dataset,
                                         batch_size=batch_size,
                                         shuffle=True,num_workers=4)

    return target_Dataloader

def create_ViT_test_dataloaders(target_domain: str, csv_dir_path: str, batch_size: int, selected_classes: list,transform, split: int) -> tuple[DataLoader, DataLoader]:
    
    path = create_target_csv(target_domain = target_domain,
                             csv_dir_path = csv_dir_path,
                             selected_classes = selected_classes,
                             split=split)
    # Create Datasets and Dataloaders for both the additional CSV files
    target_Dataset = DGCDDataset(csv_file=path, transform=transform)

    target_Dataloader = DataLoader(dataset=target_Dataset,
                                         batch_size=batch_size,
                                         shuffle=True,num_workers=4)

    return target_Dataloader
def create_ViT_train_dataloaders(source_domain: str, csv_dir_path: str, batch_size: int, selected_classes: list, transform, split: int) -> tuple[DataLoader, DataLoader]:
    domain_name=os.path.basename(source_domain)
    csv_folder_path = os.path.join(csv_dir_path, domain_name)
    os.makedirs(csv_folder_path, exist_ok=True)
    csv_path = create_target_csv(target_domain = source_domain,
                                 csv_dir_path = csv_folder_path,
                                 selected_classes = selected_classes,
                                 split = split)
    
    df = pd.read_csv(csv_path)
    filtered_df = df[df['continuous_numeric_label'].between(0, len(selected_classes)-1)]
    # Save the filtered DataFrame to a new CSV file
    csv_train_filename = f"{domain_name}_train.csv"
    train_csv_path = os.path.join(csv_folder_path, csv_train_filename)
    filtered_df.to_csv(train_csv_path, index=False) 

    train_dataset = DGCDDataset(csv_file = train_csv_path, transform = transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)

    return train_dataloader