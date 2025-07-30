#!/usr/bin/env python3

import os
import sys
import csv
import argparse
import warnings
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import wandb
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(REPO_ROOT)

from project_utils.my_utils import (
    save_model, load_model, str2bool,
    AverageMeter, log_gradients,
    ContrastiveLearningViewGenerator,
    DINOHead ,test_kmeans_mix
)
from project_utils.loss import SupervisedContrastiveLoss, bce_loss, info_nce_logits
from project_utils.data_setup import (
    create_list, create_csv, combine_csv_files,
    create_combine_dataloader ,mix_validation_dataloader
)
from data.augmentations import get_transform
from models.classifier import Classifier
# from project_utils.loss import DINOHead

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    parser = argparse.ArgumentParser(
        description="Domain Generalization on Image Data with known/ novel classes."
    )
    parser.add_argument('--global_epochs',   type=int,    default=10)
    parser.add_argument('--episodes',        type=int,    default=6)
    parser.add_argument('--task_epochs',     type=int,    default=8)
    parser.add_argument('--task_lr',         type=float,  default=0.01)
    parser.add_argument('--batch_size',      type=int,    default=128)
    parser.add_argument('--alpha',           type=float,  default=0.7)
    parser.add_argument('--n_views',         type=int,    default=2)
    parser.add_argument('--image_size',      type=int,    default=224)
    parser.add_argument('--dataset_name',    type=str,    default='PACS',
                        choices=['Office_Home','PACS','Domain_Net'])
    parser.add_argument('--source_domain_name', type=str, default='Art')
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    parser.add_argument('--transform',       type=str,    default='imagenet')
    parser.add_argument('--device_id',       type=int,    default=0)
    args = parser.parse_args()

    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    # mappings
    num_classes = {'Office_Home':40,'PACS':4,'Domain_Net':250}[args.dataset_name]

    # Hyper-parameters
    m = 0.7
    mu = 1
    lamb = 0.2
    feat_dim = 768

    # if args.dataset_name=="PACS":
    #     val_domains=["Snow_Background_Dataset","White_Background_Dataset","Water_Background_Dataset"]
    # if args.dataset_name=="Office_Home":
    #     val_domains=["IceBlue_Background_Dataset", "Grey_Background_Dataset", "Water_Background_Dataset"]
    # # if args.dataset_name=="Domain_Net":
    # #     val_domains=["Add_Grey_Background","Add_Forest_Background"]

    # initialize WandB
    # wandb.init(
    #     project=f"DGCD_{args.dataset_name}",
    #     config=args.__dict__
    # )
    # wandb.config.update({"num_classes": num_classes, "feat_dim": 768})

    # prepare paths
    CKPT_DIR       = os.path.join(REPO_ROOT, 'checkpoints', args.dataset_name, args.source_domain_name)
    EPISODE_DIR    = os.path.join(REPO_ROOT, f'Episode_all_{args.dataset_name}', args.source_domain_name)
    # DATA_DIR       = os.path.join(REPO_ROOT, 'datasets', args.dataset_name)
    DATA_DIR = "/home/shubhranil/DATASETS/PACS"
    SYN_DATA_PATH  = os.path.join(REPO_ROOT, 'data', f'syn_dataset_path_{args.dataset_name}.csv')
    RESULT_DIR     = os.path.join(REPO_ROOT, 'Results', args.dataset_name, args.source_domain_name)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(EPISODE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # task model path
    task_model_path = os.path.join(
        CKPT_DIR,
        f"DGCD_task_model_{args.source_domain_name}.pkl"
    )

    source_path = os.path.join(DATA_DIR, args.source_domain_name)
    # read target dataset paths
    syn_dataset_paths = []
    with open(SYN_DATA_PATH, newline='') as file:
        for row in csv.DictReader(file):
            syn_dataset_paths.append(row['data_path'])

    # Define a transformation pipeline for the images
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # Load the VITB16 model pre-trained with DINO
    global_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16',pretrained=True)
    global_model.head = nn.Identity()  # Remove the classification head
    global_model=global_model.to(device)

    # Unfreeze the last two transformer blocks
    for param in global_model.parameters():
        param.requires_grad = False
    for param in global_model.blocks[-1].parameters():
        param.requires_grad = True

    save_model(model=global_model,path=task_model_path)
    # Defining a Projection head to help in INFO-NCE loss
    projection_head=DINOHead(in_dim=feat_dim,out_dim=65536,nlayers=3)
    projection_head=projection_head.to(device)

    # In PACS - randomly selecting 4 classes out of 7
    # In Office Home - randomly selecting 40 classes out of 65
    # In Domain Net - randomly selecting 250 classes out of 345
    selected_classes = create_list(source_domain=source_path, num_classes=num_classes)

    # Setup the loss function
    criterion = SupervisedContrastiveLoss()

    # DGCD TRAINING
    for ge in tqdm(range(args.global_epochs), desc="Global Epochs", leave=True):                                      # OUTER LOOP
        print(f"GLOBAL EPOCH: {ge+1}")
        print("----------\n----------")
        # Shuffle the target dataset paths
        random.seed(42)
        random.shuffle(syn_dataset_paths)

        # Initialise a dictionary to store the average differences between the weights of 'Global' and 'Task' Model
        episode_weight_diffs = []
        # Store the accuracy of each task epoch
        all_accuracies = [] 
        
        for episode in tqdm(range(args.episodes), desc="Episodes", leave=False):
            print(f"EPISODE {episode + 1}:\n----------")

            ep_dir = os.path.join(EPISODE_DIR, f"episode_{episode}")
            os.makedirs(ep_dir, exist_ok=True)
            src_csv, syn_csv, num_labelled_classes = create_csv(
                source_domain = source_path,
                aug_domain = syn_dataset_paths[episode],
                csv_dir_path = ep_dir,
                selected_classes = selected_classes,
                episode = episode
            )
            combine_csv_path = os.path.join(ep_dir, f"combined_episode_{episode}.csv")
            combine_csv_files(src_csv, syn_csv, combine_csv_path)

            weight_diff = {name: torch.zeros_like(param) for name, param in global_model.named_parameters() if param.requires_grad}

            combine_dataloader=create_combine_dataloader(combine_csv=combine_csv_path,batch_size= args.batch_size,transform=train_transform)
            task_model=load_model(path=task_model_path)
            task_model.to(device)
            task_optimizer = torch.optim.SGD([
                            {'params': filter(lambda p: p.requires_grad, task_model.parameters()), 'lr': args.task_lr},
                            {'params': projection_head.parameters(), 'lr': args.task_lr}           
                        ],momentum=0.9,weight_decay=5e-5)
            # Define the cosine annealing learning rate scheduler for task_optimizer
            task_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(task_optimizer, T_max=args.task_epochs,eta_min=args.task_lr*1e-3,)

            # Defining a Projection head to help in INFO-NCE loss
            projection_head=DINOHead(in_dim=feat_dim,out_dim=65536,nlayers=3)
            projection_head=projection_head.to(device)
            
            classifier = Classifier(num_classes=num_labelled_classes+1, input_dim=feat_dim).to(device)
            optim_c = torch.optim.SGD(list(classifier.parameters()), lr=args.task_lr, momentum=0.9, weight_decay=5e-5)
            exp_lr_scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim_c,
                T_max=args.task_epochs,
                eta_min=args.task_lr* 1e-3,
            )        

            for epoch in tqdm(range(args.task_epochs), desc=f"Episode {episode + 1}"):                                # TASK EPOCH
                task_model.train()
                projection_head.train()
                classifier.train()
                total_loss = 0
                L_da=0
                total_osda_loss = 0
                total_osda_classifier_loss=0
                total_contrastive_loss = 0
                total_sup_contrastive_loss = 0
                total_margin_loss=0
                total_prob_diff = 0.0
                count_batches = 0
                loss_record = AverageMeter()
                train_acc_record = AverageMeter()
                for batch_idx,batch in tqdm(enumerate(combine_dataloader), desc="Processing Batches", leave=False, total=len(combine_dataloader)):                                            # STEP LOOP
                    images, numerical_labels,mask,mask_known=batch
                    numerical_labels =numerical_labels.to(device)
                    images = torch.cat(images, dim=0).to(device)
                    # Extract features using Task Model
                    image_features = task_model(images)                                                              
                    mask=mask.to(device).bool()                                                         # mask seprates Source from  Synthetic Domain 
                    mask_known=mask_known.to(device).bool()
                    

                    target_funk = Variable(torch.FloatTensor(images.size()[0], 2).fill_(0.5).to(device))    
                    classifier.set_lambda(1.0)
                    out_c_2 = F.softmax(classifier(image_features.detach(),reverse = True),dim=1)       
                    prob1 = torch.sum(out_c_2[:, :num_labelled_classes-1], 1).view(-1, 1)
                    prob2 = out_c_2[:,num_labelled_classes-1].contiguous().view(-1, 1)
                    prob = torch.cat((prob1, prob2), 1)
                    loss_t = bce_loss(prob, target_funk)
                    total_osda_classifier_loss+=loss_t.item()
                    optim_c.zero_grad()
                    loss_t.backward(retain_graph = True)
                    optim_c.step()

                    # 1. Adversarial Loss {L_adv}
                    out_c = classifier(image_features)
                    classifier_labels = torch.zeros_like(numerical_labels)
                    classifier_labels[mask_known] = numerical_labels[mask_known]
                    classifier_labels[~mask_known] = num_labelled_classes                                         # mask_lab apply to filter between known and novel classes
                    classifier_labels = torch.concat([classifier_labels for _ in range(2)])
                    loss_s = nn.CrossEntropyLoss()(out_c, classifier_labels)
                    total_osda_loss+=loss_s.item()
                    
                    
                    #--------PROBABILITY Difference-----#
                    out_c_2 = F.softmax(classifier(image_features[torch.cat([~mask for i in range(2)])].detach(),reverse = True),dim=1)       
                    prob2 = out_c_2[:,num_labelled_classes].contiguous().view(-1, 1)

                    max_vals, _ = torch.max(out_c_2, dim=1)  # Extract max values separately
                    prob_diff = (torch.abs(max_vals) - prob2).view(-1, 1)
                    avg_prob_diff=torch.mean(prob_diff,dim=0)
                    total_prob_diff += avg_prob_diff.item()
                    count_batches += 1
                    # wandb.log({'Batch Average Probability Difference': avg_prob_diff.item()})

                    # Open a file in append mode, so that each new value is added without overwriting existing data
                    log_path = os.path.join(RESULT_DIR, 'average_probability_differences.txt')
                    with open(log_path, "a") as file:
                        # Include epoch and batch information in each line
                        file.write(f"Epoch {epoch + 1} Batch {batch_idx + 1}: {avg_prob_diff.item()}\n")


                    # 2. Margin Loss {L_margin}
                    margin = torch.full_like(prob_diff, m).to(device)
                    # Calculate the margin loss where negative differences are clamped to zero
                    L_margin = torch.clamp(margin - prob_diff, min=0).mean()  # Use .mean() to average over all instances
                    total_margin_loss += L_margin.item()


                    # Projection head is "DINO HEAD"
                    features=projection_head(image_features)
                    # L2-normalize features
                    features = torch.nn.functional.normalize(features, dim=-1)

                    # 3. Unsupervised Contrastive loss {L^u_con}
                    con_feats = features
                    contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, device=device,n_views=args.n_views)                 # contrastive_logits.shape = (128, 256), contrastive_labels.shape = (128)
                    contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)                                      # images with labels from source domain and without labels fronm synthetic domains go to InfoNCE and DANN loss
                    total_contrastive_loss += contrastive_loss.item()
                    
                    # 4. Supervised Contrastive loss {L^s_con}
                    f1, f2 = [f[mask] for f in features.chunk(2)]
                    if np.array(mask.cpu()).any():                      # if labelled images present, then calculate Supervised Contrastive loss
                        sup_con_feats =torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                        sup_con_labels = numerical_labels[mask]
                        sup_contrastive_loss = criterion(sup_con_feats, labels=sup_con_labels)
                    else:
                        sup_contrastive_loss = 0
                    total_sup_contrastive_loss += sup_contrastive_loss.item()


                    # Domain Adversarial Loss {L_da}
                    L_da = loss_s + lamb * L_margin
                    # Total loss
                    loss = 0.65 * contrastive_loss + 0.35 * sup_contrastive_loss + mu * L_da

                    # Train acc
                    _, pred = contrastive_logits.max(1)                         # _ is maximum logit for a sample, pred is the corresponding class having the max logit
                    acc = (pred == contrastive_labels).float().mean().item()
                    train_acc_record.update(acc, pred.size(0))
                    loss_record.update(loss.item(), numerical_labels.size(0))
                    # Log the losses to WandB every 5th batch
                    if batch_idx % 5 == 0:
                        # Log gradients
                        log_gradients(task_model.named_parameters(), "task_model_gradients")
                        # wandb.log({
                        #     f'Batch Loss': loss.item(),
                        #     'Batch Sup Loss': sup_contrastive_loss.item(),
                        #     'Batch Contrastive Loss': contrastive_loss.item(),
                        #     'Batch OSDA LOSS': loss_s.item(),
                        #     'Margin Loss':L_margin.item(),
                        #     'Batch Train Accuracy': acc
                        # })
                    task_optimizer.zero_grad()
                    loss.backward()
                    task_optimizer.step()
                    total_loss += loss.item()
                print('Train Epoch: {} Avg Loss: {:.4f} | Seen Class Acc: {:.4f} '.format(epoch, loss_record.avg, train_acc_record.avg))

                avg_loss = total_loss / len(combine_dataloader)
                avg_contrastive_loss = total_contrastive_loss / len(combine_dataloader)
                avg_sup_contrastive_loss = total_sup_contrastive_loss / len(combine_dataloader)
                avg_osda_loss = total_osda_loss / len(combine_dataloader)
                avg_margin_loss=total_margin_loss/len(combine_dataloader)
                avg_osda_loss_c=total_osda_classifier_loss/len(combine_dataloader)
                # After the batch loop, at the end of each epoch
                epoch_avg_prob_diff = total_prob_diff / count_batches
                # wandb.log({'Epoch Average Probability Difference': epoch_avg_prob_diff})
                print(f'Epoch {epoch + 1}: Average Probability Difference = {epoch_avg_prob_diff}')
                print(f"\nEpoch {epoch} | Avg Loss: {avg_loss:.4f}| Avg Contrastive Loss: {avg_contrastive_loss:.6f}| Avg Sup Contrastive Loss: {avg_sup_contrastive_loss:.6f}| Avg OSDA Loss: {avg_osda_loss:.6f}| Avg Classifer OSDA Loss: {avg_osda_loss_c:.6f} | Avg Margin Loss: {avg_margin_loss:.6f}")
                # Log aggregated metrics to WandB
                # wandb.log({
                #     'Epoch Avg Loss/train': avg_loss,
                #     'Epoch Avg ContrastiveLoss/train': avg_contrastive_loss,
                #     'Epoch Avg SupContrastiveLoss/train': avg_sup_contrastive_loss,
                #     'Epoch Avg OSDALoss/train': avg_osda_loss,
                #     'Epoch Avg Margin Loss': avg_margin_loss,
                #     'Epoch': (ge * args.episodes * args.task_epochs + episode * args.task_epochs + epoch)
                # })
                # Update task_scheduler
                task_scheduler.step()
                exp_lr_scheduler_2.step()
                # wandb.log({'Task Learning Rate': task_scheduler.get_last_lr()[0]})

            torch.cuda.empty_cache()

            validation_dir = os.mkdir(os.path.join(EPISODE_DIR, 'Validation'), exist_ok=True)
            validation_loader = mix_validation_dataloader(
                csv_file=os.path.join(validation_dir, 'combined_validation_set.csv'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                transform=train_transform
            )
            with torch.no_grad():
                print('Testing on Mixed Up Validation test dataset...')
                all_acc_test, old_acc_test, new_acc_test = test_kmeans_mix(device=device,
                                                                            model = task_model, 
                                                                            test_loader = validation_loader, 
                                                                            epoch = epoch, save_name = 'Test ACC', 
                                                                            num_train_classes = num_labelled_classes,
                                                                            num_unlabelled_classes = len(selected_classes)-num_labelled_classes
                                                                            )
            print('Mix Validation Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test,
                                                                                new_acc_test))
            all_accuracies.append(all_acc_test)                 # Using All as evaluation Metrics
            # Compute the difference between the Global Model and the current Task Model
            for name, global_param in global_model.named_parameters():
                if global_param.requires_grad:
                    task_param = task_model.state_dict()[name]
                    difference = global_param.data - task_param.data
                    weight_diff[name] += difference
            episode_weight_diffs.append(weight_diff)
            # wandb.log({'Episode': args.episodes*ge+episode,
            #         "Mix_Old": old_acc_test,
            #         "Mix All": all_acc_test,
            #         "Mix New": new_acc_test
            #         })
        weights = torch.nn.functional.softmax(torch.tensor(all_accuracies), dim=0)      # we consider the 'all' accuracies as the weights of the differences of the weights
        # Initialize a dictionary to store the final weighted updates
        final_weighted_updates = {name: torch.zeros_like(param) for name, param in global_model.named_parameters() if param.requires_grad}
        # Combine differences using weights
        for i, weight in enumerate(weights):
            for name in final_weighted_updates.keys():
                final_weighted_updates[name] += episode_weight_diffs[i][name] * weight
        # Update the parameters of the Global Model with the Average difference     
        for name, param in global_model.named_parameters():
            if param.requires_grad:
                param.data -= final_weighted_updates[name]
        # Log meta loss to WandB
        # wandb.log({'ge': ge})
        save_model(
            model = global_model,
            path = os.path.join(
                CKPT_DIR,
                f"DGCD_Intermediate_{args.source_domain_name}_{ge}epoch.pkl"
            )
        )


if __name__ == "__main__":
    main()