
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup

import homography_project
import util
import test
import dataset_geoloc
import parser
import commons
import datasets_ws
import network
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs_finetuning", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")


#### Creation of Datasets
logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
logging.info(f"Train query set: {triplets_ds}")

val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
logging.info(f"Test set: {test_ds}")


############### MODEL ###############
features_extractor = network.FeaturesExtractor(args)
args.features_dim = global_features_dim = 384#*64

if args.resume_fe != None:
    # state_dict = torch.load(args.resume_fe)
    # features_extractor.load_state_dict(state_dict)
    state_dict = torch.load(args.resume_fe)["model_state_dict"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace("module.backbone","encoder")
        name = k.replace("module.aggregation","pool")
        new_state_dict[name] = v
    features_extractor.load_state_dict(new_state_dict)
else:
    logging.warning("WARNING: --resume_fe is set to None, meaning that the "
                    "Feature Extractor is not initialized!")

homography_regression = network.HomographyRegression()

if args.resume_hr != None:
    state_dict = torch.load(args.resume_hr)
    homography_regression.load_state_dict(state_dict)

model = network.Network(features_extractor, homography_regression).cuda().eval()
model = torch.nn.DataParallel(model)


### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

best_r5 = start_epoch_num = not_improved_num = 0


#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
    
    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")
        
        # Compute triplets to use in the triplet loss
        triplets_ds.is_inference = True
        triplets_ds.compute_triplets(args, model)
        triplets_ds.is_inference = False
        
        triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                 batch_size=args.train_batch_size,
                                 collate_fn=datasets_ws.collate_fn,
                                 pin_memory=(args.device=="cuda"),
                                 drop_last=True)
        
        model = model.train()
        # images shape: (train_batch_size*4)*3*H*W
        # triplets_local_indexes shape: (train_batch_size*2)*3 ; because 2 triplets per query
        for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):
            
            # Flip all triplets or none
            if args.horizontal_flip:
                images = transforms.RandomHorizontalFlip()(images)
            
            # Compute features of all images (images contains queries, positives and negatives)
            
            features, features_a = model("features_extractor", [images.to(args.device), "global"])
            loss_triplet = 0

            if args.criterion == "triplet":
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                for triplets in triplets_local_indexes:                
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T
                    loss_triplet += criterion_triplet(features[queries_indexes],
                                                      features[positives_indexes],
                                                      features[negatives_indexes])

                random_weights = (torch.rand(args.train_batch_size, 4)**0.1).cuda()
                REIloss = homography_project.reprojection_error_ofinliers(model, features_a[queries_indexes], features_a[positives_indexes], weights=random_weights)
                loss_REI = torch.sum(REIloss)
                       
            del features
            del features_a 
            loss_triplet /= (args.train_batch_size * args.negs_num_per_query)
            loss_REI /= args.train_batch_size
            loss_joint = loss_triplet + 100*loss_REI

            optimizer.zero_grad()
            loss_joint.backward()
            optimizer.step()

            batch_loss = loss_joint.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss_joint
        
        print("triplet loss:",loss_triplet.item())
        print("REI loss:",loss_REI.item())
        logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                      f"current batch joint loss = {batch_loss:.4f}, " +
                      f"average epoch joint loss = {epoch_losses.mean():.4f}")
    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch joint loss = {epoch_losses.mean():.4f}")


    torch.save(homography_regression.state_dict(),
                f"{args.save_dir}/homography_regression_{epoch_num:03d}.torch")
    torch.save(features_extractor.state_dict(),
                f"{args.save_dir}/features_extractor_{epoch_num:03d}.torch")


    ############### TEST ###############
    geoloc_test_dataset  = dataset_geoloc.GeolocDataset(args.datasets_folder, args.dataset_name, split="val",
                                                    positive_dist_threshold=args.val_positive_dist_threshold)
    logging.info(f"Geoloc test set: {geoloc_test_dataset}")
    logging.info(f"now it's test time")
    test_baseline_recalls, test_baseline_recalls_pretty_str, test_baseline_predictions, _, _, query_img, gallery_img = \
            util.compute_features(geoloc_test_dataset, model, global_features_dim)
    logging.info(f"baseline test: {test_baseline_recalls_pretty_str}")
    if args.dataset_name =="msls" or (args.dataset_name =="pitts30k" and epoch_num>=35):
        _, reranked_test_recalls_pretty_str = test.test(model, test_baseline_predictions, query_img, gallery_img, geoloc_test_dataset,
                                                        num_reranked_predictions=args.num_reranked_preds, recall_values=[1,5,10,20])
        logging.info(f"test after re-ranking - {reranked_test_recalls_pretty_str}")

