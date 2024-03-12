
import os
import sys
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torchvision.transforms.functional import hflip

import test
import util
import network
import commons
import homography_project
import dataset_qp  # Used for weakly supervised losses, it yields query-positive pairs
import dataset_geoloc  # Used for testing

import warnings
warnings.filterwarnings('ignore')


def to_cuda(list_):
    """Move to cuda all items of the list."""
    return [item.cuda() for item in list_]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training DHE network parameters
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--optim", type=str,  default="adam",
                        choices=["adam", "sgd"],
                        help="optimizer")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="epochs")
    parser.add_argument("--iterations_per_epoch", type=int, default=2000,
                        help="how many iterations each epoch should last")
    parser.add_argument("--qp_threshold", type=float, default=0.14, #only for pitts30k 
                        help="Threshold distance (in features space) for query-positive pairs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size for features-wise loss")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="num_workers")
    
    # Test parameters
    parser.add_argument("--num_reranked_preds", type=int, default=32,
                        help="number of predictions to re-rank at test time")
    
    # Others
    parser.add_argument("--exp_name", type=str, default="default",
                        help="name of generated folders with logs and checkpoints")
    parser.add_argument("--resume_fe", type=str, default=None,
                        help="path to resume for Feature Extractor")
    parser.add_argument("--resume_hr", type=str, default=None,
                        help="path to resume for Homography Regression")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="treshold distance for positives (in meters)")
    parser.add_argument("--datasets_folder", type=str, default="../datasets",
                        help="path with the datasets")
    parser.add_argument("--dataset_name", type=str, default="pitts30k",
                        help="name of folder with dataset")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed")
    parser.add_argument("--trunc_te", type=int, default=8, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=5, choices=list(range(-1, 14)))

    
    args = parser.parse_args()

    commons.make_deterministic(args.seed)
    
    # Setup
    output_folder = f"logs_trainDHE/{args.exp_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.setup_logging(output_folder)
    logging.info("python " + " ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")
    os.makedirs(f"{output_folder}/checkpoints")
    start_time = datetime.now()
    
    ############### MODEL ###############
    features_extractor = network.FeaturesExtractor(args)
    global_features_dim = 384
    
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

    
    ############### DATASETS & DATALOADERS ###############
    geoloc_test_dataset  = dataset_geoloc.GeolocDataset(args.datasets_folder, args.dataset_name, split="test",
                                                    positive_dist_threshold=args.positive_dist_threshold)
    logging.info(f"Geoloc test set: {geoloc_test_dataset}")

    geoloc_train_dataset = dataset_geoloc.GeolocDataset(args.datasets_folder, args.dataset_name, split="train",
                                                    positive_dist_threshold=args.positive_dist_threshold)
    logging.info(f"Geoloc train set: {geoloc_train_dataset}")       
    dataset_qp = dataset_qp.DatasetQP(model, global_features_dim, geoloc_train_dataset, qp_threshold=args.qp_threshold, dataset_name = args.dataset_name)
    dataloader_qp = commons.InfiniteDataLoader(dataset_qp, shuffle=True,batch_size=args.batch_size,
                                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
    data_iter_qp = iter(dataloader_qp)

    
    ############### LOSS & OPTIMIZER ###############
    mse = torch.nn.MSELoss()
    if args.optim == "adam":
        optim = torch.optim.Adam(homography_regression.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10000, gamma=0.8, last_epoch=-1)
        #optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optim = torch.optim.SGD(homography_regression.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

    
    ############### TRAIN ###############
    for epoch in range(args.n_epochs):
        
        homography_regression = homography_regression.train()
        total_loss = 0.
        for iteration in tqdm(range(args.iterations_per_epoch), desc=f"Train epoch {epoch}", ncols=100):
            queries, positives = to_cuda(next(data_iter_qp))
           
            optim.zero_grad()
            
            queries_fw = queries[:args.batch_size]
            positives_fw = positives[:args.batch_size]
            random_weights = (torch.rand(args.batch_size, 4)**0.1).cuda()
            queries_fw = model("features_extractor", queries_fw.cuda())
            positives_fw = model("features_extractor", positives_fw.cuda())
            REIloss = homography_project.reprojection_error_ofinliers(model, queries_fw, positives_fw, weights=random_weights)

            loss = torch.mean(REIloss)
            loss.backward()

            del queries, positives, queries_fw, positives_fw
            
            total_loss += loss
            optim.step()
            scheduler.step()
            if (iteration+1) % 500 == 0:
                logging.debug(f"Current loss = {loss:.6f}")
                logging.debug(f"Average loss of 500 iterations = {total_loss/500.:.6f}")
                total_loss = 0.

        if epoch%10 == 9:
            torch.save(homography_regression.state_dict(),
                    f"{output_folder}/checkpoints/homography_regression_{epoch:03d}.torch")
    
    ############### TEST ###############
    logging.info(f"The training is over in {str(datetime.now() - start_time)[:-7]}, now it's test time")
    
    homography_regression = homography_regression.eval()
    
    test_baseline_recalls, test_baseline_recalls_pretty_str, test_baseline_predictions, _, _, query_img, gallery_img = \
            util.compute_features(geoloc_test_dataset, model, global_features_dim)
    logging.info(f"baseline test: {test_baseline_recalls_pretty_str}")
    _, reranked_test_recalls_pretty_str = test.test(model, test_baseline_predictions, query_img, gallery_img, geoloc_test_dataset,
                                                    num_reranked_predictions=args.num_reranked_preds, recall_values=[1,5,10,20])
    logging.info(f"test after re-ranking - {reranked_test_recalls_pretty_str}")
    ############### TEST ###############

