import os
import sys
import torch
import logging
from datetime import datetime

import parser
import test
import util
import network
import commons
import dataset_geoloc  # Used for testing
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    args = parser.parse_arguments()
    
    # Setup
    output_folder = f"logs_test/{args.exp_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.setup_logging(output_folder, console="info")
    logging.info("python " + " ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")
    os.makedirs(f"{output_folder}/checkpoints")
    start_time = datetime.now()
    
    ############### MODEL ###############
    features_extractor = network.FeaturesExtractor(args)
    global_features_dim = 384

    homography_regression = network.HomographyRegression()
    
    if args.resume_fe != None:
        state_dict = torch.load(args.resume_fe)
        features_extractor.load_state_dict(state_dict)
    else:
        logging.warning("WARNING: --resume_fe is set to None, meaning that the "
                        "Feature Extractor is not initialized!")
    if args.resume_hr != None:
        state_dict = torch.load(args.resume_hr)
        homography_regression.load_state_dict(state_dict)
        del state_dict
    else:
        logging.warning("WARNING: --resume_hr is set to None, meaning that the "
                        "Homography Regression is not initialized!")
    
    model = network.Network(features_extractor, homography_regression).cuda().eval()
    model = torch.nn.DataParallel(model)


    ############### DATASETS & DATALOADERS ###############
    geoloc_test_dataset  = dataset_geoloc.GeolocDataset(args.datasets_folder, args.dataset_name, split="test",
                                                    positive_dist_threshold=args.val_positive_dist_threshold)
    logging.info(f"Geoloc test set: {geoloc_test_dataset}")
    ############### DATASETS & DATALOADERS ###############
    
    ############### TEST ###############
    test_baseline_recalls, test_baseline_recalls_pretty_str, test_baseline_predictions, _, _, query_img, gallery_img = \
            util.compute_features(geoloc_test_dataset, model, global_features_dim)
    logging.info(f"baseline test: {test_baseline_recalls_pretty_str}")
    _, reranked_test_recalls_pretty_str = test.test(model, test_baseline_predictions, query_img, gallery_img, geoloc_test_dataset,
                                                    num_reranked_predictions=args.num_reranked_preds, recall_values=[1,5,10,20], test_batch_size=args.infer_batch_size)
    logging.info(f"test after re-ranking - {reranked_test_recalls_pretty_str}")
    ############### TEST ###############
    
