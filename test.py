import torch
import numpy as np
from tqdm import tqdm

import util
import homography_project

def test(model, predictions, query_img, gallery_img, test_dataset, num_reranked_predictions=5,
         recall_values=[1,5,10,20], test_batch_size=8):
    """Compute the test by warping the query-prediction pairs.
    
    Parameters
    ----------
    model : network.Network
    predictions : np.array of int, containing the first 20 predictions for each query,
        with shape [queries_num, 20].
    test_dataset : dataset_geoloc.GeolocDataset, which contains the test-time images (queries and gallery).
    num_reranked_predictions : int, how many predictions to re-rank.
    recall_values : list of int, recalls to compute (e.g. R@1, R@5...).
    test_batch_size : int.
    
    Returns
    -------
    recalls : np.array of int, containing R@1, R@5, r@10, r@20.
    recalls_pretty_str : str, pretty-printed recalls
    """ 
    model = model.eval()
    reranked_predictions = predictions.copy()
    with torch.no_grad():
        for num_q in tqdm(range(test_dataset.queries_num), desc="Testing", ncols=100):
            dot_prods_wqp = np.zeros((num_reranked_predictions))
            query_path = test_dataset.queries_paths[num_q]
            query = query_img[num_q].cuda()
            for i1 in range(0, num_reranked_predictions, test_batch_size):
                batch_indexes = list(range(num_reranked_predictions))[i1:i1+test_batch_size]
                current_batch_size = len(batch_indexes)
                query_repeated_twice = torch.repeat_interleave(query.unsqueeze(0), current_batch_size, 0)
                preds = gallery_img[predictions[num_q, batch_indexes]].cuda()
                similarity = homography_project.compute_score(model, query_repeated_twice, preds)
                dot_prods_wqp[i1:i1+test_batch_size] = similarity.cpu().numpy()            
            reranking_indexes = dot_prods_wqp.argsort()[::-1]
            reranked_predictions[num_q, :num_reranked_predictions] = predictions[num_q][reranking_indexes]
 
    ground_truths = test_dataset.get_positives()
    recalls, recalls_pretty_str = util.compute_recalls(reranked_predictions, ground_truths, test_dataset, recall_values)
    return recalls, recalls_pretty_str

