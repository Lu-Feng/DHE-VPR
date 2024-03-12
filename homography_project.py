
import torch
import kornia

import torch.nn.functional as F
import numpy as np
import cv2

def get_keypoints(img_size):
    # flaten by x 
    H,W = img_size
    patch_size = 2**4
    N_h = H//patch_size
    N_w = W//patch_size
    keypoints = np.zeros((2, N_h*N_w), dtype=int) #(x,y)
    keypoints[0] = np.tile(np.linspace(patch_size//2, W-patch_size//2, N_w, 
                                       dtype=int), N_h)
    keypoints[1] = np.repeat(np.linspace(patch_size//2, H-patch_size//2, N_h,
                                         dtype=int), N_w)
    return np.transpose(keypoints)

def match_batch_tensor(fm1, fm2, thetas, trainflag, img_size):
    '''
    fm1: (l,D)
    fm2: (N,l,D)
    mask1: (l)
    mask2: (N,l)
    '''
    M = torch.matmul(fm2, fm1.T) # (N,l,l)
    
    max1 = torch.argmax(M, dim=1) #(N,l)
    max2 = torch.argmax(M, dim=2) #(N,l)
    m = max2[torch.arange(M.shape[0]).reshape((-1,1)), max1] #(N, l)
    valid = torch.arange(M.shape[-1]).repeat((M.shape[0],1)).cuda() == m #(N, l) bool
    
    kps = get_keypoints(img_size) #(l,2)
    scores_or_errors = torch.zeros(fm2.shape[0]).cuda() # images similarity scores for testing / re-projection errors for training
    feat_dis = torch.zeros(fm2.shape[0]).cuda()
    for i in range(fm2.shape[0]):
        idx1 = torch.nonzero(valid[i,:]).squeeze()
        idx2 = max1[i,:][idx1]
        assert idx1.shape==idx2.shape
        if len(idx1.shape)<1 or idx1.shape[0]<4: 
            scores_or_errors[i]=torch.Tensor([0.]).requires_grad_()
            continue
               
        idx1,idx2=idx1.cpu(),idx2.cpu()
        pix1 = np.ones([len(idx1),3])
        pix2 = np.ones([len(idx2),2])
        pix1[:,:-1] = kps[idx1]
        pix2 = kps[idx2]

        pix1 = torch.from_numpy(pix1).to(torch.float32).cuda()
        pix2 = torch.from_numpy(pix2).to(torch.float32).cuda()

        theta = thetas[i] 
        tpix2 = torch.matmul(theta,pix1.T).T
        tmp = tpix2[:,-1]   
        if 0 in tmp:
            zeroindex = (tmp==0).nonzero()
            for k in zeroindex:
                tmp[k]=1e-8
        tmp = torch.repeat_interleave(tmp.unsqueeze(1), 2, 1)
        tpix2norm = tpix2[:,:-1]/tmp
        dis = torch.sqrt(torch.sum((tpix2norm - pix2).pow(2), dim=1))
        if trainflag:
            thetaGT, mask = cv2.findHomography(kps[idx1],kps[idx2], cv2.FM_RANSAC,
                                            ransacReprojThreshold=(2**4)*1.5)
            if thetaGT is None:
                scores_or_errors[i]=torch.Tensor([0.]).requires_grad_()
                continue 

            inliers_ind = dis<(2**4)*1.5

            inliers_num = np.sum(mask)
            if inliers_num==0:
                scores_or_errors[i]=torch.Tensor([0.]).requires_grad_()
                continue
            dis,_ = torch.topk(dis, k=inliers_num, dim=0, largest=False)
            scores_or_errors[i] = torch.mean(dis)
        else:
            scores_or_errors[i] = torch.sum((dis<(2**4)*3))
    if trainflag:
        return scores_or_errors[0]
    else:
        return scores_or_errors

def reprojection_error_ofinliers(model, tensor_img_1, tensor_img_2, weights=None):
    pred_points_1to2, features_1, features_2 = model("similarity_and_regression", [tensor_img_1, tensor_img_2])
    B, C, H, W = features_1.shape
    mean_pred_points_1 = pred_points_1to2[:,:4]
    mean_pred_points_2 = pred_points_1to2[:,4:8]
    thetas = kornia.geometry.homography.find_homography_dlt(mean_pred_points_1, mean_pred_points_2, weights)

    queries = features_1.permute(0, 2, 3, 1)
    preds = features_2.permute(0, 2, 3, 1)
    queries,preds = queries.view(-1,24*24,384),preds.view(-1,24*24,384)
    reproject_error = torch.zeros(B).cuda()
    for i in range(B):
        query,pred = queries[i],preds[i].unsqueeze(0)
        theta = [thetas[i]]
        reproject_error[i] = match_batch_tensor(query, pred, theta, trainflag=True, img_size=(384,384))
    return reproject_error

def compute_score(model, tensor_img_1, tensor_img_2, weights=None):
    pred_points_1to2, features_1, features_2 = model("similarity_and_regression", [tensor_img_1, tensor_img_2])
    B, C, H, W = features_1.shape
    mean_pred_points_1 = pred_points_1to2[:,:4]
    mean_pred_points_2 = pred_points_1to2[:,4:8]
    thetas = kornia.geometry.homography.find_homography_dlt(mean_pred_points_1, mean_pred_points_2, weights)

    query = features_1.permute(0, 2, 3, 1)[0]
    preds = features_2.permute(0, 2, 3, 1)
    query,preds = query.view(24*24,384),preds.view(-1,24*24,384)
    scores = match_batch_tensor(query, preds, thetas, trainflag=False, img_size=(384,384))
    return scores