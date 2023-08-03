
from typing import Dict

import torch

from lib.vis.vis_ground import vis_src_and_target_point

def get_projection_loss(
    xb_gt:torch.Tensor,
    xt_gt:torch.Tensor,
    xt_pred:torch.Tensor,
    vis_data_pack:Dict={'flag': False}
    ) -> (torch.Tensor):
    '''
    input :
        * xb_gt [3, n] each col [u, v, 1]
        * xt_gt [3, n] each col [u, v, 1]
        * xt_pred [3, n] each col [`u, v, 1]
    '''

    '''
    k is the direction vector of (xt_pred - xb_gt)
    n is perpendicular to vec

    then xt_pred = a*k + b*n (a, b is scalar)
    loss = f(b) + f(a - 1.45)
    '''

    #ignore 1 of [u, v, 1]
    xb_gt = xb_gt[0:2, :]
    xt_gt = xt_gt[0:2, :]
    xt_pred = xt_pred[0:2, :]

    loss_vec = 0

    vecs_pg = xt_gt - xb_gt
    vecs_pq = xt_pred - xb_gt
    bias = vecs_pg - vecs_pq
    
    mods_bias = torch.norm(bias, p=2, dim=0)
    mods_pg = torch.norm(vecs_pg, p=2, dim=0)
    mods_pq = torch.norm(vecs_pq, p=2, dim=0)

    loss_mod = torch.mean(torch.abs(mods_pq - mods_pg) / mods_pg)
    loss_pixel = torch.mean(mods_bias / mods_pg)

    if vis_data_pack['flag'] :
        vis_src_and_target_point(
            vis_data_pack,
            point_src_uv1t=xt_pred.detach().cpu().numpy(),
            point_target_uv1t=xt_gt.detach().cpu().numpy(),
            point_ref_uv1t=xb_gt.detach().cpu().numpy()
        )

    return loss_vec, loss_mod, loss_pixel


    for i in range(xb_gt.shape[1]) :
        '''
        P: xb_gt
        G: xt_gt
        Q: xt_pred
        
        vec: vector
        k: direction vector
        n: normal of vector
        '''

        vec_pg = (xt_gt[:, i] - xb_gt[:, i]) #[2]
        vec_pq = (xt_pred[:, i] - xb_gt[:, i]) #[2]

        #print(vec_pg, vec_pq)

        k_pg = vec_pg / torch.norm(vec_pg, p=2)
        #n_pg = torch.zeros_like(k_pg)
        #n_pg[0] = k_pg[1]
        #n_pg[1] = -k_pg[0]

        k_pq = vec_pq / torch.norm(vec_pq, p=2)

        #pq = a*k + b*n
        #a = torch.dot(vec_pq, k_pg)
        #b = torch.dot(vec_pq, n_pg)

        #print(vec_pg, vec_pq)
        #print(a, b, torch.norm(vec_pg, p=2))

        bias = (vec_pg - vec_pq)

        #loss += torch.norm(bias, p=2)
        #print(torch.abs(b), torch.norm(bias, p=2))
        #theta = torch.arccos(torch.dot(vec_pg, vec_pq) / torch.norm(vec_pg, p=2) / torch.norm(vec_pq, p=2))
        #print(torch.arccos(torch.dot(vec_pg, vec_pg) / torch.norm(vec_pg, p=2) / torch.norm(vec_pg, p=2)))
                      
        #print('bias vec_pg',bias, vec_pg)

        loss_term_1 = torch.norm(bias, p=2) / torch.norm(vec_pg, p=2)
        loss_term_2 = ((torch.norm(vec_pg, p=2) - torch.norm(vec_pq, p=2)) / torch.norm(vec_pg, p=2)) ** 2
        loss_term_3 = torch.sum((k_pq - k_pg) ** 2)
        
        loss_vec += loss_term_3
        loss_mod += loss_term_2
        loss_pixel += loss_term_1
        

    


    loss_vec /= xb_gt.shape[1]
    loss_mod /= xb_gt.shape[1]
    loss_pixel /= xb_gt.shape[1]
    


    return loss_vec, loss_mod, loss_pixel

    
