import os, cv2, pickle
import numpy as np

def compare_replace(cur, cur_xy, L, L_xy, tre_ratio=0.1):   # tre_ratio=0.1
    '''
    return: 0 add cur, 1 delete cur, 2 delete pre
    '''
    if len(L) == 0:
        return 0, None, None
    cur = np.reshape(cur, [1, 2])
    data = np.array(L) - cur
    dist = np.sum(data ** 2, 1) ** 0.5
    min_ = np.min(dist)

    cur_x1, cur_y1, cur_x2, cur_y2 = cur_xy
    cur_size = np.abs(cur_x2 - cur_x1)
    tre = cur_size * tre_ratio
    if min_ < tre:
        min_id = np.where(dist == min_)[0][0]
        pre_x1, pre_y1, pre_x2, pre_y2 = L_xy[min_id]
        pre_cx, pre_cy = L[min_id][0] - pre_x1, L[min_id][1] - pre_y1
        cur_cx, cur_cy = cur[0][0] - cur_x1, cur[0][1] - cur_y1
        pre_size = np.abs(pre_x2 - pre_x1)
        if pre_x1 < cur_x1:  # up
            # last_d = pre_size - pre_cx
            # cur_d = cur_cx
            last_d = (pre_size - pre_cx)/pre_size
            cur_d = cur_cx/cur_size 
            if cur_d > last_d:
                L = L[:min_id] + L[min_id + 1:]
                L_xy = L_xy[:min_id] + L_xy[min_id + 1:]
                return 2, L, L_xy
            else:
                return 1, None, None
        elif pre_x1 == cur_x1:  # left
            # last_d = pre_size - pre_cy
            # cur_d = cur_cy
            last_d = (pre_size - pre_cy)/pre_size
            cur_d = cur_cy/cur_size
            if cur_d > last_d:
                L = L[:min_id] + L[min_id + 1:]
                L_xy = L_xy[:min_id] + L_xy[min_id + 1:]
                return 2, L, L_xy
            else:
                return 1, None, None
        else:
            print("pre_x1 > cur_x1")
    else:
        return 0, None, None

def merge_return_cid(position_path, row_col_path, name_pre, all_results, mask_path='', mode='bc'):
    '''
    mode == 'bc' or 'tc'   corresponding to body center or torso center.
    '''
    center_types={'bc':'center', 'tc': 'torso_center'}
    results=all_results.copy()
    position = np.load(position_path)
    row_col = np.load(row_col_path)
    cps2Id_dict={}
    x2row_dict={}
    y2col_dict={}
    center_points = []
    center_points_xy = []
    last_row = -1
    cur_row_cps = []
    last_row_cps = []
    cur_row_xy = []
    last_row_xy = []
    cur_cps = []
    cur_xy = []
    last_size = 0
    total_cps_num = 0
    
    scene_mask=None
    if mask_path != '':
        scene_mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


    for i in range(position.shape[0]):
        x1, y1, x2, y2 = position[i].astype(int)
        row, col = row_col[i].astype(int)
        x2row_dict[str(x1) + '_' + str(x2)] = row
        y2col_dict[str(y1) + '_' + str(y2)] = col
        cur_name = name_pre + '_' + str(row) + '_' + str(col) + '_' + str(x1) + '_' + str(y1) + '_' + str(
            x2) + '_' + str(
            y2) + '.jpg'
        
        if last_row < row:  # new line
            center_points.append(cur_row_cps)
            center_points_xy.append(cur_row_xy)
            last_row += 1
            last_row_cps = cur_row_cps
            cur_row_cps = []
            last_row_xy = cur_row_xy
            cur_row_xy = []

        cur_cps = []
        cur_xy = []
        # if row>5:
        #     print(cur_name)

        if cur_name in results:

            cur_results = results[cur_name]
            for c_id in range(len(cur_results)):
                total_cps_num += 1
                # c_y, c_x=cur_results[c_id]['torso_center'][:2]
                # c_y, c_x= cur_results[c_id]['center'][:2]
                c_y, c_x= cur_results[c_id][center_types[mode]][:2]

                c_x = x1 + c_x
                c_y = y1 + c_y
                
                if scene_mask is not None and scene_mask[c_x, c_y]==0:
                    continue
                
                
                cur = np.array([c_x, c_y])
                # print('cps2Id_dict',str(c_x)+'_'+str(c_y)+'_'+str(x1)+'_'+str(y1)+'_'+str(x2)+'_'+str(y2))
                cps2Id_dict[str(c_x)+'_'+str(c_y)+'_'+str(x1)+'_'+str(y1)+'_'+str(x2)+'_'+str(y2)]=c_id

                add_flag = 0
                # left
                r1, a, b = compare_replace(cur, (x1, y1, x2, y2), cur_row_cps, cur_row_xy)
                if r1 == 0:  # add cur
                    add_flag = 1

                elif r1 == 1:  # delete cur
                    pass
                elif r1 == 2:  # delete pre and add cur
                    add_flag = 2
                    cur_row_cps, cur_row_xy = a, b
                    cur_cps.append(cur)
                    cur_xy.append((x1, y1, x2, y2))


                # up
                r2, c, d = compare_replace(cur, (x1, y1, x2, y2), last_row_cps, last_row_xy)
                if r2 == 0:  # not relate
                    pass
                elif r2 == 1:  # delete cur
                    add_flag = 0
                elif r2 == 2:  # delete pre
                    last_row_cps, last_row_xy = c, d
                    center_points[-1] = last_row_cps
                    center_points_xy[-1] = last_row_xy
                    # if r1 == 1:
                    #     add_flag=3
                    #     cur_cps.append(cur)
                    #     cur_xy.append((x1, y1, x2, y2))

                if add_flag == 1:
                    cur_cps.append(cur)
                    cur_xy.append((x1, y1, x2, y2))


            cur_row_cps = cur_row_cps + cur_cps
            cur_row_xy = cur_row_xy + cur_xy
        if i == position.shape[0] - 1:
            center_points.append(cur_row_cps)
            center_points_xy.append(cur_row_xy)
    final_center_points = []
    final_center_points_xy = []
    for i in range(len(center_points)):
        final_center_points = final_center_points + center_points[i]
        final_center_points_xy = final_center_points_xy + center_points_xy[i]
    # print(len(final_center_points), total_cps_num)
    return final_center_points, final_center_points_xy, total_cps_num, cps2Id_dict, x2row_dict, y2col_dict


def extract_results_after_merged(final_center_points, final_center_points_xy, cps2id_dict, all_results, x2row_dict, y2col_dict, name_pre):
    new_results={}
    
    for i in range(len(final_center_points)):
        x, y=final_center_points[i]
        x1,y1,x2,y2=final_center_points_xy[i]

        row1=x2row_dict[str(x1) + '_' + str(x2)]
        col1=y2col_dict[str(y1) + '_' + str(y2)]
        cur_name = name_pre + '_' + str(row1) + '_' + str(col1) + '_' + str(x1) + '_' + str(y1) + '_' + str(
            x2) + '_' + str(
            y2) + '.jpg'

        if cur_name not in new_results:
            new_results[cur_name]=[]
    
   
        cid=cps2id_dict[str(x)+'_'+str(y)+'_'+str(x1)+'_'+str(y1)+'_'+str(x2)+'_'+str(y2)]
        new_results[cur_name].append(all_results[cur_name][cid])
    return new_results