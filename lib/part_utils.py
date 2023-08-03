import cv2, math, os
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import shutil


def box(image, save_name, position, width=20, color_type='random'):
    h, w, c = image.shape
    assert position.min() >= 0
    assert position[:, 0].max() <= h and position[:, 2].max() <= h
    assert position[:, 1].max() <= w and position[:, 3].max() <= w
    for p in position:
        x1, y1, x2, y2 = p.astype(int)
        delta = width // 2
        if color_type == 'random':
            # color=plt.cm.Accent(np.random.randn()*100)
            # color=np.array(color[:3])*255
            color = np.array(np.random.rand(3)) * 255
        image[max(x1 - delta, 0):min(x1 + delta, h), max(y1 - delta, 0):min(y2 + delta, w)] = color
        image[max(x2 - delta, 0):min(x2 + delta, h), max(y1 - delta, 0):min(y2 + delta, w)] = color
        image[max(x1 - delta, 0):min(x2 + delta, h), max(y1 - delta, 0):min(y1 + delta, w)] = color
        image[max(x1 - delta, 0):min(x2 + delta, h), max(y2 - delta, 0):min(y2 + delta, w)] = color
    cv2.imwrite(save_name, image)


def box2(image, save_name, position, width=20, color_type='random'):
    h, w, c = image.shape
    assert position.min() >= 0
    assert position[:, 0].max() <= h and position[:, 2].max() <= h
    assert position[:, 1].max() <= w and position[:, 3].max() <= w
    image_null = np.ones_like(image)
    for p in position:
        x1, y1, x2, y2 = p.astype(int)
        delta = width
        if color_type == 'random':
            # color=plt.cm.Accent(np.random.randn()*100)
            # color=np.array(color[:3])*255
            color = np.array(np.random.rand(3)) * 255
        cv2.rectangle(image, (y1, x1), (y2, x2), color, width)
        cv2.rectangle(image_null, (y1, x1), (y2, x2), color, width)

    cv2.imwrite(save_name, image)
    cv2.imwrite(save_name.replace('.png', 'null.png').replace('.jpg', 'null.png'), image_null)


def my_rectangle(image, x1, x2, y1, y2, color):
    patch = image[x1:x2, y1:y2, :]
    blk = np.ones_like(patch)
    blk[:, :, :] = color
    image[x1:x2, y1:y2, :] = cv2.addWeighted(patch, 0.8, blk, 0.2, 1)
    return image


def box3(image, save_name, position, width=20, color_type='random'):
    h, w, c = image.shape
    assert position.min() >= 0
    assert position[:, 0].max() <= h and position[:, 2].max() <= h
    assert position[:, 1].max() <= w and position[:, 3].max() <= w
    image_null = np.zeros_like(image)
    for p in position:
        x1, y1, x2, y2 = p.astype(int)
        if color_type == 'random':
            # color=plt.cm.Accent(np.random.randn()*100)
            # color=np.array(color[:3])*255
            color = np.array(np.random.rand(3)) * 255
        image = my_rectangle(image, x1, x2, y1, y2, color)
        image_null = my_rectangle(image_null, x1, x2, y1, y2, color)
    cv2.imwrite(save_name, image)
    cv2.imwrite(save_name.replace('.png', 'null.png').replace('.jpg', 'null.png'), image_null)


def random_points(image, color, ratio):
    temp = np.random.rand(image.shape[0], image.shape[1])
    index = temp < ratio
    image[index] = color
    return image


def draw_sample_points_line(image, x1, x2, y1, y2, color, ratio=0.4):
    patch = image[x1:x2, y1:y2]
    line_patch = random_points(patch, color, ratio)
    image[x1:x2, y1:y2] = patch
    return image


def box4(image, save_name, position, width=20, color_type='random'):
    h, w, c = image.shape
    assert position.min() >= 0
    assert position[:, 0].max() <= h and position[:, 2].max() <= h
    assert position[:, 1].max() <= w and position[:, 3].max() <= w
    image_null = np.ones_like(image)*255
    for p in position:
        x1, y1, x2, y2 = p.astype(int)
        delta = width // 2
        if color_type == 'random':
            # color=plt.cm.Accent(np.random.randn()*100)
            # color=np.array(color[:3])*255
            color = np.array(np.random.rand(3)) * 255

        image = draw_sample_points_line(image, max(x1 - delta, 0), min(x1 + delta, h), max(y1 - delta, 0),
                                        min(y2 + delta, w), color)
        image = draw_sample_points_line(image, max(x2 - delta, 0), min(x2 + delta, h), max(y1 - delta, 0),
                                        min(y2 + delta, w), color)
        image = draw_sample_points_line(image, max(x1 - delta, 0), min(x2 + delta, h), max(y1 - delta, 0),
                                        min(y1 + delta, w), color)
        image = draw_sample_points_line(image, max(x1 - delta, 0), min(x2 + delta, h), max(y2 - delta, 0),
                                        min(y2 + delta, w), color)

        image_null = draw_sample_points_line(image_null, max(x1 - delta, 0), min(x1 + delta, h), max(y1 - delta, 0),
                                        min(y2 + delta, w), color)
        image_null = draw_sample_points_line(image_null, max(x2 - delta, 0), min(x2 + delta, h), max(y1 - delta, 0),
                                        min(y2 + delta, w), color)
        image_null = draw_sample_points_line(image_null, max(x1 - delta, 0), min(x2 + delta, h), max(y1 - delta, 0),
                                        min(y1 + delta, w), color)
        image_null = draw_sample_points_line(image_null, max(x1 - delta, 0), min(x2 + delta, h), max(y2 - delta, 0),
                                        min(y2 + delta, w), color)
    cv2.imwrite(save_name, image)
    # cv2.imwrite(save_name.replace('.png', 'null.png').replace('.jpg', 'null.png'), image_null)



def part(image, save_folder, position, row_col, name_pre='', save_orign_blocks=True, save_scale_blocks=True, scale_size=512, cover=False):
    assert len(position) == len(row_col)
    if save_orign_blocks or save_scale_blocks:
        save_image_folder = os.path.join(save_folder, 'part_images')
        save_scale_image_folder=os.path.join(save_folder, 'part_images_scale')
        
        if save_orign_blocks:
            if not os.path.exists(save_image_folder):
                os.makedirs(save_image_folder)
            else:
                if cover:
                    shutil.rmtree(save_image_folder)
                    os.makedirs(save_image_folder)
                    
        if save_scale_blocks:
            if not os.path.exists(save_scale_image_folder):
                os.makedirs(save_scale_image_folder)
            else:
                if cover: 
                    shutil.rmtree(save_scale_image_folder)
                    os.makedirs(save_scale_image_folder)
            
        for i in range(position.shape[0]):
            x1, y1, x2, y2 = position[i].astype(int)
            row, col = row_col[i].astype(int)
            cur_name = name_pre + '_' + str(row) + '_' + str(col) + '_' + str(x1) + '_' + str(y1) + '_' + str(
                x2) + '_' + str(y2) + '.jpg'
            if save_orign_blocks:
                cv2.imwrite(os.path.join(save_image_folder, cur_name), image[x1:x2, y1:y2])
            if save_scale_blocks:
                scale_block=cv2.resize(image[x1:x2, y1:y2], (scale_size, scale_size))
                cv2.imwrite(os.path.join(save_scale_image_folder, cur_name), scale_block) 
    np.save(os.path.join(save_folder, 'position.npy'), position)
    np.save(os.path.join(save_folder, 'row_col.npy'), row_col)

def part_image(image_path, save_folder, humanheight_min, humanheight_max, h_start, h_end, image_name='part.jpg',
            cor=2, only_part_params=False, save_orign_blocks=True, save_scale_blocks=True, scale_size=512, visual=True, image_shape=None, cover=False):
    '''
    h_start sightly less than the min position height of human
    h_end sightly greater than the max position height of human
    mode: difference, ratio

    '''

    # compute part position
    interval_1 = cor * humanheight_min
    interval_n = cor * humanheight_max

    num = 0
    min_diff=9999999
    final_diff=0
    q_use=0
    for n in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
        def temp_func(x):
            return interval_1 * (1 - x ** n) - (1 - x) * (h_end - h_start)
        q = fsolve(temp_func, [3])
        an=interval_1 * q ** (n-1)
        diff=abs(an-interval_n)
        if diff < min_diff:
            min_diff=diff
            final_diff=an-interval_n
            num=n
            q_use=q
 
    
    interval_list = [int(interval_1 * q_use ** i) for i in range(num)]
    # print( 'the last interval add %.2f%%, num %s' % (final_diff/interval_n * 100, num))
    position_h_list = [h_start]
    for ii in range(num - 1):
        position_h_list.append(position_h_list[-1] + interval_list[ii])
    position_h_part_start, position_h_part_end = [], []
    for ii in range(num - 1):
        position_h_part_start.append(position_h_list[ii])
        position_h_part_end.append(position_h_list[ii + 1])
        position_h_part_start.append(position_h_list[ii] + interval_list[ii] // 2)
        position_h_part_end.append(position_h_list[ii + 1] + interval_list[ii + 1] // 2)


    position_h_part_start.append(position_h_list[-1])
    position_h_part_end.append(h_end)

    if image_shape is not None and only_part_params:
        image_h, image_w= image_shape
    else:
        # load image
        image = cv2.imread(image_path)  # [...,::-1]
        image_h, image_w, _ = image.shape
        assert image_h >= h_end


    part_cor = []
    row_col = []
    for row in range(len(position_h_part_start)):
        size = position_h_part_end[row] - position_h_part_start[row]
        left = 0
        col = 0
        while left + size <= image_w:
            part_cor.append([position_h_part_start[row], left, position_h_part_end[row], left + size])
            row_col.append([row, col])
            left += size // 2
            col += 1
        if abs(left + size // 2 - image_w) > 2:
            part_cor.append([position_h_part_start[row], image_w - size, position_h_part_end[row], image_w])
            row_col.append([row, col])
    part_cor = np.array(part_cor).astype(int)
    row_col = np.array(row_col).astype(int)

    if only_part_params:
        return part_cor, row_col

    name_pre = image_path.split('/')[-1]
    name_pre = name_pre.split('.')[0]
    os.makedirs(save_folder, exist_ok=True)
    part(image, save_folder, part_cor, row_col, name_pre=name_pre, save_orign_blocks=save_orign_blocks, save_scale_blocks=save_scale_blocks, scale_size=scale_size, cover=cover)
    if visual:
        box4(image, os.path.join(save_folder, image_name), part_cor)
    

