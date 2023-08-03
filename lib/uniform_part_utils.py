import os, cv2
import numpy as np

def part(image, save_folder, position, row_col, name_pre=''):
    assert len(position) == len(row_col)
    # print(position.shape, row_col.shape)
    save_image_folder = os.path.join(save_folder, 'part_images')
    if not os.path.exists(save_image_folder):
        os.makedirs(save_image_folder)
    # print('*****************', save_image_folder)
    for i in range(position.shape[0]):
        x1, y1, x2, y2 = position[i].astype(int)
        row, col = row_col[i].astype(int)
        cur_name = name_pre + '_' + str(row) + '_' + str(col) + '_' + str(x1) + '_' + str(y1) + '_' + str(
            x2) + '_' + str(y2) + '.jpg'
        # print(os.path.join(save_folder, cur_name))
        # print(save_image_folder, cur_name)
        # print(os.path.join(save_image_folder, cur_name))
        cv2.imwrite(os.path.join(save_image_folder, cur_name), image[x1:x2, y1:y2])
    np.save(os.path.join(save_folder, 'position.npy'), position)
    np.save(os.path.join(save_folder, 'row_col.npy'), row_col)

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

def box5(image, save_name, position, width=20, color_type='random'):
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
        if color_type == 'white':
            color=np.array([255, 255, 255.])

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


def uniform_part_overlap(image_path, max_height, part_flag=False, save_folder='', visual_flag=False):
    image = cv2.imread(image_path)  # [...,::-1]
    image_h, image_w, _ = image.shape

    num_along_h=image_h//int(max_height)
    crop_size=image_h // num_along_h


    position_h_part_start, position_h_part_end = [], []

    for ii in range(num_along_h - 1):
        start=ii * crop_size
        position_h_part_start.append(start)
        position_h_part_end.append(start+crop_size)
        position_h_part_start.append(start + crop_size // 2)
        position_h_part_end.append(position_h_part_start[-1] + crop_size)

    position_h_part_start.append(image_h-crop_size)
    position_h_part_end.append(image_h)

    part_cor = []
    row_col = []
    for row in range(len(position_h_part_start)):
        size = crop_size
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
    if part_flag or visual_flag:
        os.makedirs(save_folder, exist_ok=True)
    if part_flag:
        name_pre=os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
        part(image, save_folder, part_cor, row_col, name_pre=name_pre)
    if visual_flag:
        box5(image, os.path.join(save_folder, 'visual_'+os.path.basename(image_path)), part_cor, width=5, color_type='white')

    return part_cor, row_col