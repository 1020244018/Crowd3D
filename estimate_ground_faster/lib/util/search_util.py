
from typing import Tuple, List
import numpy as np

def get_list_fov(range_fov:List[float], num_interval:int)->(List[float]):
    list_fov = []
    len_interval = (range_fov[1] - range_fov[0]) / num_interval
    for i in range(num_interval + 1):
        fov = range_fov[0] + len_interval * i
        list_fov.append(fov)
    return list_fov

def get_next_range_with_unstable_allowance(list_loss:np.ndarray)->(Tuple[int, int]):
    
    losses = np.array(list_loss)
    index_min_loss = np.argmin(losses)

    if index_min_loss <= 2:
        index_left = 0
    elif losses[index_min_loss - 2] > losses[index_min_loss - 1]:
        index_left = index_min_loss - 2
    elif index_min_loss <= 3:
        index_left = 0
    elif losses[index_min_loss - 3] > losses[index_min_loss - 2]:
        index_left = index_min_loss - 3
    else:
        #raise(Exception(index_min_loss, losses))
        print('Warning, the loss is not convex in this FOV range', losses, index_min_loss)
        return None

    if index_min_loss >= len(losses) - 1 - 2:
        index_right = len(losses) - 1
    elif losses[index_min_loss + 2] > losses[index_min_loss + 1]:
        index_right = index_min_loss + 2
    elif index_min_loss >= len(losses) - 1 - 3:
        index_right = len(losses) - 1
    elif losses[index_min_loss + 3] > losses[index_min_loss + 2]:
        index_right = index_min_loss + 3
    else:
        print('Warning, the loss is not convex in this FOV range', losses, index_min_loss)
        return None
    return [index_left, index_right]