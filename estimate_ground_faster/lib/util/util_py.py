
from typing import List

def generate_info(
    list_label:List[str], 
    list_values:List[List]
    ) -> (List[str]):
    list_info = []
    assert len(list_label) == len(list_values)
    len_items = len(list_values[0])
    for values in list_values :
        assert len(values) == len_items

    for index_items in range(len_items) :
        info = ''
        for index_list in range(len(list_values)) :
            info =  info + \
                    list_label[index_list] + \
                    ':' + \
                    '%.3f' % (list_values[index_list][index_items]) + \
                    '  '
        list_info.append(info)
    return list_info