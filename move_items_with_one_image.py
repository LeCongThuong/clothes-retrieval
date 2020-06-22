import glob
import random
import shutil
import os
import json


def take_item_codes(type_statistics_file):
    with open(type_statistics_file, 'r') as f:
        content = dict(json.load(f))

    item_codes = []
    # keys = list(content.keys())  # Python 3; use keys = d.keys() in Python 2
    # random.shuffle(keys)
    # num_items = 0
    for key, value in content.items():
        if int(content[key]) == 1:
            item_codes.append(key)
            # num_items += 1
    return item_codes


def move_rename(src_file_path: str, output_dir) -> str:
    des_file_name = src_file_path.split('/')[-1]
    des_file_path = os.path.join(output_dir, des_file_name)
    shutil.move(src_file_path, des_file_path)


def main():
    type_statistics_file = '/home/love_you/statistics/type_statisttics_after_combine.txt'
    images_dir = '/home/love_you/object-detection-output/object-detection-result'
    out_dir = '/home/love_you/object-detection-output/one_item_with_one_image_object-detection-result'
    one_item_one_image_paths_logs = '/home/love_you/statistics/item_with_one_images.txt'
    total_items = 0
    item_with_one_image_list = []
    images_path_list = glob.glob(images_dir + '/*')
    item_codes = take_item_codes(type_statistics_file)
    for image_path in images_path_list:
        for image_code in item_codes:
            if image_code in image_path:
                move_rename(image_path, out_dir)
                total_items += 1
                item_with_one_image_list.append(image_path.split('/')[-1])
                break

    print("Total items: ", total_items)
    with open(one_item_one_image_paths_logs, 'a+') as f:
        f.write('\n'.join(item_with_one_image_list))


if __name__ == '__main__':
    main()
