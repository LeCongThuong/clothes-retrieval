import glob
import random
import shutil
import os
import json


def take_n_item_codes(type_statistics_file, n=5000):
    with open(type_statistics_file, 'r') as f:
        content = dict(json.load(f))

    item_codes = []
    keys = list(content.keys())  # Python 3; use keys = d.keys() in Python 2
    random.shuffle(keys)
    num_items = 0
    for key in keys:
        if int(content[key]) == 5:
            item_codes.append(key)
            num_items += 1
            if num_items > n:
                break
    return item_codes


def move_rename(src_file_path: str, output_dir) -> str:
    des_file_name = src_file_path.split('/')[-1]
    des_file_path = os.path.join(output_dir, des_file_name)
    shutil.move(src_file_path, des_file_path)


def main():
    type_statistics_file = '/home/love_you/statistics/type_statisttics_after_combine.txt'
    images_dir = '/home/love_you/object-detection-output/object-detection-result'
    out_dir = '/home/love_you/object-detection-output/eval_data'
    eval_file_paths_logs = '/home/love_you/statistics/eval_file_paths_file.txt'
    total_image = 5000
    eval_file_list = []
    images_path_list = glob.glob(images_dir + '/*')
    item_codes = take_n_item_codes(type_statistics_file, 5000)
    for image_path in images_path_list:
        for image_code in item_codes:
            if image_code in image_path:
                move_rename(image_path, out_dir)
                total_image += 1
                eval_file_list.append(image_path.split('/')[-1])
                
                continue
        continue

    with open(eval_file_paths_logs, 'a+') as f:
        f.write('\n'.join(eval_file_list))


if __name__ == '__main__':
    main()
