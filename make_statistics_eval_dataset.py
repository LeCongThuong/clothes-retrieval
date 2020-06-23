import glob
import json


def main():
    eval_dir = "/home/love_you/object-detection-output/eval_data"
    eval_image_path_list = glob.glob(eval_dir + "/*")
    eval_type_statistics_dict = {}
    eval_type_statistics_file = '/home/love_you/statistics/statistics_eval_dataset.txt'

    for eval_image_path in eval_image_path_list:
        eval_image_name = eval_image_path.split('/')[-1]
        eval_image_name_components = eval_image_name.split('~')
        eval_image_cate = '~'.join(eval_image_name_components[2:4])

        if eval_image_cate in eval_type_statistics_dict:
            eval_type_statistics_dict[eval_image_cate] += 1
        else:
            eval_type_statistics_dict[eval_image_cate] = 1

    with open(eval_type_statistics_file, 'w') as f:
        json.dump(eval_type_statistics_dict, f)


if __name__ == '__main__':
    main()