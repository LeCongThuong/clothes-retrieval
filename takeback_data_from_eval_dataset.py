import glob
import shutil
import os


def move_rename(src_file_path: str, output_dir) -> str:
    des_file_name = src_file_path.split('/')[-1]
    des_file_path = os.path.join(output_dir, des_file_name)
    shutil.move(src_file_path, des_file_path)


def main():
    eval_dir = '/home/love_you/object-detection-output/eval_data'
    extend_train_dir = '/home/love_you/object-detection-output/extend_data'
    eval_image_path_list = glob.glob(eval_dir + '/*')
    label_list = []
    for eval_image_path in eval_image_path_list:
        eval_name = eval_image_path.split('/')[-1]
        label_of_this_eval = eval_name.split('~')[-4]
        if label_of_this_eval in label_list:
            move_rename(eval_image_path, extend_train_dir)
        else:
            label_list.append(label_of_this_eval)


if __name__ == '__main__':
    main()