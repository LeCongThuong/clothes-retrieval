import glob
import json


def main():
    images_dir = '/home/love_you/object-detection-output/object-detection-result'
    output_file = '/home/love_you/statistics/label_to_samples.txt'
    prepend_str = '/content/clothes-retrieval/data_train'
    label_to_samples = {}

    # list all files in image dir
    image_paths_list = glob.glob(images_dir + '/*')
    for image_path in image_paths_list:
        print('\r ', image_path, end='')
        image_name = image_path.split('/')[-1]
        name_path_component = image_name.split('~')
        label = name_path_component[-4]
        new_image_path = prepend_str + '/' + image_name
        if label in label_to_samples:
            label_to_samples[label].append(new_image_path)
        else:
            label_to_samples[label] = [new_image_path]

    with open(output_file, 'w') as f:
        json.dump(label_to_samples, f)


if __name__ == '__main__':
    main()




