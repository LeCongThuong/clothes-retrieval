import glob


def main():
    images_dir = '/home/love_you/object-detection-output/object-detection-result'
    output_file = 'data_train_list_file.txt'
    prepend_str = '/content/clothes-retrieval/data_train'

    prepend_image_list = []
    image_paths_list = glob.glob(images_dir, '/*')
    for image_path in image_paths_list:
        image_name = image_path.split('/')[-1]
        prepend_image_name = prepend_str + '/' + image_name
        prepend_image_list.append(prepend_image_name)

    with open(output_file, 'a+') as f:
        f.write('\n'.join(prepend_image_list))


if __name__ == '__main__':
    main()