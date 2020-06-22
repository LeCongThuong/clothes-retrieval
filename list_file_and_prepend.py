import glob


def main():
    images_dir = ''
    output_file = ''
    prepend_str = ''

    prepend_image_list = []
    image_paths_list = glob.glob(images_dir, '/*')
    for image_path in image_paths_list:
        image_name = image_path.split('/')[-1]
        prepend_image_name = prepend_str + '/' + image_name
        prepend_image_list.append(prepend_image_name)

    with open(output_file, 'w') as f:
        f.write('\n'.join(prepend_image_list))