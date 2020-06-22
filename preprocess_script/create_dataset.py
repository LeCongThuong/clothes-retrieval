import glob
import json
import os
from skimage import io

def main():
  #  root_dir = "/home/love_you/Documents/Study/Thesis/clothes-image-retrieval/tmp/"
    images_dir = "/home/love_you/Documents/Study/Thesis/clothes-image-retrieval/tmp/images"
    annos_dir = "/home/love_you/Documents/Study/Thesis/clothes-image-retrieval/tmp/annos"
    dataset_dir = "/home/love_you/Documents/Study/Thesis/clothes-image-retrieval/tmp/dataset"
    error_file = '/home/love_you/Documents/Study/Thesis/clothes-image-retrieval/tmp/error.txt'

    image_list = sorted(glob.glob(images_dir + '/*'))
    annos_dir = sorted(glob.glob(annos_dir + '/*'))
    print(image_list)
    print(annos_dir)
    if len(image_list) != len(annos_dir):
        raise Exception("Number of images is not same as number of annos")
    num_count = 0
    style_zero = 0
    total_count = 0
    for index, annos_path in enumerate(annos_dir):
        try:

            image = io.imread(image_list[index])
        except Exception as ex:
            print("Reading image has problem" + annos_path.split('/')[-1])
            with open(error_file, 'a+') as err:
                err.write("Reading:" + annos_path.split('/')[-1] + "\n")
            continue

        with open(annos_path, 'r') as f:
            annos_content = json.load(f)
        pair_id = annos_content["pair_id"]
        item_list = [value for key, value in annos_content.items() if key.startswith("item")]
        for item in item_list:
            total_count += 1
            style_id = item["style"]

            if style_id != 0:
                category_id = item["category_id"]
                folder_name = str(category_id) + "_" + str(pair_id) + "_" + str(style_id)
                folder_path = os.path.join(dataset_dir, folder_name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                x_min, y_min, x_max, y_max = item["bounding_box"]
                try:
                    crop_image = image[y_min:y_max + 1, x_min: x_max + 1]
                except Exception as ex:
                    print("Cropping has problem" + annos_path.split('/')[-1])
                    with open(error_file, 'a+') as err:
                        err.write("Cropping:" + annos_path.split('/')[-1] + "\n")
                    continue
                crop_image_name = str(count) + ".jpg"
                count = count + 1
                image_path = str(os.path.join(folder_path, crop_image_name))
                try:
                    io.imsave(image_path, crop_image)
                except Exception as ex:
                    print("Saving has problem" + annos_path.split('/')[-1])
                    with open(error_file, 'a+') as err:
                        err.write("Saving:" + annos_path.split('/')[-1] + "\n")
                    continue
                print("\r", count, end='')
            else:
                style_zero += 1
                print(annos_path.split("/")[-1])

    print("zero style: ", style_zero)
    print("total count: ", total_count)


if __name__ =='__main__':
    main()