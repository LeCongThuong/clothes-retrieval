import json


def main():
    type_statistics_file = '/home/love_you/statistics/type_statisttics_after_combine.txt'
    label_logs = '/home/love_you/statistics/labels_without_one_image_items.txt'
    total_items = 0
    labels_list = []

    with open(type_statistics_file, 'r') as f:
        content = dict(json.load(f))

    for key, value in content.items():
        if int(content[key]) > 1:
            labels_list.append(key)
            total_items += 1

    print("Total label: ", total_items)
    with open(label_logs, 'a+') as f:
        f.write('\n'.join(labels_list))


if __name__ == '__main__':
    main()