import glob
import json


def main():
    type_statistics_path = '/home/love_you/Documents/Study/Thesis/clothes-detection/docs/statistics_thoi_trang_nam/after_efficientDet_before_combine_statistic/type_statisttics_after_combine.txt'
    with open(type_statistics_path, 'r') as f:
        content = dict(json.load(f))
    print("Number of items: ", len(content.keys()))


if __name__ == '__main__':
    main()