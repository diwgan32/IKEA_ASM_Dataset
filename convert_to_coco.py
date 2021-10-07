import argparse
import cv2
import numpy as np
import json
from toolbox import tb_file_utils as util

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_path', type=str, default='/home/ubuntu/ProcessedDatasets/ikea/ANU_ikea_dataset_poses',
                    help='path to the ANU IKEA assembly video dataset')
parser.add_argument('--devices', nargs='+',  default=['dev1', 'dev2', 'dev3'],
                    help='dev1 | dev2 | dev3 list of device to export')
args = parser.parse_args()


if __name__ == "__main__":
	all_files = util.get_list_of_all_files(args.annotation_path)

	for file in all_files:
		print(file)
		input("? ")

		