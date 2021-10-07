import argparse
import os
import cv2
import tb_file_utils as utils
from multiprocessing import Process
import random

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def process(scan_list, machine_num, args):
    i = 0
    for scan in scan_list:
        utils.extract_frames(scan, args.dataset_path, args.output_path)
        print(f"Completed {i} of {len(scan_list)} on machine {machine_num}")
        i += 1

NUM_CPUS = 32
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset_processed',
                    help='path to the ANU IKEA assembly video dataset')
parser.add_argument('--output_path', type=str, default='/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset_processed_frames',
                    help='path to output location of the frames extracted from the video dataset')
parser.add_argument('--devices', nargs='+',  default=['dev1', 'dev2', 'dev3'],
                    help='dev1 | dev2 | dev3 list of device to export')
args = parser.parse_args()


category_path_list, scan_list, rgb_path_list, depth_path_list, depth_params_files, \
rgb_params_files, normals_path_list = utils.get_scan_list(args.dataset_path, devices=args.devices)

print('Video dataset path: ' + args.dataset_path)
print('Individual frames dataset will be saved to ' + args.output_path)


os.makedirs(args.output_path, exist_ok=True)
for scan_list in [depth_path_list, normals_path_list]:
    #num_cores = multiprocessing.cpu_count()
    #multiprocessing.Parallel(n_jobs=num_cores)(delayed(utils.extract_frames(scan, args.dataset_path,
    #                                                        args.output_path) for scan, _ in enumerate(scan_list)))
    # # Non-parallel implementation
    partitioned_list = partition(scan_list, NUM_CPUS)
    processes = []
    for i in range(NUM_CPUS):
        processes.append(Process(target=process, args=(partitioned_list[i], i, args)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()

