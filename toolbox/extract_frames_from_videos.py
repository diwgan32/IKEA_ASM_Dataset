import argparse
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset_processed',
                    help='path to the ANU IKEA assembly video dataset')
parser.add_argument('--output_path', type=str, default='/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset_processed_frames',
                    help='path to output location of the frames extracted from the video dataset')
parser.add_argument('--devices', nargs='+',  default=['dev1', 'dev2', 'dev3'],
                    help='dev1 | dev2 | dev3 list of device to export')
args = parser.parse_args()

def get_scan_list(input_path, devices='all'):
    '''
    get_scan_list retreieves all of the subdirectories under the dataset main directories:
    :param input_path: path to ANU IKEA Dataset directory

    :return:
    scan_path_list: path to all available scans
    category_path_list: path to all available categories
    '''

    category_path_list = get_subdirs(input_path)
    scan_path_list = []
    for category in category_path_list:
        if os.path.basename(category) != 'Calibration':
            category_scans = get_subdirs(category)
            for category_scan in category_scans:
                scan_path_list.append(category_scan)

    rgb_path_list = []
    depth_path_list = []
    normals_path_list = []
    rgb_params_files = []
    depth_params_files = []
    for scan in scan_path_list:
        device_list = get_subdirs(scan)
        for device in device_list:
            if os.path.basename(device) in devices:
                rgb_path = os.path.join(device, 'images')
                depth_path = os.path.join(device, 'depth')
                normals_path = os.path.join(device, 'normals')
                if os.path.exists(rgb_path):
                    rgb_path_list.append(rgb_path)
                    rgb_params_files.append(os.path.join(device, 'ColorIns.txt'))
                if os.path.exists(depth_path):
                    if 'dev3' in device:  # remove redundant depths - remove this line for full 3 views
                        depth_path_list.append(depth_path)
                        depth_params_files.append(os.path.join(device, 'DepthIns.txt'))
                if os.path.exists(normals_path):
                    normals_path_list.append(normals_path)

    return category_path_list, scan_path_list, rgb_path_list, depth_path_list, depth_params_files, rgb_params_files, normals_path_list

def extract_frames(scan, dataset_path, out_path, fps=25, video_format='avi'):
    """
    Extract individual frames from a scan video
    Parameters
    ----------
    scan : path to a single video scan (.avi file)
    out_path : output path
    fps : frames per second

    Returns
    -------

    """
    out_dir = scan.replace(dataset_path, out_path)
    os.makedirs(out_dir, exist_ok=True)
    scan_video_filename = os.path.join(scan, 'scan_video.' + video_format)

    print('Extracting frames from: ' + scan_video_filename)

    cap = cv2.VideoCapture(scan_video_filename)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if 'depth' not in scan:
            cv2.imwrite(os.path.join(out_dir,  str(i).zfill(6) + '.jpg'), frame)
        else:
            frame = get_absolute_depth(frame)
            cv2.imwrite(os.path.join(out_dir, str(i).zfill(6) + '.png'), frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    print('Done')

category_path_list, scan_list, rgb_path_list, depth_path_list, depth_params_files, \
rgb_params_files, normals_path_list = get_scan_list(args.dataset_path, devices=args.devices)

print('Video dataset path: ' + args.dataset_path)
print('Individual frames dataset will be saved to ' + args.output_path)


os.makedirs(args.output_path, exist_ok=True)
for scan_list in [rgb_path_list, depth_path_list, normals_path_list]:
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(utils.extract_frames(scan, args.dataset_path,
                                                            args.output_path) for scan, _ in enumerate(scan_list)))
    # # Non-parallel implementation
    # for scan in scan_list:
    #     utils.extract_frames(scan, args.dataset_path, args.output_path)

