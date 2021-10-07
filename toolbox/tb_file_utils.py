import argparse
import os
import cv2
import numpy as np

def get_subdirs(input_path):
    '''
    get a list of subdirectories in input_path directory
    :param input_path: parent directory (in which to get the subdirectories)
    :return:
    subdirs: list of subdirectories in input_path
    '''
    subdirs = [os.path.join(input_path, dir_i) for dir_i in os.listdir(input_path)
               if os.path.isdir(os.path.join(input_path, dir_i))]
    subdirs.sort()
    return subdirs

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


def get_absolute_depth(img, min_depth_val=0.0, max_depth_val = 4500, colormap='jet'):
    '''
    Convert the relative depth image to absolute depth. uses fixed minimum and maximum distances
    to avoid flickering
    :param img: depth image
           min_depth_val: minimum depth in mm (default 50cm)
           max_depth_val: maximum depth in mm ( default 10m )
    :return:
    absolute_depth_frame: absolute depth converted into cv2 gray
    '''

    absolute_depth_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # absolute_depth_frame = cv2.convertScaleAbs(absolute_depth_frame, alpha=(max_depth_val/255.0),
    #                                            beta=min_depth_val).astype(np.uint16) #converts to 8 bit
    absolute_depth_frame = absolute_depth_frame * float(max_depth_val/255.0)
    return absolute_depth_frame.astype(np.uint16)

def get_list_of_all_files(dir_path, file_type='.jpg'):
    '''
    get a list of all files of a given type in input_path directory
    :param dir_path: parent directory (in which to get the file list)
    :return:
    allFiles: list of files in input_path
    '''
    listOfFile = os.listdir(dir_path)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dir_path, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_all_files(fullPath, file_type=file_type)
        else:
            if fullPath.endswith(file_type):
                allFiles.append(fullPath)

    return allFiles

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
