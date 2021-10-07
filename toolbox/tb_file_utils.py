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

def get_files(input_path, file_type='.png'):
    '''
    get a list of files in input_path directory
    :param input_path: parent directory (in which to get the file list)
    :return:
    files: list of files in input_path
    '''
    files = [os.path.join(input_path, file_i) for file_i in os.listdir(input_path)
               if os.path.isfile(os.path.join(input_path, file_i)) and file_i.endswith(file_type)]
    files.sort()
    return files
    
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


def get_gt_dirs(input_path, camera_id='dev3'):
    """Get all directories with ground-truth 2D human pose annotations
    """
    gt_path_list = []
    category_path_list = get_subdirs(input_path)
    for category in category_path_list:
        if os.path.basename(category) != 'Calibration':
            category_scans = get_subdirs(category)
            for category_scan in category_scans:
                device_list = get_subdirs(category_scan)
                for device_path in device_list:
                    if camera_id in device_path:
                        if os.path.exists(os.path.join(device_path, 'pose3d')): # 2D annotations exist
                            gt_path_list.append(device_path) # eg <root>/Lack_TV_Bench/0007_white_floor_08_04_2019_08_28_10_47/dev3
    return gt_path_list

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


def get_rgb_ins_params(param_file):
    '''
    read the rgb intrinsic parameters file
    :param param_file: path to depth intrinsic parameters file DepthIns.txt
    :return:
    rgb_ins_params: a libfreenect2 ColorCameraParams object
    '''
    with open(param_file, 'r') as f:
        rgb_ins_params = [float(line.strip()) for line in f if line]

    rgb_camera_params_obj = {
        "fx" : rgb_ins_params[0],
        "fy" : rgb_ins_params[1],
        "cx" : rgb_ins_params[2],
        "cy" : rgb_ins_params[3],

        "shift_d" : rgb_ins_params[4],
        "shift_m" : rgb_ins_params[5],
        "mx_x3y0" : rgb_ins_params[6],
        "mx_x0y3" : rgb_ins_params[7],
        "mx_x2y1" : rgb_ins_params[8],
        "mx_x1y2" : rgb_ins_params[9],
        "mx_x2y0" : rgb_ins_params[10],
        "mx_x0y2" : rgb_ins_params[11],
        "mx_x1y1" : rgb_ins_params[12],
        "mx_x1y0" : rgb_ins_params[13],
        "mx_x0y1" : rgb_ins_params[14],
        "mx_x0y0" : rgb_ins_params[15],

        "my_x3y0" : rgb_ins_params[16],
        "my_x0y3" : rgb_ins_params[17],
        "my_x2y1" : rgb_ins_params[18],
        "my_x1y2" : rgb_ins_params[19],
        "my_x2y0" : rgb_ins_params[20],
        "my_x0y2" : rgb_ins_params[21],
        "my_x1y1" : rgb_ins_params[22],
        "my_x1y0" : rgb_ins_params[23],
        "my_x0y1" : rgb_ins_params[24],
        "my_x0y0" : rgb_ins_params[25]
    }
    return rgb_camera_params_obj

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
