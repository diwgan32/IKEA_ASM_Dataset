import argparse
import cv2
import numpy as np
import json
import os
import random

from toolbox import tb_file_utils as util

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_path', type=str, default='/home/ubuntu/ProcessedDatasets/ikea/ANU_ikea_dataset_poses',
                    help='path to the ANU IKEA assembly video dataset')
parser.add_argument('--frames_path', type=str, default='/home/ubuntu/ProcessedDatasets/ikea/ANU_ikea_dataset_frames',
                            help='path to the ANU IKEA assembly video dataset')
parser.add_argument('--videos_path', type=str, default='/home/ubuntu/ProcessedDatasets/ikea/ANU_ikea_dataset_video',
                                    help='path to the ANU IKEA assembly video dataset')
parser.add_argument('--devices', nargs='+',  default=['dev1', 'dev2', 'dev3'],
                    help='dev1 | dev2 | dev3 list of device to export')
args = parser.parse_args()

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 2

    return proj_pts

def vis_keypoints(frame, joints2d):
    for i in range(joints2d.shape[0]):
        if (np.isnan(joints2d[i][0]) or np.isnan(joints2d[i][1])):
            continue
        frame = cv2.circle(frame, (int(joints2d[i][0]), int(joints2d[i][1])), 5, (0, 0, 0), 2)

    return frame

def process_frame(joints_3d, camera_params):
    joints_np = np.array(joints_3d["pose_keypoints_3d"]).reshape((17, 4))
    K = np.array([
        [camera_params["fx"], 0, camera_params["cx"]],
        [0, camera_params["fy"], camera_params["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)
    joints2d = project_3D_points(K, joints_np[:, 0:3])
    
    return joints_np, joints2d, K

def get_bbox(uv, frame_shape):
    x = min(uv[:, 0]) - 10
    y = min(uv[:, 1]) - 10

    x_max = min(max(uv[:, 0]) + 10, frame_shape[1])
    y_max = min(max(uv[:, 1]) + 10, frame_shape[0])

    return [
        float(max(0, x)), float(max(0, y)), float(x_max - x), float(y_max - y)
    ]

if __name__ == "__main__":
    gt_dirs = util.get_gt_dirs(args.annotation_path, "dev2") # TODO change camera id?
    
    output = {
        "images": [],
        "annotations": [],
        "categories": [{
            'supercategory': 'person',
            'id': 1,
            'name': 'person'
        }]
    }
    idx = 0
    for gt_dir in gt_dirs:
        prediction_dir = os.path.join(gt_dir, 'pose3d')
        cam = gt_dir.split('/')[-2]
        label = gt_dir.split('/')[-3]
        scene = label.split('_')[3]
        assembly = gt_dir.split('/')[-4]
        files = util.get_files(gt_dir, file_type=".json")
        print(gt_dir)

        for f in files:
            basename = os.path.basename(f)
            image_path = os.path.join(args.frames_path, assembly, label, cam,  "images", basename.split(".")[0]+".jpg")
            camera_calib_path = os.path.join(args.videos_path, assembly, label, cam, "ColorIns.txt")
            camera_params = util.get_rgb_ins_params(camera_calib_path)
            frame = cv2.imread(image_path)
            fd = open(f, "r")
            joint_3d = json.load(fd)
            fd.close()

            joints_np, joints_2d, K = process_frame(joint_3d, camera_params)
            output["images"].append({
                "id": idx,
                "width": frame.shape[1],
                "height": frame.shape[0],
                "file_name": os.path.join(assembly, label, cam,  "images", basename.split(".")[0]+".jpg"),
                "camera_param": {
                    "focal": [float(K[0][0]), float(K[1][1])],
                    "princpt": [float(K[0][2]), float(K[1][2])]
                }
            })

            output["annotations"].append({
                "id": idx,
                "image_id": idx,
                "category_id": 1,
                "is_crowd": 0,
                "joint_cam": joints_np.tolist(),
                "bbox": get_bbox(joints_2d, frame.shape) # x, y, w, h
            })

    assert(len(output["images"]) == len(output["annotations"]))
    print(f"Len: {len(output['images'])}")
    f = open("ikea_training.json", "w")
    json.dump(f, output)
    f.close()


