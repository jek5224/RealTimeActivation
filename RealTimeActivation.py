import cv2
import time
import argparse
import torch
import numpy as np
from torchvision.transforms import Normalize

from core.cfgs import parse_args
from models import pymaf_net
from core import path_config, constants
from utils.renderer import PyRenderer_video
from utils.imutils import crop
from multi_person_tracker_yolov8 import MPT8_camera
from utils.demo_utils import convert_crop_cam_to_orig_img
from datasets.data_utils.img_utils import get_single_image_crop_demo

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import imgui
import os
import glfw

MIN_NUM_FRAMES = 1
CHECKPOINT_DIR = 'data/pretrained_model/PyMAF_model_checkpoint.pt'

is_SMPL = False
is_update = True
is_one_person = True
is_video = False

# For PyMAC
from viewer.viewer import GLFWApp
from core.env import Env

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', type=str,
                        help='Path to a single input image')
    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='input image folder')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='output folder to write results')
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn', 'yolov3', 'yolov8'],
                        help='object detector to be used for bbox tracking')
    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')
    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')
    parser.add_argument('--staf_dir', type=str, default='/home/jd/Projects/2D/STAF',
                        help='path to directory STAF pose tracking method.')
    parser.add_argument('--regressor', type=str, default='pymaf_net',
                        help='Name of the SMPL regressor.')
    parser.add_argument('--cfg_file', type=str, default='configs/pymaf_config.yaml',
                        help='config file path.')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to network checkpoint')
    parser.add_argument('--misc', default=None, type=str, nargs="*",
                        help='other parameters')
    parser.add_argument('--model_batch_size', type=int, default=8,
                        help='batch size for SMPL prediction')
    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')
    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')
    parser.add_argument('--with_raw', action='store_true',
                        help='attach raw image.')
    parser.add_argument('--empty_bg', action='store_true',
                        help='render meshes on empty background.')
    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')
    parser.add_argument('--image_based', action='store_true',
                        help='image based reconstruction.')
    parser.add_argument('--use_gt', action='store_true',
                        help='use the ground truth tracking annotations.')
    parser.add_argument('--anno_file', type=str, default='',
                        help='path to tracking annotation file.')
    parser.add_argument('--render_ratio', type=float, default=1.,
                        help='ratio for render resolution')
    parser.add_argument('--recon_result_file', type=str, default='',
                        help='path to reconstruction result file.')
    parser.add_argument('--pre_load_imgs', action='store_true',
                        help='pred-load input images.')
    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    # PyMAC
    parser.add_argument('--checkpoint_muscle', type=str, default=None, help='Checkpoint_path')
    parser.add_argument('--env_path', type=str, default='data/env.xml', help='Env_path')
    parser.add_argument('--no_vqvae_plot', action='store_true', help='No VQ-VAE plot')

    args = parser.parse_args()
    return parse_args(args)


def process_image(img, input_res=224):
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    # Assume that the person is centerered in the image
    height = img.shape[0]
    width = img.shape[1]
    center = np.array([width // 2, height // 2])
    scale = max(height, width) / 200

    # (480, 640, 3), [320, 240], 3.2, (224, 224)
    img_np = crop(img, center, scale, (input_res,input_res))
    # img_np = img
    img = img_np.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]

    # Cropped image, divided by 255, normalized image
    return img_np, norm_img


def process_video(vid):
    vidcap = cv2.VideoCapture(vid)

    img_list = []
    success, image = vidcap.read()
    count = 0
    while success:
        # cv2.imwrite("/path_output_frame/%06d.jpg" % count, image)     # save frame as JPEG file
        img_list.append(image)
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        # count += 1

    return img_list

def SMPL_inference(img_shape, pos_before, app):
    global is_one_person, is_update, is_SMPL
    bbox_scale = 1.0
    crop_size = 224

    tracking_results = mot(img_shape)
    key_list = list(tracking_results.keys())

    orig_width, orig_height = img_shape.shape[:2]

    cand_joint_list = []
    cand_vert_list = []
    cand_cam_trans_list = []

    if is_one_person:
        if len(key_list) == 0:
            draw_key = -1
        elif len(key_list) == 1:
            draw_key = key_list[0]
        elif len(key_list) >= 1:
            draw_key = key_list[0]
            bbox = tracking_results[draw_key]['bbox'][0]
            area = bbox[2] * bbox[3]
            for k_ in key_list:
                cand_bbox = tracking_results[k_]['bbox'][0]
                cand_area = cand_bbox[2] * cand_bbox[3]

                if cand_area > area:
                    area = cand_area
                    draw_key = k_
                elif cand_area == area:
                    if np.linalg.norm(np.array(cand_bbox[0], cand_bbox[1]) - pos_before) < np.linalg.norm(np.array(bbox[0], bbox[1]) - pos_before):
                        area = cand_area
                        draw_key = k_
            pos_before = np.array(tracking_results[draw_key]['bbox'][2:])

        # mid1 = time.time()
        # fps = 1 / (mid1 - pred_time)
        # print(f'YOLO FPS: {fps:.2f}')

        if draw_key != -1:
            bboxes = tracking_results[draw_key]['bbox']
            bbox = bboxes[0]
            img_RGB = cv2.cvtColor(img_shape, cv2.COLOR_BGR2RGB)

            norm_img, _, _ = get_single_image_crop_demo(
                img_RGB,
                bbox,
                kp_2d=None,
                scale=bbox_scale,
                crop_size=crop_size)
            
            norm_img = norm_img.reshape((1, 3, crop_size, crop_size))
            
            with torch.no_grad():
                preds_dict, _ = model(norm_img.to(device))

                output = preds_dict['smpl_out'][-1]

                pred_cam = torch.cat([output['theta'][:, :3].reshape(1, -1)], dim=0)
                pred_verts = torch.cat([output['verts'].reshape(1, -1, 3)], dim=0)
                # pred_joints3d = torch.cat([output['kp_3d'].reshape(1, -1, 3)], dim=0)

                smpl_kp_3d = torch.cat([output['smpl_kp_3d'].reshape(1, -1, 3)], dim=0)

            # mid2 = time.time()
            # fps = 1 / (mid2 - mid1)
            # print(f'PyMAF FPS: {fps:.2f}')
            # mid1 = mid2

            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            # pred_joints3d = pred_joints3d.cpu().numpy() # 49 = 25 OPENPOSE + 24 SPIN

            pred_smpl_kp_3d = smpl_kp_3d.cpu().numpy() # 24 SPIN joints

            orig_cam = convert_crop_cam_to_orig_img(
                    cam=pred_cam,
                    bbox=bboxes,
                    img_width=orig_height,
                    img_height=orig_width,
                )
            
            if is_SMPL:
                img_shape = renderer(
                    verts=pred_verts[0],
                    img=img_shape,
                    cam=orig_cam[0],
                    color_type='white',
                    # kp_3d=pred_joints3d[0],
                    kp_3d=pred_smpl_kp_3d[0],
                    # random=True
                )
                # img_shape = renderer.draw_smpl_kp_3d(
                #     pred_joints3d[0],
                #     img=img_shape,
                #     cam=orig_cam[0],
                #     color_type='white',
                # )

            # Draw bbox
            cv2.rectangle(img_shape, (int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)), (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)), (0, 0, 255), 2)

            # mid2 = time.time()
            # fps = 1 / (mid2 - mid1)
            # print(f'Rendering FPS: {fps:.2f}')
            # mid1 = mid2

            glfw.make_context_current(app.window)

            pelvis_point = pred_smpl_kp_3d[0][0].copy()

            cand_joint = pred_smpl_kp_3d[0]# - pelvis_point
            cand_vert = np.array(cand_joint[3] - cand_joint[0])
            cand_vert /= np.linalg.norm(cand_vert)

            cam = orig_cam[0]
            _, sy, tx, ty = cam
            t = np.array([- tx, ty, 2 * 5000 / (orig_height * sy + 1e-9) / 10 - 1])

            cand_joint_list.append(cand_joint)
            cand_vert_list.append(cand_vert)
            cand_cam_trans_list.append(t)

            if app.smpl_cam_trans is not None and np.linalg.norm(t[2] - app.smpl_cam_trans[0][2]) <= 10:
                app.smpl_joint = cand_joint_list
                app.smpl_vert = cand_vert_list
                app.smpl_cam_trans = cand_cam_trans_list

            elif is_update:
                app.smpl_joint = cand_joint_list
                app.smpl_vert = cand_vert_list
                app.smpl_cam_trans = cand_cam_trans_list
                is_update = False

    else:
        for k in key_list:
            bboxes = tracking_results[k]['bbox']
            bbox = bboxes[0]
            img_RGB = cv2.cvtColor(img_shape, cv2.COLOR_BGR2RGB)

            norm_img, _, _ = get_single_image_crop_demo(
                img_RGB,
                bbox,
                kp_2d=None,
                scale=bbox_scale,
                crop_size=crop_size)
            
            norm_img = norm_img.reshape((1, 3, crop_size, crop_size))
            
            with torch.no_grad():
                preds_dict, _ = model(norm_img.to(device))

                output = preds_dict['smpl_out'][-1]

                pred_cam = torch.cat([output['theta'][:, :3].reshape(1, -1)], dim=0)
                pred_verts = torch.cat([output['verts'].reshape(1, -1, 3)], dim=0)
                # pred_joints3d = torch.cat([output['kp_3d'].reshape(1, -1, 3)], dim=0)

                smpl_kp_3d = torch.cat([output['smpl_kp_3d'].reshape(1, -1, 3)], dim=0)

            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            # pred_joints3d = pred_joints3d.cpu().numpy() # 49 = 25 OPENPOSE + 24 SPIN

            pred_smpl_kp_3d = smpl_kp_3d.cpu().numpy() # 24 SPIN joints

            orig_cam = convert_crop_cam_to_orig_img(
                    cam=pred_cam,
                    bbox=bboxes,
                    img_width=orig_height,
                    img_height=orig_width,
                )
            
            if is_SMPL:
                img_shape = renderer(
                    verts=pred_verts[0],
                    img=img_shape,
                    cam=orig_cam[0],
                    color_type='white',
                    # kp_3d=pred_joints3d[0],
                    kp_3d=pred_smpl_kp_3d[0],
                    # random=True
                )

            pelvis_point = pred_smpl_kp_3d[0][0].copy()

            cand_joint = pred_smpl_kp_3d[0]# - pelvis_point
            cand_vert = np.array(cand_joint[3] - cand_joint[0])
            cand_vert /= np.linalg.norm(cand_vert)

            cam = orig_cam[0]
            _, sy, tx, ty = cam
            t = np.array([- tx, ty, 2 * 5000 / (orig_height * sy + 1e-9) / 10 - 1])

            cand_joint_list.append(cand_joint)
            cand_vert_list.append(cand_vert)
            cand_cam_trans_list.append(t)

        app.smpl_joint = cand_joint_list
        app.smpl_vert = cand_vert_list
        app.smpl_cam_trans = cand_cam_trans_list

    if is_SMPL:
        glfw.make_context_current(app.window)

    return img_shape, pos_before

if __name__ == '__main__':
    cfg = init()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    vid_list = None
    if cfg.VID_FILE is not None:
        vid_list = process_video(cfg.VID_FILE)
        vid_len = len(vid_list)
        vid_count = 0
    
    # ========= PyMAF ========= #

    # ========= Define model ========= #

    mot = MPT8_camera()

    model = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(device)

    if is_video:
        vid_shape = vid_list[0].shape
        orig_height = vid_shape[0]
        orig_width = vid_shape[1]
    else:
        orig_height = 640
        orig_width = 480
    renderer = PyRenderer_video(resolution=(orig_width, orig_height))

    z_offset = 7

    # ========= Load pretrained weights ========= #

    checkpoint = torch.load(CHECKPOINT_DIR)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("[Error] : Could not open webcam.")
        exit(0)
    status, frame = webcam.read()
    
    app = GLFWApp()
    app.draw_vqvae_plot = not cfg.NO_VQVAE_PLOT
    if cfg.CHECKPOINT_MUSCLE:
        app.loadNetwork(cfg.CHECKPOINT_MUSCLE)
    else:
        env_str = None
        with open(cfg.ENV_PATH, "r") as file:
            env_str = file.read()
        app.setEnv(Env(env_str))

    monitor = glfw.get_primary_monitor()
    pos = glfw.get_monitor_pos(monitor)
    size = frame.shape[:2]
    mode = glfw.get_video_mode(monitor)
    
    cv2.namedWindow('webcam inference')   # create a named window
    cv2.moveWindow('webcam inference', 0, 0)
    cv2.namedWindow('video inference')
    cv2.moveWindow('video inference', 0, size[1])

    plt.title("CodeBook") 
    pos_before = np.zeros(2)
    vid_pos_before = np.zeros(2)
    while webcam.isOpened() and not glfw.window_should_close(app.window):
        status, frame = webcam.read()
        if status:
            pred_time = time.time()

            img_shape = frame
            if not is_video:
                img_shape, pos_before = SMPL_inference(img_shape, pos_before, app)

            # img_shape = np.concatenate((img_shape, frame), axis=1)   

            cv2.imshow("webcam inference", img_shape)

            vid_shape = vid_list[vid_count].copy()
            vid_count += 1
            if vid_count == vid_len:
                vid_count = 0

            if is_video:
                vid_shape, vid_pos_before = SMPL_inference(vid_shape, vid_pos_before, app)

            cv2.imshow('video inference', vid_shape)

            fps = 1 / (time.time() - pred_time)
            print(f'Total FPS: {fps:.2f}\n')
                

        if cv2.waitKey(1) == ord('q'):
            print('Quit')
            break
        elif cv2.waitKey(1) == 0x1B:
            print('Quit')
            break
        elif cv2.waitKey(1) == ord('s'):
            is_SMPL = not is_SMPL
        elif cv2.waitKey(1) == ord('u'):
            is_update = not is_update
        elif cv2.waitKey(1) == ord('v'):
            is_video = not is_video
            if is_video:
                vid_shape = vid_list[0].shape
                orig_height = vid_shape[0]
                orig_width = vid_shape[1]
            else:
                orig_height = 640
                orig_width = 480

            renderer = PyRenderer_video(resolution=(orig_width, orig_height))
            app.smpl_joint = None
            app.smpl_vert = None
            app.smpl_cam_trans = None
            glfw.make_context_current(app.window)

            is_update = True
        
        app.impl.process_inputs()
        glfw.poll_events()

        if app.is_simulation:
            app.update()

        # OpenGL.error.Error: Attempt to retrieve context when no valid context
        app.drawSimFrame()
        app.drawUIFrame()            
        app.impl.render(imgui.get_draw_data())

        glfw.swap_buffers(app.window)

        # app.smpl_joint = None

    app.impl.shutdown()

    glfw.terminate()

    webcam.release()
    cv2.destroyAllWindows()