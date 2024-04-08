import cv2
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader

from core.cfgs import cfg, parse_args
from models import pymaf_net, SMPL
from core import path_config, constants
from utils.renderer import PyRenderer_video
from utils.imutils import crop
from datasets.inference import Inference_camera
from multi_person_tracker_yolov8 import MPT8_camera
from utils.demo_utils import convert_crop_cam_to_orig_img, prepare_rendering_results
from datasets.data_utils.img_utils import get_single_image_crop_demo

MIN_NUM_FRAMES = 1
CHECKPOINT_DIR = 'data/pretrained_model/PyMAF_model_checkpoint.pt'

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

    args = parser.parse_args()
    parse_args(args)

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


image = cv2.imread('examples/COCO_val2014_000000019667.jpg', cv2.IMREAD_COLOR)
img_height, img_width= image.shape[:-1]
video = True

#input_res = 480
if __name__ == '__main__':
    init()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # ========= Define model ========= #

    bbox_scale = 1.0
    mot = MPT8_camera()

    model = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(device)
    orig_height = 640
    orig_width = 480
    crop_size = 224
    renderer = PyRenderer_video(resolution=(orig_width, orig_height))
    # renderer = PyRenderer_video(resolution=(img_width, img_height))
    # renderer = OpenDRenderer(resolution=(orig_width, orig_height))

    # ========= Load pretrained weights ========= #

    checkpoint = torch.load(CHECKPOINT_DIR)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("[Error] : Could not open webcam.")
        exit(0)
    
    while webcam.isOpened():
        status, frame = webcam.read()

        if status:
            if video:
                pred_time = time.time()

                img_shape = frame
                
                tracking_results = mot(img_shape)

                mid1 = time.time()
                fps = 1 / (mid1 - pred_time)
                print(f'YOLO FPS: {fps:.2f}')

                key_list = list(tracking_results.keys())
                for person_id in key_list:
                    bboxes = tracking_results[person_id]['bbox']
                    bbox = bboxes[0]
                    frames = tracking_results[person_id]['frames']
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

                    mid2 = time.time()
                    fps = 1 / (mid2 - mid1)
                    print(f'PyMAF FPS: {fps:.2f}')
                    mid1 = mid2

                    # ========= Save results to a pickle file ========= #
                    pred_cam = pred_cam.cpu().numpy()
                    pred_verts = pred_verts.cpu().numpy()
                    # pred_joints3d = pred_joints3d.cpu().numpy()

                    orig_cam = convert_crop_cam_to_orig_img(
                        cam=pred_cam,
                        bbox=bboxes,
                        img_width=orig_height,
                        img_height=orig_width,
                        # img_width=img_width,
                        # img_height=img_height,
                    )

                    img_shape = renderer(
                        pred_verts[0],
                        img=img_shape,
                        cam=orig_cam[0],
                        color_type='purple',
                    )

                    mid2 = time.time()
                    fps = 1 / (mid2 - mid1)
                    print(f'Rendering FPS: {fps:.2f}')
                    mid1 = mid2

                    # Draw one person
                    break

                fps = 1 / (time.time() - pred_time)
                print(f'Total FPS: {fps:.2f}\n')

                cv2.imshow("webcam inference", img_shape)
            
            else:
                pred_time = time.time()

                img_np, norm_img = process_image(frame, input_res=224)

                with torch.no_grad():
                    preds_dict, _ = model(norm_img.to(device))
                    output = preds_dict['smpl_out'][-1]
                    pred_camera = output['theta'][:, :3]
                    pred_vertices = output['verts']

                img_shape = renderer(
                    pred_vertices[0].cpu().numpy(),
                    img=img_np,
                    cam=pred_camera[0].cpu().numpy(),
                    color_type='purple',
                )

                img_shape = cv2.resize(img_shape, dsize=(700, 700), interpolation=cv2.INTER_CUBIC)
                end = time.time()
                fps = 1 / (end - pred_time)

                print(f'FPS: {fps:.2f}')

                cv2.imshow("webcam inference", img_shape)

        if cv2.waitKey(1) == ord('q'):
            print('Quit')
            break
        if cv2.waitKey(1) == ord('v'):
            video = not video
        

    webcam.release()
    cv2.destroyAllWindows()
