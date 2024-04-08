import cv2
import time
import argparse
import torch
import numpy as np
from torchvision.transforms import Normalize

from core.cfgs import cfg, parse_args
from models import pymaf_net, SMPL
from core import path_config, constants
from utils.renderer import PyRenderer
from utils.imutils import crop

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

    img_np = crop(img, center, scale, (input_res, input_res))
    img = img_np.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img_np, img, norm_img


if __name__ == '__main__':
    init()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ========= Define model ========= #
    model = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(device)

    # ========= Load pretrained weights ========= #

    checkpoint = torch.load(CHECKPOINT_DIR)
    model.load_state_dict(checkpoint['model'], strict=True)

    # Load SMPL model
    smpl = SMPL(path_config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = PyRenderer(resolution=(constants.IMG_RES, constants.IMG_RES))

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("[Error] : Could not open webcam.")
        exit(0)

    while webcam.isOpened():
        status, frame = webcam.read()

        if status:
            # Preprocess input image and generate predictions
            img_np, img, norm_img = process_image(frame, input_res=constants.IMG_RES)
            with torch.no_grad():
                preds_dict, _ = model(norm_img.to(device))
                output = preds_dict['smpl_out'][-1]
                pred_camera = output['theta'][:, :3]
                pred_vertices = output['verts']

            # Calculate camera parameters for rendering
            camera_translation = torch.stack([pred_camera[:, 1], pred_camera[:, 2], 2 * constants.FOCAL_LENGTH / (
                        constants.IMG_RES * pred_camera[:, 0] + 1e-9)], dim=-1)
            camera_translation = camera_translation[0].cpu().numpy()
            pred_vertices = pred_vertices[0].cpu().numpy()

            #img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            # Render front-view shape
            save_mesh_path = None
            img_shape = renderer(
                pred_vertices,
                img=img_np,
                cam=pred_camera[0].cpu().numpy(),
                color_type='purple',
                mesh_filename=save_mesh_path
            )
            # img_shape = cv2.resize(img_shape, dsize=(700, 700), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("test", img_shape)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
