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
from utils.renderer import PyRenderer
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


# image = cv2.imread('examples/COCO_val2014_000000019667.jpg', cv2.IMREAD_COLOR)
image = cv2.imread('examples/COCO_val2014_000000019667.jpg', cv2.IMREAD_COLOR)

#input_res = 480
input_res = constants.IMG_RES
is_image = False
together = False
threshold = False
original = False
video = True
new_size = 800
vid_ratio = 1
if __name__ == '__main__':
    init()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # ========= Define model ========= #
    if video:
        bbox_scale = 1.0
        mot = MPT8_camera()
    else:
        # Setup renderer for visualization
        renderer = PyRenderer(resolution=(input_res,input_res))

    model = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(device)

    # ========= Load pretrained weights ========= #

    checkpoint = torch.load(CHECKPOINT_DIR)
    model.load_state_dict(checkpoint['model'], strict=True)

    # Load SMPL model
    # smpl = SMPL(path_config.SMPL_MODEL_DIR,
    #             batch_size=1,
    #             create_transl=False).to(device)
    model.eval()

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("[Error] : Could not open webcam.")
        exit(0)

    detector = 'yolo'
    
    while webcam.isOpened():
        status, frame = webcam.read()

        if status:
            # Preprocess input image and generate predictions
            if video:
                pred_time = time.time()

                if is_image:
                    img = image
                else:
                    img = frame

                orig_height, orig_width = img.shape[:2]

                img_shape = img.copy()
                renderer = PyRenderer(resolution=(orig_height, orig_width))

                # vid_res = (orig_width // vid_ratio, orig_height // vid_ratio)
                # img_shape = cv2.resize(img, dsize=vid_res, interpolation=cv2.INTER_CUBIC)
                # renderer = PyRenderer(resolution=vid_res)
                
                tracking_results = mot(img_shape)

                end = time.time()
                fps = 1 / (end - pred_time)
                print(f'YOLO FPS: {fps:.2f}')

                # remove tracklets if num_frames is less than MIN_NUM_FRAMES
                # for person_id in list(tracking_results.keys()):
                #     if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
                #         del tracking_results[person_id]

                # ========= Run pred on each person ========= #
                # print(f'Running reconstruction on each tracklet...')

                # for person_id in list(tracking_results.keys()):
                #     bboxes = tracking_results[person_id]['bbox']
                #     frames = tracking_results[person_id]['frames']

                #     dataset = Inference_camera(
                #         img_shape,
                #         frames=frames,
                #         bboxes=bboxes,
                #         scale=bbox_scale,
                #     )

                #     bboxes = dataset.bboxes
                #     frames = dataset.frames
                #     dataloader = DataLoader(dataset, batch_size=1, num_workers=1)

                #     with torch.no_grad():
                #         pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

                #         batch = next(iter(dataloader)).to(device)
                #         preds_dict, _ = model(batch)

                #         output = preds_dict['smpl_out'][-1]

                #         pred_cam.append(output['theta'][:, :3].reshape(1, -1))
                #         pred_verts.append(output['verts'].reshape(1, -1, 3))
                #         pred_pose.append(output['theta'][:, 13:85].reshape(1, -1))
                #         pred_betas.append(output['theta'][:, 3:13].reshape(1, -1))
                #         pred_joints3d.append(output['kp_3d'].reshape(1, -1, 3))

                #         pred_cam = torch.cat(pred_cam, dim=0)
                #         pred_verts = torch.cat(pred_verts, dim=0)
                #         pred_pose = torch.cat(pred_pose, dim=0)
                #         pred_betas = torch.cat(pred_betas, dim=0)
                #         pred_joints3d = torch.cat(pred_joints3d, dim=0)

                #         del batch

                #     # ========= Save results to a pickle file ========= #
                #     pred_cam = pred_cam.cpu().numpy()
                #     pred_verts = pred_verts.cpu().numpy()
                #     pred_pose = pred_pose.cpu().numpy()
                #     pred_betas = pred_betas.cpu().numpy()
                #     pred_joints3d = pred_joints3d.cpu().numpy()

                #     orig_cam = convert_crop_cam_to_orig_img(
                #         cam=pred_cam,
                #         bbox=bboxes,
                #         # img_width=orig_width,
                #         # img_height=orig_height,
                #         img_width=orig_width // vid_ratio,
                #         img_height=orig_height // vid_ratio
                #     )

                #     output_dict = {
                #         'pred_cam': pred_cam,
                #         'orig_cam': orig_cam,
                #         'verts': pred_verts,
                #         'pose': pred_pose,
                #         'betas': pred_betas,
                #         'joints3d': pred_joints3d,
                #         'bboxes': bboxes,
                #         'frame_ids': frames,
                #     }

                #     pred_results[person_id] = output_dict

                key_list = list(tracking_results.keys())
                if len(key_list) > 0:
                    person_id = key_list[0]
                    # bboxes = tracking_results[person_id]['bbox']
                    # frames = tracking_results[person_id]['frames']

                    # dataset = Inference_camera(
                    #     img_shape,
                    #     frames=frames,
                    #     bboxes=bboxes,
                    #     scale=bbox_scale,
                    # )

                    # bboxes = dataset.bboxes
                    # frames = dataset.frames
                    # dataloader = DataLoader(dataset, batch_size=1, num_workers=1)

                    # with torch.no_grad():
                    #     pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

                    #     batch = next(iter(dataloader)).to(device)
                    #     preds_dict, _ = model(batch)

                    #     output = preds_dict['smpl_out'][-1]

                    #     pred_cam.append(output['theta'][:, :3].reshape(1, -1))
                    #     pred_verts.append(output['verts'].reshape(1, -1, 3))
                    #     pred_pose.append(output['theta'][:, 13:85].reshape(1, -1))
                    #     pred_betas.append(output['theta'][:, 3:13].reshape(1, -1))
                    #     pred_joints3d.append(output['kp_3d'].reshape(1, -1, 3))

                    #     pred_cam = torch.cat(pred_cam, dim=0)
                    #     pred_verts = torch.cat(pred_verts, dim=0)
                    #     pred_pose = torch.cat(pred_pose, dim=0)
                    #     pred_betas = torch.cat(pred_betas, dim=0)
                    #     pred_joints3d = torch.cat(pred_joints3d, dim=0)

                    #     del batch

                    bboxes = tracking_results[person_id]['bbox']
                    bbox = bboxes[0]
                    frames = tracking_results[person_id]['frames']
                    scale = bbox_scale
                    crop_size = 224

                    img_RGB = cv2.cvtColor(img_shape, cv2.COLOR_BGR2RGB)

                    norm_img, _, _ = get_single_image_crop_demo(
                        img_RGB,
                        bbox,
                        kp_2d=None,
                        scale=scale,
                        crop_size=crop_size)
                    
                    norm_img = norm_img.reshape((1, 3, crop_size, crop_size))
                    
                    with torch.no_grad():
                        # pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d = [], [], [], [], []
                        pred_cam, pred_verts, pred_joints3d = [], [], []

                        preds_dict, _ = model(norm_img.to(device))

                        output = preds_dict['smpl_out'][-1]

                        pred_cam = torch.cat([output['theta'][:, :3].reshape(1, -1)], dim=0)
                        pred_verts = torch.cat([output['verts'].reshape(1, -1, 3)], dim=0)
                        pred_joints3d = torch.cat([output['kp_3d'].reshape(1, -1, 3)], dim=0)

                    # ========= Save results to a pickle file ========= #
                    pred_cam = pred_cam.cpu().numpy()
                    pred_verts = pred_verts.cpu().numpy()
                    pred_joints3d = pred_joints3d.cpu().numpy()

                    orig_cam = convert_crop_cam_to_orig_img(
                        cam=pred_cam,
                        bbox=bboxes,
                        img_width=orig_width,
                        img_height=orig_height,
                    )

                    # orig_cam = convert_crop_cam_to_orig_img(
                    #     cam=pred_cam,
                    #     bbox=bboxes,
                    #     img_width=orig_width // vid_ratio,
                    #     img_height=orig_height // vid_ratio
                    # )

                    # output_dict = {
                    #     'pred_cam': pred_cam,
                    #     'orig_cam': orig_cam,
                    #     'verts': pred_verts,
                    #     'pose': pred_pose,
                    #     'betas': pred_betas,
                    #     'joints3d': pred_joints3d,
                    #     'bboxes': bboxes,
                    #     'frame_ids': frames,
                    # }

                    output_dict = {
                        'orig_cam': orig_cam,
                        'verts': pred_verts,
                        'joints3d': pred_joints3d,
                        'frame_ids': frames,
                    }

                    pred_results = {}
                    pred_results[person_id] = output_dict

                    frame_result = prepare_rendering_results(pred_results, 1)[0]

                    person_data = frame_result[next(iter(frame_result))]
                    frame_verts = person_data['verts']
                    frame_cam = person_data['cam']
                    frame_joints3d = person_data['joints3d']

                    img_shape = renderer(
                        frame_verts,
                        img=img_shape,
                        cam=frame_cam,
                        color_type='purple',
                        mesh_filename=None,

                        # kp_3d=frame_joints3d
                    )
                else:
                    img_shape = img.copy()

                # img_shape = cv2.resize(img_shape, dsize=(orig_width, orig_height), interpolation=cv2.INTER_CUBIC)

                if together:
                    img_shape = np.concatenate((img_shape, img), axis=1)   

                end = time.time()
                fps = 1 / (end - pred_time)

                print(f'Total FPS: {fps:.2f}')

                cv2.imshow("webcam inference", img_shape)
            else:
                pred_time = time.time()
                if is_image:
                    img_np, norm_img = process_image(image, input_res=input_res)
                else:
                    img_np, norm_img = process_image(frame, input_res=input_res)

                if original:
                    img_shape = img_np
                else:
                    with torch.no_grad():
                        preds_dict, _ = model(norm_img.to(device))
                        output = preds_dict['smpl_out'][-1]
                        pred_camera = output['theta'][:, :3]
                        pred_vertices = output['verts']
                        # kp_2d = output['kp_2d'][-1]
                        smpl_kp_3d = output['smpl_kp_3d'][0].cpu().numpy()
                        kp_3d = output['kp_3d'][0].cpu().numpy()

                    # preds_dict['smpl_out']: multiple smpl results
                    # output is last smpl prediction of preds_dict['smpl_out']
                    # output has {'theta', 'verts', 'kp_2d', 'kp_3d', 'smpl_kp_3d', 'rotmat', 'pred_cam', 'pred_shape', 'pred_pose'}

                    # Calculate camera parameters for rendering
                    camera_translation = torch.stack([pred_camera[:, 1], pred_camera[:, 2], 2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9)], dim=-1)
                    camera_translation = camera_translation[0].cpu().numpy()
                    pred_vertices = pred_vertices[0].cpu().numpy()

                    #img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

                    # Render front-view shape

                    if threshold:
                        if camera_translation[2] < constants.FOCAL_LENGTH / 100:
                            img_shape = cv2.resize(img_np, dsize=(700, 700), interpolation=cv2.INTER_CUBIC)
                        else:
                            save_mesh_path = None
                            img_shape = renderer(
                                pred_vertices,
                                img=img_np,
                                cam=pred_camera[0].cpu().numpy(),
                                color_type='purple',
                                mesh_filename=save_mesh_path,

                                # smpl_kp_3d=smpl_kp_3d,  # Red
                                # kp_3d=kp_3d   # Blue
                            )
                            # img_shape = renderer.draw_smpl_kp_3d(
                            #     smpl_kp_3d,
                            #     img=img_shape,
                            #     cam=pred_camera[0].cpu().numpy(),
                            # )
                    else:
                        save_mesh_path = None
                        img_shape = renderer(
                            pred_vertices,
                            img=img_np,
                            cam=pred_camera[0].cpu().numpy(),
                            color_type='purple',
                            mesh_filename=save_mesh_path,

                            # smpl_kp_3d=smpl_kp_3d,  # Red
                            # kp_3d=kp_3d   # Blue
                        )

                # save_mesh_path = None
                # img_shape = renderer(
                #     pred_vertices,
                #     img=img_np,
                #     cam=pred_camera[0].cpu().numpy(),
                #     color_type='white',
                #     mesh_filename=save_mesh_path,
                # )
                # img_shape = renderer.draw_smpl_kp_3d(
                #     kp_3d,
                #     img=img_shape,
                #     cam=pred_camera[0].cpu().numpy(),
                #     color_type='white',
                # )
                    
                if is_image:
                    if together:
                        img_shape = cv2.resize(img_shape, dsize=(image.shape[0], image.shape[0]), interpolation=cv2.INTER_CUBIC)
                        img_shape = np.concatenate((img_shape, image), axis=1)
                    else:
                        img_shape = cv2.resize(img_shape, dsize=(new_size, new_size), interpolation=cv2.INTER_CUBIC)
                else:
                    if together:
                        img_shape = cv2.resize(img_shape, dsize=(frame.shape[0], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
                        img_shape = np.concatenate((img_shape, frame), axis=1)   
                    else:
                        img_shape = cv2.resize(img_shape, dsize=(new_size, new_size), interpolation=cv2.INTER_CUBIC)
                end = time.time()
                fps = 1 / (end - pred_time)

                print(f'FPS: {fps:.2f}')

                cv2.imshow("webcam inference", img_shape)
                # cv2.imshow("test", img_np)
                # cv2.imshow("test", frame)

        if cv2.waitKey(33) == ord('q'):
            print('Quit')
            break
        elif cv2.waitKey(33) == 0x1B:
            print('Quit')
            break
        # elif cv2.waitKey(33) == ord('i'):
        #     is_image = not is_image
        # elif cv2.waitKey(33)  == ord('t'):
        #     together = not together
        # elif cv2.waitKey(33) == ord('f'):
        #     threshold = not threshold
        # elif cv2.waitKey(33) == ord('o'):
        #     original  = not original
        # elif cv2.waitKey(33) == ord('v'):
        #     video = not video
        

    webcam.release()
    cv2.destroyAllWindows()
