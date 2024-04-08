# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os
import pycocotools.mask as mask_util
import math
import torchvision

from .colormap import colormap
from .keypoints import get_keypoints
from .imutils import normalize_2d_kp

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.transform import resize

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
    }
    return colors



def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes


def vis_bbox_opencv(img, bbox, thick=1):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def vis_one_image(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, body_uv=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='pdf'):
    """Visual debugging of detections."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    if segms is not None:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    dataset_keypoints, _ = get_keypoints()

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        print(dataset.classes[classes[i]], score)
        # show box (off by default, box_alpha=0.0)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='g',
                          linewidth=0.5, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, dataset),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = ax.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    ax.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)
                if kps[2, i2] > kp_thresh:
                    ax.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = ax.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = ax.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)

    #   DensePose Visualization Starts!!
    ##  Get full IUV image out
    if body_uv is not None and len(body_uv) > 1:
        IUV_fields = body_uv[1]
        #
        All_Coords = np.zeros(im.shape)
        All_inds = np.zeros([im.shape[0], im.shape[1]])
        K = 26
        ##
        inds = np.argsort(boxes[:, 4])
        ##
        for i, ind in enumerate(inds):
            entry = boxes[ind, :]
            if entry[4] > 0.65:
                entry = entry[0:4].astype(int)
                ####
                output = IUV_fields[ind]
                ####
                All_Coords_Old = All_Coords[entry[1]: entry[1] + output.shape[1], entry[0]:entry[0] + output.shape[2], :]
                All_Coords_Old[All_Coords_Old == 0] = output.transpose([1, 2, 0])[All_Coords_Old == 0]
                All_Coords[entry[1]: entry[1] + output.shape[1], entry[0]:entry[0] + output.shape[2], :] = All_Coords_Old
                ###
                CurrentMask = (output[0, :, :] > 0).astype(np.float32)
                All_inds_old = All_inds[entry[1]: entry[1] + output.shape[1], entry[0]:entry[0] + output.shape[2]]
                All_inds_old[All_inds_old == 0] = CurrentMask[All_inds_old == 0] * i
                All_inds[entry[1]: entry[1] + output.shape[1], entry[0]:entry[0] + output.shape[2]] = All_inds_old
        #
        All_Coords[:, :, 1:3] = 255. * All_Coords[:, :, 1:3]
        All_Coords[All_Coords > 255] = 255.
        All_Coords = All_Coords.astype(np.uint8)
        All_inds = All_inds.astype(np.uint8)
        #
        IUV_SaveName = os.path.basename(im_name).split('.')[0] + '_IUV.png'
        INDS_SaveName = os.path.basename(im_name).split('.')[0] + '_INDS.png'
        cv2.imwrite(os.path.join(output_dir, '{}'.format(IUV_SaveName)), All_Coords)
        cv2.imwrite(os.path.join(output_dir, '{}'.format(INDS_SaveName)), All_inds)
        print('IUV written to: ', os.path.join(output_dir, '{}'.format(IUV_SaveName)))
        ###
        ### DensePose Visualization Done!!
    #
    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    plt.close('all')

    #   SMPL Visualization
    if body_uv is not None and len(body_uv) > 2:
        smpl_fields = body_uv[2]
        #
        All_Coords = np.zeros(im.shape)
        # All_inds = np.zeros([im.shape[0], im.shape[1]])
        K = 26
        ##
        inds = np.argsort(boxes[:, 4])
        ##
        for i, ind in enumerate(inds):
            entry = boxes[ind, :]
            if entry[4] > 0.75:
                entry = entry[0:4].astype(int)
                center_roi = [(entry[2]+entry[0]) / 2., (entry[3]+entry[1]) / 2.]
                ####
                output, center_out = smpl_fields[ind]
                ####
                x1_img = max(int(center_roi[0] - center_out[0]), 0)
                y1_img = max(int(center_roi[1] - center_out[1]), 0)

                x2_img = min(int(center_roi[0] - center_out[0]) + output.shape[2], im.shape[1])
                y2_img = min(int(center_roi[1] - center_out[1]) + output.shape[1], im.shape[0])

                All_Coords_Old = All_Coords[y1_img:y2_img, x1_img:x2_img, :]

                x1_out = max(int(center_out[0] - center_roi[0]), 0)
                y1_out = max(int(center_out[1] - center_roi[1]), 0)

                x2_out = x1_out + (x2_img - x1_img)
                y2_out = y1_out + (y2_img - y1_img)

                output = output[:, y1_out:y2_out, x1_out:x2_out]

                # All_Coords_Old = All_Coords[entry[1]: entry[1] + output.shape[1], entry[0]:entry[0] + output.shape[2],
                #                  :]
                All_Coords_Old[All_Coords_Old == 0] = output.transpose([1, 2, 0])[All_Coords_Old == 0]
                All_Coords[y1_img:y2_img, x1_img:x2_img, :] = All_Coords_Old
                ###
                # CurrentMask = (output[0, :, :] > 0).astype(np.float32)
                # All_inds_old = All_inds[entry[1]: entry[1] + output.shape[1], entry[0]:entry[0] + output.shape[2]]
                # All_inds_old[All_inds_old == 0] = CurrentMask[All_inds_old == 0] * i
                # All_inds[entry[1]: entry[1] + output.shape[1], entry[0]:entry[0] + output.shape[2]] = All_inds_old
        #
        All_Coords = 255. * All_Coords
        All_Coords[All_Coords > 255] = 255.
        All_Coords = All_Coords.astype(np.uint8)

        image_stacked = im[:, :, ::-1]
        image_stacked[All_Coords > 20] = All_Coords[All_Coords > 20]
        # All_inds = All_inds.astype(np.uint8)
        #
        SMPL_SaveName = os.path.basename(im_name).split('.')[0] + '_SMPL.png'
        smpl_image_SaveName = os.path.basename(im_name).split('.')[0] + '_SMPLimg.png'
        # INDS_SaveName = os.path.basename(im_name).split('.')[0] + '_INDS.png'
        cv2.imwrite(os.path.join(output_dir, '{}'.format(SMPL_SaveName)), All_Coords)
        cv2.imwrite(os.path.join(output_dir, '{}'.format(smpl_image_SaveName)), image_stacked)
        # cv2.imwrite(os.path.join(output_dir, '{}'.format(INDS_SaveName)), All_inds)
        print('SMPL written to: ', os.path.join(output_dir, '{}'.format(SMPL_SaveName)))
        ###
        ### SMPL Visualization Done!!
    #
    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    plt.close('all')


def vis_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name=None, nrow=8, padding=1, pad_value=1):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True, pad_value=pad_value)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break

            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            flip = 1
            count = -1

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                flip *= -1
                count += 1
                if joint_vis[0]:
                    try:
                        if flip > 0:
                            cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 1, [255, 0, 0], 1)
                        else:
                            cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 1, [0, 255, 0], 1)
                        cv2.putText(ndarr, str(count), (int(joint[0]), int(joint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (255, 0, 0), 1)
                    except Exception as e:
                        print(e)
            k = k + 1

    return ndarr


def vis_img_3Djoint(batch_img, joints, pairs=None, joint_group=None):
    n_sample = joints.shape[0]
    max_show = 2
    if n_sample > max_show:
        if batch_img is not None:
            batch_img = batch_img[:max_show]
        joints = joints[:max_show]
        n_sample = max_show

    color = ['#00B0F0', '#00B050', '#DC6464', '#207070', '#BC4484']
    # color = ['g', 'b', 'r']

    def m_l_r(idx):

        if joint_group is None:
            return 1

        for i in range(len(joint_group)):
            if idx in joint_group[i]:
                return i

    for i in range(n_sample):
        if batch_img is not None:
            # ax_img = plt.subplot(n_sample, 2, i * 2 + 1)
            ax_img = plt.subplot(2, n_sample, i + 1)
            img_np = batch_img[i].cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0)) # H*W*C
            ax_img.imshow(img_np)
            ax_img.set_axis_off()
            ax_pred = plt.subplot(2, n_sample, n_sample + i + 1, projection='3d')

        else:
            ax_pred = plt.subplot(1, n_sample, i + 1, projection='3d')

        plot_kps = joints[i]
        if plot_kps.shape[1] > 2:
            if joint_group is None:
                ax_pred.scatter(plot_kps[:, 2], plot_kps[:, 0], plot_kps[:, 1], s=10, marker='.')
                ax_pred.scatter(plot_kps[0, 2], plot_kps[0, 0], plot_kps[0, 1], s=10, c='g', marker='.')
            else:
                for j in range(len(joint_group)):
                    ax_pred.scatter(plot_kps[joint_group[j], 2], plot_kps[joint_group[j], 0], plot_kps[joint_group[j], 1], s=30, c=color[j], marker='s')

            if pairs is not None:
                for p in pairs:
                    ax_pred.plot(plot_kps[p, 2], plot_kps[p, 0], plot_kps[p, 1], c=color[m_l_r(p[1])], linewidth=2)

        # ax_pred.set_axis_off()

        ax_pred.set_aspect('equal')
        set_axes_equal(ax_pred)

        ax_pred.xaxis.set_ticks([])
        ax_pred.yaxis.set_ticks([])
        ax_pred.zaxis.set_ticks([])



def vis_img_2Djoint(batch_img, joints, pairs=None, joint_group=None):
    n_sample = joints.shape[0]
    max_show = 2
    if n_sample > max_show:
        if batch_img is not None:
            batch_img = batch_img[:max_show]
        joints = joints[:max_show]
        n_sample = max_show

    color = ['#00B0F0', '#00B050', '#DC6464', '#207070', '#BC4484']
    # color = ['g', 'b', 'r']

    def m_l_r(idx):

        if joint_group is None:
            return 1

        for i in range(len(joint_group)):
            if idx in joint_group[i]:
                return i

    for i in range(n_sample):
        if batch_img is not None:
            # ax_img = plt.subplot(n_sample, 2, i * 2 + 1)
            ax_img = plt.subplot(2, n_sample, i + 1)
            img_np = batch_img[i].cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0)) # H*W*C
            ax_img.imshow(img_np)
            ax_img.set_axis_off()
            ax_pred = plt.subplot(2, n_sample, n_sample + i + 1)

        else:
            ax_pred = plt.subplot(1, n_sample, i + 1)

        plot_kps = joints[i]
        if plot_kps.shape[1] > 1:
            if joint_group is None:
                ax_pred.scatter(plot_kps[:, 0], plot_kps[:, 1], s=300, c='#00B0F0', marker='.')
                # ax_pred.scatter(plot_kps[:, 0], plot_kps[:, 1], s=10, marker='.')
                # ax_pred.scatter(plot_kps[0, 0], plot_kps[0, 1], s=10, c='g', marker='.')
            else:
                for j in range(len(joint_group)):
                    ax_pred.scatter(plot_kps[joint_group[j], 0], plot_kps[joint_group[j], 1], s=100, c=color[j], marker='o')

            if pairs is not None:
                for p in pairs:
                    ax_pred.plot(plot_kps[p, 0], plot_kps[p, 1], c=color[m_l_r(p[1])], linestyle=':', linewidth=3)

        ax_pred.set_axis_off()

        ax_pred.set_aspect('equal')
        ax_pred.axis('equal')
        # set_axes_equal(ax_pred)

        ax_pred.xaxis.set_ticks([])
        ax_pred.yaxis.set_ticks([])
        # ax_pred.zaxis.set_ticks([])

def draw_skeleton(image, kp_2d, dataset='common', unnormalize=True, thickness=2):

    if unnormalize:
        kp_2d[:,:2] = normalize_2d_kp(kp_2d[:,:2], 224, inv=True)

    kp_2d[:,2] = kp_2d[:,2] > 0.3
    kp_2d = np.array(kp_2d, dtype=int)

    rcolor = get_colors()['red'].tolist()
    pcolor = get_colors()['green'].tolist()
    lcolor = get_colors()['blue'].tolist()

    common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
    for idx,pt in enumerate(kp_2d):
        if pt[2] > 0: # if visible
            if idx % 2 == 0:
                color = rcolor
            else:
                color = pcolor
            cv2.circle(image, (pt[0], pt[1]), 4, color, -1)
            # cv2.putText(image, f'{idx}', (pt[0]+1, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
    
    if dataset == 'common' and len(kp_2d) != 15:
        return image

    skeleton = eval(f'kp_utils.get_{dataset}_skeleton')()
    for i,(j1,j2) in enumerate(skeleton):
        if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0: # if visible
            if dataset == 'common':
                color = rcolor if common_lr[i] == 0 else lcolor
            else:
                color = lcolor if i % 2 == 0 else rcolor
            pt1, pt2 = (kp_2d[j1, 0], kp_2d[j1, 1]), (kp_2d[j2, 0], kp_2d[j2, 1])
            cv2.line(image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

    return image

# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])