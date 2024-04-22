#pip install imgui[glfw]
import imgui
import glfw
import numpy as np
import dartpy as dart
import viewer.gl_function as mygl
import quaternion
import torch

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from imgui.integrations.glfw import GlfwRenderer
from viewer.TrackBall import TrackBall
from learning.ray_model import loading_network
from numba import jit
from core.env import Env

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import os
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
# os.environ["SDL_VIDEO_X11_FORCE_EGL"] = "1"
# os.environ['PYGLFW_LIBRARY_VARIANT'] = 'wayland'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

plt.ion()
fig, ax = plt.subplots()

from PIL import Image


OP_JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
]

SPIN_JOINT_NAMES = [
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',   # 2
'Left Hip',
'Left Knee',    # 4
'Left Ankle',
'Right Wrist',   # 6
'Right Elbow',
'Right Shoulder',  # 8
'Left Shoulder',
'Left Elbow',  # 10
'Left Wrist',
'Neck (LSP)',  # 12
'Top of Head (LSP)',
'Pelvis (MPII)',  # 14
'Thorax (MPII)',
'Spine (H36M)',  # 16
'Jaw (H36M)',
'Head (H36M)',  # 18
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

edges = [
    (0,1),  (1,4),  (4,7),  (7,10),
    (0,2),  (2,5),  (5,8),  (8,11),
    (0,3),  (3,6),  (6,9),
    (9,14),  (14,17),  (17,19),  (19,21),  (21,23),
    (9,13),  (13,16),  (16,18),  (18,20),  (20,22),
    (9,12),  (12,15)
]

## Light Option 
ambient = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
diffuse = np.array([0.6, 0.6, 0.6, 1.0], dtype=np.float32)
front_mat_shininess = np.array([60.0], dtype=np.float32)
front_mat_specular = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
front_mat_diffuse = np.array([0.5, 0.28, 0.38, 1.0], dtype=np.float32)
lmodel_ambient = np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32)
lmodel_twoside = np.array([GL_FALSE])
light_pos = [    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 
                np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 3.0, 0.0, 0.0], dtype=np.float32)]

def initGL():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # glEnable(GL_BLEND)
    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
    # glShadeModel(GL_SMOOTH)
    # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    # glEnable(GL_LIGHT0)
    # glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
    # glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
    # glLightfv(GL_LIGHT0, GL_POSITION, light_pos[0])
    # glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient)
    # glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside)

    # glEnable(GL_LIGHT1)
    # glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse)
    # glLightfv(GL_LIGHT1, GL_POSITION, light_pos[1])

    # glEnable(GL_LIGHT2)
    # glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse)
    # glLightfv(GL_LIGHT2, GL_POSITION, light_pos[2])

    # glEnable(GL_LIGHTING)

    # glEnable(GL_COLOR_MATERIAL)
    # glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess)
    # glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, front_mat_specular)
    # glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, front_mat_diffuse)

    # glEnable(GL_DEPTH_TEST)
    # glDepthFunc(GL_LEQUAL)
    # glEnable(GL_NORMALIZE)
    # glEnable(GL_MULTISAMPLE)
    

## GLFW Initilization Function
def impl_glfw_init(window_name="Muscle Simulation", width=1920, height=1080):
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    monitor = glfw.get_primary_monitor()
    pos = glfw.get_monitor_pos(monitor)
    size = glfw.get_window_size(window)
    mode = glfw.get_video_mode(monitor)

    glfw.set_window_pos(
        window,
        int(pos[0] + (mode.size.width - size[0]) / 2),
        int(pos[1] + (mode.size.height - size[1]) / 2))

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

class GLFWApp():
    def __init__(self):
        super().__init__()

        ## Settin window option and initialization        
        self.name = "Muscle Simulation"
        self.width = 1920 
        self.height = 1080
        
        ## Camera Setting
        self.perspective = 45.0
        self.trackball = TrackBall()
        self.eye = np.array([0.0, 0.0, 1.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.trans = np.array([0.0, 0.0, 0.0])
        self.zoom = 1.0

        self.trackball.set_trackball(np.array([self.width * 0.5, self.height * 0.5]), self.width * 0.5)
        self.trackball.set_quaternion(np.quaternion(1.0, 0.0, 0.0, 0.0))
        
        ## Camera transform flag
        self.mouse_down = False
        self.rotate = False
        self.translate = False
        self.focus = 0

        self.mouse_x = 0
        self.mouse_y = 0
        self.motion_skel = None

        ## Flag         
        self.is_simulation = False
        self.draw_mesh = False
        self.draw_target_motion = False
        self.draw_pd_target = False
        self.draw_collision = False
        self.draw_muscles = True
        self.draw_vae_plot = True
        self.draw_ground = True
        self.draw_sim_character = True

        self.smpl_joint = None
        self.smpl_vert = None
        self.smpl_cam_trans = None

        ## Motion 
        self.motion_list = []
        self.motion_idx = 0

        # load all file list which ancester is the data/motion recursively
        for root, _, files in os.walk("data/motion"):
            for file in files:
                if file.endswith(".bvh"):
                    self.motion_list.append(os.path.join(root, file))

        glutInit()

        imgui.create_context()
        self.window = impl_glfw_init(self.name, self.width, self.height)
        self.impl = GlfwRenderer(self.window)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glShadeModel(GL_SMOOTH)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos[0])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient)
        glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside)

        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT1, GL_POSITION, light_pos[1])

        glEnable(GL_LIGHT2)
        glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse)
        glLightfv(GL_LIGHT2, GL_POSITION, light_pos[2])

        glEnable(GL_LIGHTING)

        glEnable(GL_COLOR_MATERIAL)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, front_mat_specular)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, front_mat_diffuse)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_NORMALIZE)
        glEnable(GL_MULTISAMPLE)
    

        # Set Callback Function        
        ## Framebuffersize Callback Function
        def framebuffer_size_callback(window, width, height):
            self.width = width
            self.height = height
            glViewport(0, 0, width, height)
        glfw.set_framebuffer_size_callback(self.window, framebuffer_size_callback)

        ## Mouse Callback Function 
        ### mouseButtonCallback
        def mouseButtonCallback(window, button, action, mods):
            # wantcapturemouse
            if not imgui.get_io().want_capture_mouse:
                self.mousePress(button, action, mods)
        glfw.set_mouse_button_callback(self.window, mouseButtonCallback)

        ### cursorPosCall back
        def cursorPosCallback(window, xpos, ypos):
            if not imgui.get_io().want_capture_mouse:
                self.mouseMove(xpos, ypos)
        glfw.set_cursor_pos_callback(self.window, cursorPosCallback)

        ### scrollCallback
        def scrollCallback(window, xoffset, yoffset):
            if not imgui.get_io().want_capture_mouse:
                self.mouseScroll(xoffset, yoffset)
        glfw.set_scroll_callback(self.window, scrollCallback)

        ## Keyboard Callback Function  
        def keyCallback(window, key, scancode, action, mods):
            if not imgui.get_io().want_capture_mouse:
                self.keyboardPress(key, scancode, action, mods)
        glfw.set_key_callback(self.window, keyCallback)

        self.env = None
        self.nn = None
        self.mus_nn = None

        ## For Graph Logging
        self.reward_buffer = []

        ## For Value Logging
        self.value_buffer = []
        self.marginalized_value_buffer = None
        self.sampling_prob_value_buffer = None
        self.muscle_accuracy = []

        ## VAE 
        self.vae_codebook_frequency = None
        self.vae_random_sampling = False
        self.reduced_codebook = None
        self.scatter = None
        self.sim_count = 0

        ## Record 
        self.is_recording = False
        self.record_idx = 0

    def setCamera(self):
        if self.focus == 1:
            self.trans = -self.env.skel.getCOM()
            self.trans[1] = -1
            self.trans *= 1000

    def setEnv(self, env):
        self.env = env
        self.motion_skel = self.env.skel.clone()
        self.reset()
    
    def loadNetwork(self, path):
        self.nn, mus_nn, spd_nn, marginal_nn, env_str = loading_network(path)
        if env_str != None:
            self.setEnv(Env(env_str))   
        
        self.env.muscle_nn = mus_nn
        self.env.spd_nn = spd_nn
        
        if marginal_nn != None:
            scaling_weight = 4
            self.env.marginal_nn = marginal_nn
            dist_resolution = 100
            input = np.ones((len(self.env.bvhs),dist_resolution,2))
            input[:,:,1] = np.array(range(dist_resolution)) * 0.01
            for i in range(len(self.env.bvhs)):
                input[i,:,0] = i
            phases = torch.tensor(input.reshape(-1, input.shape[-1]), device="cuda", dtype=torch.float32)
            
            self.marginalized_value_buffer = self.env.marginal_nn.get_value(phases)[:, 0]
            
            self.env.sampling_prob_values = self.marginalized_value_buffer - np.min(self.marginalized_value_buffer)
            self.env.sampling_prob_values = np.max(self.env.sampling_prob_values) - self.env.sampling_prob_values
            self.env.sampling_prob_values = self.env.sampling_prob_values ** scaling_weight
            self.env.sampling_prob_values = self.env.sampling_prob_values / np.sum(self.env.sampling_prob_values)
            
        if self.nn.policy.config["VAE"] == "vq":
            self.vae_codebook_frequency = np.zeros(self.nn.policy.p_fc.num_embeddings, dtype=np.int32)
            ## Umap Code about the self.nn.policy.p_fc.quantizer.embeddings
            if self.draw_vae_plot:
                import umap
                reducer = umap.UMAP()
                codebook = self.nn.policy.p_fc.quantizer.embeddings.weight.cpu().detach().numpy()
                self.reduced_codebook = reducer.fit_transform(codebook)
                self.scatter = plt.scatter(self.reduced_codebook[:,0], self.reduced_codebook[:,1])
                self.scatter_colors = np.ones((len(self.reduced_codebook[:,0]), 4)) * 0.5
    ## mousce button callback function
    def mousePress(self, button, action, mods):
        if action == glfw.PRESS:
            self.mouse_down = True
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.rotate = True
                self.trackball.start_ball(self.mouse_x, self.height - self.mouse_y)
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.translate = True
        elif action == glfw.RELEASE:
            self.mouse_down = False
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.rotate = False
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.translate = False

    ## mouse move callback function
    def mouseMove(self, xpos, ypos):
        dx = xpos - self.mouse_x
        dy = ypos - self.mouse_y

        self.mouse_x = xpos
        self.mouse_y = ypos

        if self.rotate:
            if dx != 0 or dy != 0:
                self.trackball.update_ball(xpos, self.height - ypos)

        if self.translate:
            rot = quaternion.as_rotation_matrix(self.trackball.curr_quat)
            self.trans += (1.0 / self.zoom) * rot.transpose() @ np.array([dx, -dy, 0.0])

    ## mouse scroll callback function
    def mouseScroll(self, xoffset, yoffset):
        if yoffset < 0:
            self.eye *= 1.05
        elif (yoffset > 0) and (np.linalg.norm(self.eye) > 0.5):
            self.eye *= 0.95
    

    def update(self):
        if self.nn is not None:
            obs = self.env.get_obs()
            if self.vae_random_sampling:
                action = self.nn.vae_sampling_action(obs)
            else: 
                action = self.nn.get_action(obs)
            _, _, done, _ = self.env.step(action)
            # if self.env.muscles:
            #     self.muscle_accuracy += self.env.muscle_accuracy
            if self.nn.policy.config["VAE"] == "vq":
                self.vae_codebook_frequency[self.nn.policy.p_fc.quantizer.idx] += 1
        else:
            _, _, done, _ = self.env.step(np.zeros(self.env.num_action))

        self.reward_buffer.append(self.env.get_reward(True))
        if self.nn:
            self.value_buffer.append(self.nn.get_value(self.env.get_obs())[0])
        
        self.sim_count += 1

    def drawShape(self, shape, color):
        if not shape:
            return
        
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glColor4d(color[0], color[1], color[2], color[3])
        if not self.draw_mesh:
            ## check the shape type
            if type(shape) == dart.dynamics.BoxShape:
                mygl.draw_cube(shape.getSize())
            
    def drawSkeleton(self, pos, color = np.array([0.5, 0.5, 0.5, 0.3])):
        self.motion_skel.setPositions(pos)

        for bn in self.motion_skel.getBodyNodes():
            glPushMatrix()
            glMultMatrixd(bn.getWorldTransform().matrix().transpose())
            
            for sn in bn.getShapeNodes():
                if not sn:
                    return
                va = sn.getVisualAspect()

                if not va or va.isHidden():
                    return
                
                glPushMatrix()
                glMultMatrixd(sn.getRelativeTransform().matrix().transpose())
                self.drawShape(sn.getShape(), color)               
                
                glPopMatrix()
            glPopMatrix()
        pass

    def drawCollision(self):
        c_results = self.env.world.getLastCollisionResult()
        
        for c in c_results.getContacts():
            v = c.point
            f = c.force / 1000.0

            glLineWidth(2.0)
            glColor3f(0.8, 0.8, 0.2)
            glBegin(GL_LINES)
            glVertex3f(v[0], v[1], v[2])
            glVertex3f(v[0] + f[0], v[1] + f[1], v[2] + f[2])
            glEnd()
            glColor3f(0.8, 0.8, 0.2)
            glPushMatrix()
            glTranslated(v[0], v[1], v[2])
            mygl.draw_sphere(0.01)
            glPopMatrix()
        
        # mygl.draw_sphere(1.0)

        pass

    def drawMuscles(self):
        vertexarray = np.concatenate(self.env.muscle_pos,dtype=np.float32)
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glVertexPointer(3, GL_FLOAT, 0, vertexarray)
        f0s = self.env.muscles.getF0s()
        mus_idxs = self.env.muscle_vbo_idx
        idx = 0
        for mus_idx in mus_idxs:
            a = self.env.muscle_activation_levels[idx]
            glLineWidth(3.0 * f0s[idx] / 1000.0)
            glColor4d(1.0 * a,  0.2 * a, 0.2 * a, 0.2 + 0.6 * a)
            glDrawArrays(GL_LINE_STRIP, mus_idx[0], mus_idx[-1] - mus_idx[0] + 1)
            idx+=1
        glDisableClientState(GL_VERTEX_ARRAY)
        glFlush()
            
    def drawSimFrame(self):
        
        initGL()
        self.setCamera()     
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glViewport(0, 0, self.width, self.height)
        
        gluPerspective(self.perspective, (self.width / self.height), 0.1, 100.0)
        gluLookAt(self.eye[0], self.eye[1], self.eye[2], 0.0, 0.0, -1.0, self.up[0], self.up[1], self.up[2])
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.trackball.set_center(np.array([self.width * 0.5, self.height * 0.5]))
        self.trackball.set_radius(min(self.width, self.height) * 0.4)
        self.trackball.apply_gl_roatation()

        glScalef(self.zoom, self.zoom, self.zoom)
        glTranslatef(self.trans[0] * 0.001, self.trans[1] *0.001, self.trans[2] * 0.001)
        glEnable(GL_DEPTH_TEST)

        if self.draw_ground:
            mygl.drawGround(-1E-3)

        if self.mouse_down:
            mygl.draw_axis()
        

        if self.draw_sim_character:
            self.drawSkeleton(self.env.skel.getPositions())

        if self.smpl_joint is not None:
            self.drawJoints(None)

            # Aligning to Skeleton; Can't make skeleton and SMPL separate
            # self.motion_skel.setPositions(self.env.skel.getPositions())
            # for bn in self.motion_skel.getBodyNodes():
            #     if bn.getName() == 'Pelvis':
            #         glPushMatrix()
            #         mulmat = bn.getWorldTransform().matrix()
            #         mulmat[:3, :3] = [[1,0,0],[0,1,0],[0,0,1]]
            #         glMultMatrixd(mulmat.transpose())
            #         root_h = mulmat[1][3]       
            #         for sn in bn.getShapeNodes():
            #             if not sn:
            #                 return
            #             va = sn.getVisualAspect()

            #             if not va or va.isHidden():
            #                 return
                        
            #             glPushMatrix()
            #             glMultMatrixd(sn.getRelativeTransform().matrix().transpose())
                        
            #             self.drawJoints(root_h)            
            #             glPopMatrix()
            #             break
            #         glPopMatrix()
            #         break

        glPushMatrix()
        glScalef(1,1E-3,1)
        
        if self.draw_sim_character:
            self.drawSkeleton(self.env.skel.getPositions(), np.array([0.3, 0.3, 0.3, 1.0]))
        
        glPopMatrix()
        
        if self.draw_target_motion:
            for target_pos in self.env.target_positions:
                self.drawSkeleton(target_pos, np.array([1.0, 0.3, 0.3, 0.5]))
                
        
        if self.draw_pd_target:
            self.drawSkeleton(self.env.pd_target, np.array([0.3, 0.3,1.0, 0.5]))
            
        if self.draw_collision:
            self.drawCollision()
        
        if self.env.muscles and self.draw_muscles:
            self.drawMuscles()

        # if self.smpl_joint is not None:
        #     self.drawJoints()

    def drawJoints(self, h, color = np.array([1.0, 0.3, 0.3, 0.1])):
        # glTranslatef(0, -h, 0)
        glLineWidth(10)

        for i in range(len(self.smpl_joint)):
            glPushMatrix()
            
            c = self.smpl_cam_trans[i]
            glTranslatef(-c[0], 0, -c[2])
            # glTranslatef(-c[0], c[1], -c[2])

            js = self.smpl_joint[i]
            l = np.argmax(js[:, 1]) # smpl joint is flipped in x axis

            # This can't make jump motions; SMPL joints always attached to floor
            lowest = np.max(js[l])
            glTranslatef(0, lowest, 0)
            
            glRotatef(180, 1, 0, 0)

            # glBegin(GL_LINES)
            # glColor4f(1,0,0,0.1)
            # glVertex3f(0, 0, 0)
            # glVertex3f(0, lowest - h, 0)
            # glEnd()
            
            glColor4d(color[0], color[1], color[2], color[3])
            glBegin(GL_LINES)
            for edge in edges:
                s = js[edge[0]]
                e = js[edge[1]]

                glVertex3f(s[0], s[1], s[2])
                glVertex3f(e[0], e[1], e[2])
            glEnd()

            for j in js:
                glPushMatrix()
                glTranslatef(j[0], j[1], j[2])
                mygl.draw_sphere(0.01)
                glPopMatrix()

            glPushMatrix()
            glColor4f(0,0,1,0.1)

            glTranslatef(js[l][0], js[l][1], js[l][2])
            mygl.draw_sphere(0.02)
            glPopMatrix()

            glPushMatrix()
            glColor4f(0,1,0,0.1)

            glTranslatef(js[0][0], js[0][1], js[0][2])
            mygl.draw_sphere(0.02)
            glPopMatrix()

            glPopMatrix()

        glFlush()


    def drawUIFrame(self):
        imgui.new_frame()
        
        # imgui.show_test_window()

        imgui.set_next_window_size(400,400, condition=imgui.ONCE)
        imgui.set_next_window_position(self.width - 410, 10, condition = imgui.ONCE)        

        # State Information 
        imgui.begin("Information")
        imgui.text("Elapsed\tTime\t:\t%.2f" % self.env.world.getTime())
        
        if imgui.tree_node("Observation"):
            imgui.plot_histogram(
                label="##obs",
                values=self.env.get_obs().astype(np.float32),
                values_count=self.env.num_obs,
                scale_min=-10.0,
                scale_max =10.0,
                graph_size = (imgui.get_content_region_available_width(), 200)
            )
            imgui.tree_pop()
        
        if self.env.marginal_nn and imgui.tree_node("Marginalized Value"):
            dist_resolution = 100

            # # if type of self.marginalized_value_buffer is not numpy type
            if type(self.marginalized_value_buffer) == np.ndarray:
                imgui.plot_histogram(
                    label="Marginalized Value",
                    values=self.marginalized_value_buffer.astype(np.float32),
                    values_count=len(self.marginalized_value_buffer),
                    scale_min=0.0,
                    scale_max=100.0,
                    graph_size=(imgui.get_content_region_available_width(), 200)
                )

                current_idx = dist_resolution * self.env.bvh_idx + int(dist_resolution * (self.env.world.getTime() % self.env.get_current_bvh().bvh_time) // self.env.get_current_bvh().bvh_time)
                dummy_values = np.zeros(dist_resolution * len(self.env.bvhs), dtype=np.float32)
                dummy_values[current_idx] = 1.0
                imgui.plot_histogram(
                    label="##dummy",
                    values=dummy_values,
                    values_count=len(dummy_values),
                    scale_min=0.0,
                    scale_max=1.0,
                    graph_size=(imgui.get_content_region_available_width(), 20)
                )

                ## Sampling Prob Distribution 
                imgui.plot_histogram(
                    label="Sampling Prob",
                    values=self.env.sampling_prob_values.astype(np.float32),
                    values_count=len(self.env.sampling_prob_values),
                    scale_min=0.0,
                    scale_max=np.max(self.env.sampling_prob_values),
                    graph_size=(imgui.get_content_region_available_width(), 50)
                )
            imgui.tree_pop()

        if self.nn and imgui.tree_node("Value"):
            width = 60
            data_width = min(width, len(self.value_buffer))
            values = np.zeros(width, dtype=np.float32)
            values[-data_width:] = np.array(self.value_buffer[-data_width:], dtype=np.float32)
            imgui.plot_lines(
                label="Value",
                values=values,
                overlay_text="Value",
                scale_min=-0.0,
                scale_max=100.0,
                graph_size=(imgui.get_content_region_available_width(), 50)
            )
            imgui.tree_pop()


        if imgui.tree_node("Reward"):
            width = 60
            data_width = min(width, len(self.reward_buffer))                   
            values = [None for _ in range(len(self.reward_buffer[0]))]
            reward_names = ["Total", "q", "dq", "com", "ee"]

            for i in range(len(self.reward_buffer[0])):            
                values[i] = np.zeros(width, dtype=np.float32)
                values[i][-data_width:] = np.array(self.reward_buffer[-data_width:], dtype=np.float32)[:, i]

                imgui.plot_lines(
                    label="Reward_" + reward_names[i],
                    values=values[i],
                    overlay_text = reward_names[i],
                    scale_min=0.0,
                    scale_max=1.0,
                    graph_size=(imgui.get_content_region_available_width(), 50)
                )

            imgui.tree_pop()

        if imgui.tree_node("Rendering Mode"):
            _, self.draw_sim_character = imgui.checkbox("Draw Sim Character", self.draw_sim_character)
            _, self.draw_target_motion = imgui.checkbox("Draw Target Motion", self.draw_target_motion)
            _, self.draw_pd_target = imgui.checkbox("Draw PD Target", self.draw_pd_target)
            _, self.draw_collision = imgui.checkbox("Draw Collision", self.draw_collision)
            _, self.draw_ground = imgui.checkbox("Draw Ground", self.draw_ground)
            if self.env.muscles:
                _, self.draw_muscles = imgui.checkbox("Draw Muscles", self.draw_muscles)

            imgui.tree_pop()
        
        if imgui.tree_node("Metadata"):
            imgui.text(self.env.metadata)
            imgui.tree_pop()
        
        if imgui.tree_node("Joint Position"):
            ## Slider about joint position 
            joint_lower_limit = self.env.skel.getPositionLowerLimits()
            joint_upper_limit = self.env.skel.getPositionUpperLimits()
            joint_pos = self.env.skel.getPositions()

            for i in range(6, len(joint_pos)):
                _, joint_pos[i] = imgui.slider_float("Joint %d" % i, joint_pos[i], joint_lower_limit[i], joint_upper_limit[i])

            self.env.skel.setPositions(joint_pos)
            imgui.tree_pop()

        if imgui.tree_node("Motion Information"):
            ## Current Loaded motion
            cur_bvh = self.env.get_current_bvh()

            _, self.env.bvh_idx = imgui.listbox(
                label = "##BVH List", 
                current = self.env.bvh_idx,
                items = self.env.bvh_names,
                height_in_items = 10,
            )

            ## Information about motion 
            imgui.text("Current BVH : %s" % cur_bvh.bvh_file)
            imgui.text("Current Frame : %d" % int(self.env.world.getTime() // cur_bvh.frame_time))
            imgui.text("Total Frames : %d" % int(self.env.bvh_total_num_frames))
            ## Slider about motion
            if self.env.get_current_bvh().num_frames > 0:
                t = self.env.world.getTime() // cur_bvh.bvh_time
                _, cur_frame = imgui.slider_int("Frame", (self.env.world.getTime() - t * cur_bvh.bvh_time) // cur_bvh.frame_time, 0, cur_bvh.num_frames - 1)
                self.env.world.setTime(t * cur_bvh.bvh_time + cur_frame * cur_bvh.frame_time)
                if not self.is_simulation:
                    self.env.update_target(t * cur_bvh.bvh_time + cur_frame * cur_bvh.frame_time)

            if imgui.button("Reset"):
                self.reset(self.env.world.getTime())

            # All motion files list box 
            _, self.motion_idx = imgui.listbox(
                label = "##Motion List",
                current = self.motion_idx,
                items = self.motion_list,
                height_in_items = 5
            )
            
            if imgui.button("Add"):
                self.env.add_bvh(self.motion_list[self.motion_idx])
            # Same line 
            imgui.same_line()

            if imgui.button("Add(FirstT)"):
                self.env.add_bvh(self.motion_list[self.motion_idx], 1)

            imgui.tree_pop()

        # if self.env.muscles and imgui.tree_node("Muscle Accuracy"):
        #     if len(self.muscle_accuracy) > 0:
        #         imgui.plot_lines(
        #             label="Muscle Accuracy",
        #             values=np.array(self.muscle_accuracy, dtype=np.float32),
        #             overlay_text="Muscle Accuracy",
        #             scale_min=0.0,
        #             scale_max=200.0,
        #             graph_size=(imgui.get_content_region_available_width(), 200)
        #         )    
        #     imgui.tree_pop()

        if self.nn != None and self.nn.policy.config["VAE"]:
            if imgui.tree_node("VAVQE"):

                imgui.plot_histogram(
                    label="##vae",
                    values=self.vae_codebook_frequency.astype(np.float32),
                    values_count=self.nn.policy.p_fc.num_embeddings,
                    scale_min=0,
                    scale_max=100,
                    graph_size=(imgui.get_content_region_available_width(), 200)
                )
                ## Button 
                _, self.vae_random_sampling = imgui.checkbox("VAE Random Sampling", self.vae_random_sampling)
        
                imgui.tree_pop()
        
        if imgui.tree_node("Loaded Network"):
            ## checkbox 
            if self.env.spd_nn:
                _, self.env.use_spd_nn = imgui.checkbox("Use Spd NN", self.env.use_spd_nn)            
            imgui.tree_pop()
      


        ## Muscle Information 
        if self.env.muscles and imgui.tree_node("Muscle Acitvation"):
            # Histgram of muscle activation
            imgui.plot_histogram(
                label="##muscle_activation",
                values=self.env.muscle_activation_levels.astype(np.float32),
                values_count=len(self.env.muscle_activation_levels),
                scale_min=0.0,
                scale_max=1.0,
                graph_size=(imgui.get_content_region_available_width(), 200)
            )
            imgui.tree_pop()
        imgui.end()
        imgui.render()

    def save_current_frame(self, name, include_UI = False):
        self.drawSimFrame()   
        if include_UI:
            self.drawUIFrame()
        ## Save Current Screen
        glReadBuffer(GL_FRONT)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (self.width, self.height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save("screenshots/%s.png" % name) 


    def reset(self, time = None):
        if self.nn != None and self.nn.policy.config["VAE"]:
            self.vae_codebook_frequency = np.zeros(self.nn.policy.p_fc.num_embeddings, dtype=np.int32)
        self.env.reset(time)
        self.reward_buffer = [self.env.get_reward(True)]
        if self.nn:
            self.value_buffer = [self.nn.get_value(self.env.get_obs())[0]]
        self.sim_count = 0

        self.muscle_accuracy = []
    def keyboardPress(self, key, scancode, action, mods):
        
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window, True)
            elif key == glfw.KEY_SPACE:
                self.is_simulation = not self.is_simulation
            elif key == glfw.KEY_S:
                self.update()
            elif key == glfw.KEY_R:
                self.reset()
            elif key == glfw.KEY_F:
                self.focus += 1
                self.focus = self.focus % 3
            elif key == glfw.KEY_0:
                self.is_recording = not self.is_recording
                if self.is_recording:
                    self.record_idx = 0 

            elif key == glfw.KEY_Q:
                glfw.set_window_should_close(self.window, True)

        pass

    def startLoop(self):        
        plt.title("CodeBook") 
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            
            self.impl.process_inputs()
            if self.is_simulation:
                self.update()
                if self.nn and self.draw_vae_plot and self.nn.policy.isVAE and self.nn.policy.config["VAE"] == "vq":
                    self.scatter_colors[self.nn.policy.p_fc.quantizer.idx] = np.array([1.0, 0.0, 0.0, 1.0])
                    if self.sim_count % 10 == 0:
                        self.scatter.set_facecolors(self.scatter_colors)
                        plt.show()
                        plt.pause(1E-6)
                        self.scatter_colors *= 0
                        self.scatter_colors += 0.5
            ## Rendering Simulation
            if self.is_recording:
                self.save_current_frame("frame_%04d" % self.record_idx)
                self.record_idx += 1
            else:
                self.drawSimFrame()            
                self.drawUIFrame()

            self.impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)
        self.impl.shutdown()
        glfw.terminate()
        return

