import random
import os
import time
import sys
import numpy as np
import pdb
import distutils.dir_util
import glob
from pkg_resources import parse_version
import gym
import pickle
import math
import cv2

import robot_Shallow as robot
from matplotlib import pyplot as plt
import scipy.spatial.distance 
import scipy.ndimage
from scipy import ndimage

def check_outside_tray(obj_pos, tray_bbox):
  diff = tray_bbox - obj_pos
  sign = np.sign(diff[0,:] * diff[1, :])[:2]
  return np.any(sign > 0) 

HOME_DIR = "/juno/u/lins2"

class RobotEnv():
  """Class for Kuka environment with diverse objects.

  In each episode some objects are chosen from a set of 1000 diverse objects.
  These 1000 objects are split 90/10 into a train and test set.
  """
  def __init__(self,
               worker_id,
               p_id,
               actionRepeat=80,
               isEnableSelfCollision=True,
               renders=False,
               maxSteps=80,
               dv=0.01,
               dt=0.001,
               blockRandom=0.01,
               cameraRandom=0,
               width=640,
               height=480,
               start_pos = [0.5, 0.3, 0.5],
               fixture_offset=np.zeros((3,)),
               isTest=False,
               is3D=False):
    self._timeStep = 1./240.
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = 50#maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180 
    self._cam_pitch = -40
    self._dv = dv
    self.p = p_id
    self.delta_t = dt
    self.fixture_offset = fixture_offset

    self.p.setTimeStep(self.delta_t)

    self.p.setPhysicsEngineParameter(enableConeFriction=1)
    self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
    self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

    self.p.setPhysicsEngineParameter(numSolverIterations=20)
    self.p.setPhysicsEngineParameter(numSubSteps=10)

    self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
    self.p.setPhysicsEngineParameter(enableFileCaching=0)

    self.p.setTimeStep(1 / 30.0)
    self.p.setGravity(0,0,-9.81)

    self._blockRandom = blockRandom
    self._cameraRandom = cameraRandom
    self._width = width
    self._height = height
    self._isTest = isTest
    self._wid = worker_id
    self._gripper_left_tip_index = 13
    self._gripper_right_tip_index = 17
    self.termination_flag = False
    self.success_flag = False

    self.start_pos = start_pos

    self.resource_dir = "../resource"
    self.initPose_dir = os.path.join(self.resource_dir,"initPose","wrenchPose.npy")
    self.initGripper_dir = os.path.join(self.resource_dir,"initPose","wrenchGripper.npy")
    self.texture_dir = os.path.join(self.resource_dir,"texture")
    self.cameraPose_dir = os.path.join(self.resource_dir,"cameraPose")

    self.robot = robot
    qlist = np.load(self.initPose_dir)
    self.q_null = qlist[-1]
  
    self._env_step = 0
    self.urdf_root = os.path.join(HOME_DIR,"EnvironmentLeverage/simulation")

    self.urdf_dir = os.path.join(self.resource_dir,"urdf")
    self.p.loadURDF(os.path.join(self.urdf_dir,"plane.urdf"),[0,0,0])
    self.env_textid = self.p.loadTexture(os.path.join(self.texture_dir,"texture1.jpg"))
    self.p.changeVisualShape(0,-1,textureUniqueId=self.env_textid)

    #### robot initialization
    self.robot = robot.Robot(pybullet_api=self.p,urdf_path=self.urdf_dir)

    #### table initialization
    self.table_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[0.3,0.2,0.15])
    self.table_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[0.3,0.2,0.15])
    mass = 0
    self.table_id = self.p.createMultiBody(mass,baseCollisionShapeIndex=self.table_c,baseVisualShapeIndex=self.table_v,basePosition=(0.5,0.0,0.15))
    self.table_color = [128/255.0,128/255.0,128/255.0,1.0]
    self.p.changeVisualShape(self.table_id,-1,rgbaColor=self.table_color)

    self.red_color = [0.9254901, 0.243137, 0.086274509,1.0]
    self.blue_color = [0.12156, 0.3804, 0.745, 1.0]
    self.yellow_color = [0.949, 0.878, 0.0392, 1.0]

    self.init_obj()
      

  def init_obj(self):
    self.table_z = self.p.getAABB(self.table_id)[1][2]
    objb_w = 0.08/2
    objb_h = 0.12/2
    objb_d = 0.03

    self.box_w = 0.00 + np.random.uniform(low=-0.06,high=0.06)
    self.box_h = 0.33 + np.random.uniform(low=-0.06,high=0.06)
    self.box_d = self.table_z + objb_d * 2

    self.box_center_pos = [self.box_h, self.box_w, self.box_d]

    self.box_y = self.box_w
    self.box_x = self.box_h
    self.box_z = self.box_d

    ########## peg
    self.peg_w = 0.04
    self.peg_h = 0.05
    self.peg_d = 0.005
    self.obj_position = [self.box_h, self.box_w-0.025,self.box_d+self.peg_h]
    self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2,0.0,0])#math.pi/2])
    self.obj_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[self.peg_w, self.peg_h, self.peg_d])
    self.obj_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[self.peg_w, self.peg_h, self.peg_d])
    self.obj_id = self.p.createMultiBody(0.01, self.obj_c, self.obj_v, self.obj_position)
    self.p.changeVisualShape (self.obj_id, -1, rgbaColor=self.yellow_color,specularColor=[1.,1.,1.])

    AABB= self.p.getAABB(self.obj_id)

    self.hole_w = 0.05
    self.hole_h = 0.04
    mass = 0

    ####### base
    self.obj1_position = [self.box_x, self.box_y, self.table_z + objb_d]
    self.obj1_orientation = [0.0, 0.0, 0.0, 1.0]
    self.obj1_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[objb_w, objb_h, objb_d])
    self.obj1_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[objb_w, objb_h, objb_d])
    self.obj1_id = self.p.createMultiBody(mass, self.obj1_c, self.obj1_v, self.obj1_position)
    self.p.changeVisualShape (self.obj1_id, -1, rgbaColor=self.blue_color,specularColor=[1.,1.,1.])

    #### top base
    obj3_w = 0.01
    obj3_h = 0.04
    obj3_d = 0.007
    self.obj3_position = [self.box_h, self.box_w - objb_h + obj3_w , self.box_d + obj3_d - 0.01]
    self.obj3_orientation = [0.0, 0.0, 0.0, 1.0]
    self.obj3_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[obj3_h, obj3_w, obj3_d])
    self.obj3_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[obj3_h, obj3_w, obj3_d])
    self.obj3_id = self.p.createMultiBody(mass, self.obj3_c, self.obj3_v, self.obj3_position)
    self.p.changeVisualShape (self.obj3_id, -1, rgbaColor=self.blue_color,specularColor=[1.,1.,1.])

    self.box_AABB = self.p.getAABB(self.obj1_id)#[self.box_h - self.hole_h - 1 * obj3_h, self.box_w - self.hole_w - 1 * obj3_w, self.box_d, self.box_h + self.hole_h + 1 * obj3_h, self.box_w + self.hole_w + 1 * obj3_w, obj1_d + self.box_d]


  def obj_reset(self):
    self.table_z = self.p.getAABB(self.table_id)[1][2]
    objb_w = 0.08/2
    objb_h = 0.12/2
    objb_d = 0.03

    self.box_w = 0.00 + np.random.uniform(low=-0.04,high=0.04)
    self.box_h = 0.33 + np.random.uniform(low=-0.04,high=0.04)
    self.box_d = self.table_z + objb_d * 2

    self.box_center_pos = [self.box_h, self.box_w, self.box_d]

    self.box_y = self.box_w
    self.box_x = self.box_h
    self.box_z = self.box_d

    ########## peg
    self.peg_w = 0.04
    self.peg_h = 0.05
    self.peg_d = 0.005
    self.obj_position = [self.box_h, self.box_w-0.025,self.box_d+self.peg_h]
    self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2,0.0,0])#math.pi/2])
    self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)

    AABB= self.p.getAABB(self.obj_id)

    self.hole_w = 0.05
    self.hole_h = 0.04
    mass = 0

    ####### base
    self.obj1_position = [self.box_x, self.box_y, self.table_z + objb_d]
    self.obj1_orientation = [0.0, 0.0, 0.0, 1.0]
    self.p.resetBasePositionAndOrientation (self.obj1_id, self.obj1_position, self.obj1_orientation)

    #### top base
    obj3_w = 0.01
    obj3_h = 0.04
    obj3_d = 0.007
    self.obj3_position = [self.box_h, self.box_w - objb_h + obj3_w , self.box_d + obj3_d]
    self.obj3_orientation = [0.0, 0.0, 0.0, 1.0]
    self.p.resetBasePositionAndOrientation (self.obj3_id, self.obj3_position, self.obj3_orientation)


    self.box_AABB = self.p.getAABB(self.obj1_id)#[self.box_h - self.hole_h - 1 * obj3_h, self.box_w - self.hole_w - 1 * obj3_w, self.box_d, self.box_h + self.hole_h + 1 * obj3_h, self.box_w + self.hole_w + 1 * obj3_w, obj1_d + self.box_d]


    self.p.changeDynamics(self.obj_id,-1,mass=0.8)
    self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)
    self.p.changeDynamics(self.obj_id, -1, contactStiffness=1.0, contactDamping=0.1)

    obj_friction_ceof = 1.0
    self.p.changeDynamics(self.obj3_id, -1, lateralFriction=obj_friction_ceof)
    self.p.changeDynamics(self.obj3_id, -1, rollingFriction=obj_friction_ceof)
    self.p.changeDynamics(self.obj3_id, -1, spinningFriction=obj_friction_ceof)

    self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
    self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
    self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)

    self.p.changeDynamics(self.obj1_id, -1, lateralFriction=obj_friction_ceof)
    self.p.changeDynamics(self.obj1_id, -1, rollingFriction=obj_friction_ceof)
    self.p.changeDynamics(self.obj1_id, -1, spinningFriction=obj_friction_ceof)
 
  def reset(self):
    """Environment reset called at the beginning of an episode.
    """
    self.robot.reset()
    self.obj_reset()

    # Set the camera settings.
    viewMatrix = np.loadtxt(os.path.join(self.cameraPose_dir,"handeye.txt"))
    cameraEyePosition = viewMatrix[:3,3]  
    cameraUpVector = viewMatrix[:3,1] * -1.0
    cameraTargetPosition = viewMatrix[:3,3] + viewMatrix[:3,2] * 0.001
 
    self._view_matrix = self.p.computeViewMatrix(cameraEyePosition,cameraTargetPosition,cameraUpVector)
    self._view_matrix_np = np.eye(4)
    self._view_matrix_np = np.array(self._view_matrix)
    self._view_matrix_np = self._view_matrix_np.reshape((4,4)).T
    self._view_matrix_inv = np.linalg.inv(self._view_matrix_np)

    self.cameraMatrix = np.load(os.path.join(self.cameraPose_dir,"cameraExPar.npy"))
    fov =  2 * math.atan(self._height / (2 * self.cameraMatrix[1,1])) / math.pi * 180.0
    self.fov = fov
    aspect = float(self._width) / float(self._height)
    near = 0.02
    far = 4
    self._proj_matrix = self.p.computeProjectionMatrixFOV(fov, aspect, near, far)
    self.far = far
    self.near = near
   
    self._proj_matrix = np.array(self._proj_matrix)
 
    self._attempted_grasp = False
    self._env_step = 0
    self.terminated = 0

    ########################
    self._envStepCounter = 0

    # Compute xyz point cloud from depth
    nx, ny = (self._width, self._height)
    x_index =  np.linspace(0,nx-1,nx)
    y_index =  np.linspace(0,ny-1,ny)
    self.xx, self.yy = np.meshgrid(x_index, y_index)
    self.xx -= float(nx)/2
    self.yy -= float(ny)/2

    self._camera_fx = self._width/2.0 / np.tan(fov/2.0 / 180.0 * np.pi)
    self._camera_fy = self._height/2.0 / np.tan(fov/2.0 / 180.0 * np.pi)
    self.xx /= self._camera_fx
    self.yy /= self._camera_fy
    self.xx *= -1.0

    initial_pos =[-1.3933841779817748, -1.0244997927415005, 0.673529227525627, -2.066219411895667, 0.6026837034499274, 1.2232697040626301, -0.7701732907076426] #[-1.365082966465024, -1.0944634372663833, 0.7148632733641577, -2.2154123896834905, 0.6478194523384135, 1.3068242585044993, -0.7913471829487223]
    self.null_q = initial_pos

    orn = self.p.getQuaternionFromEuler([math.pi,0.0,0.0])
    self.fix_orn = orn
  
    gv = 100 
    self.robot.setJointValue(initial_pos,gv)

    pos = np.copy(self.obj_position)#[0.30,0.01,0.533]
    pos[2] += 0.18 

    for i in range(19):
      self.robot.operationSpacePositionControl(pos,orn,null_pose=self.null_q,gripperPos=gv)

    self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)

    #input("raw closing 100 to 240 gripp")
    gv = 240 
    for i in range(19):
      self.robot.operationSpacePositionControl(pos,orn,null_pose=self.null_q,gripperPos=gv)
    self._env_step = 0 

    #input("changin")
    current_pos = np.array(self.robot.getEndEffectorPos())
    current_pos[1] += 0.06
    current_pos[2] -= 0.01
    offset = 0.4
    gv = 240
    current_orn = self.p.getQuaternionFromEuler([math.pi - offset, 0.0, 0.0])
    for i in range(20):
      self.robot.operationSpacePositionControl(current_pos,orn=current_orn,null_pose=self.q_null,gripperPos=gv)


    cur_center = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
    cur_orn = self.p.getEulerFromQuaternion(self.p.getBasePositionAndOrientation(self.obj_id)[1])[1]
    target_orn = math.pi/2.0#np.array(self.p.getQuaternionFromEuler([0,math.pi/2.0,math.pi/2.0]))
    target_center = np.array(self.box_center_pos)

    dist_orn = np.linalg.norm(target_orn - cur_orn)
    dist_cen = np.linalg.norm(cur_center - target_center)

    self.prevOrn = dist_orn
    self.prevPos = dist_cen
    #print("reset is finished")
    return self._get_observation()


  def _get_observation(self):
    """Return the observation as an image.
    """
    img_arr = self.p.getCameraImage(width=self._width + 20,
                                      height=self._height + 10,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix,
                                      shadow=0, lightAmbientCoeff=0.6,lightDistance=100,lightColor=[1,1,1],lightDiffuseCoeff=0.4,lightSpecularCoeff=0.1,renderer=self.p.ER_TINY_RENDERER
)

    rgb = img_arr[2][:-10,20:,:3]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 3))
    np_img_arr = cv2.resize(np_img_arr,dsize=(160,120),interpolation=cv2.INTER_CUBIC)
    return np_img_arr


  def _get_observation_img(self):
    """Return the observation as an image.
    """
    img_arr = self.p.getCameraImage(width=self._width + 20,
                                      height=self._height + 10,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix,
                                      shadow=0, lightAmbientCoeff=0.6,lightDistance=100,lightColor=[1,1,1],lightDiffuseCoeff=0.4,lightSpecularCoeff=0.1,renderer=self.p.ER_TINY_RENDERER
)

    rgb = img_arr[2][:-10,20:,:3]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 3))
    return np_img_arr



  def _get_observation_imgseg(self):
    """Return the observation as an image.
    """
    img_arr = self.p.getCameraImage(width=self._width + 20,
                                      height=self._height + 10,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix,
                                      shadow=0, lightAmbientCoeff=0.6,lightDistance=100,lightColor=[1,1,1],lightDiffuseCoeff=0.4,lightSpecularCoeff=0.1,renderer=self.p.ER_TINY_RENDERER
)

    rgb = img_arr[2][:-10,20:,:3]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 3))
    np_img_arr = cv2.resize(np_img_arr,dsize=(160,120),interpolation=cv2.INTER_CUBIC)
    seg = img_arr[4][:-10,20:]
    return np_img_arr, seg

  def _get_imgseg(self):

    img_arr = self.p.getCameraImage(width=self._width + 20,
                                      height=self._height + 10,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix,
                                      shadow=-1,
                                      flags=self.p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                      renderer=self.p.ER_TINY_RENDERER)
    rgb = img_arr[2][:-10,20:,:3]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 3))
    np_img_arr = cv2.resize(np_img_arr,dsize=(160,120),interpolation=cv2.INTER_CUBIC)
    seg = img_arr[4][:-10,20:]
    return np_img_arr, seg 


  def step(self, action):
    current_pos = np.array(self.robot.getEndEffectorPos())
    current_pos[1] += action[0]
    current_pos[2] += action[1]
    current_pos[0] = self.box_center_pos[0]
    euler_orn = self.p.getEulerFromQuaternion(self.robot.getEndEffectorOrn())
    offset = euler_orn[0]
    offset += action[2] * 10
    gv = 240#(action[3] + 0.03)/0.06*255.0 
    if offset < math.pi/2.0 - math.pi/18.0:
      offset = math.pi/2.0 - math.pi/18.0
    current_orn = self.p.getQuaternionFromEuler([offset, 0.0, 0.0])
    for i in range(5):
      self.robot.operationSpacePositionControl(current_pos,orn=current_orn,null_pose=self.q_null,gripperPos=gv)
    
    observation, seg = self._get_imgseg()
    reward, done, suc = self._reward(seg)
    return observation, reward, done, suc

  def _reward(self,seg):
    reward = 0
    self._graspSuccess = 0
    
    cur_center = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
    cur_orn = self.p.getEulerFromQuaternion(self.p.getBasePositionAndOrientation(self.obj_id)[1])[0]
    target_orn = 0.0
    target_center = np.array(self.box_center_pos)

    dist_orn = np.linalg.norm(target_orn - cur_orn)
    dist_cen = np.linalg.norm(cur_center[2] - target_center[2])

    reward_orn = 1.2 - cur_orn
    self.prevPos = dist_cen
    AABB_obj = self.p.getAABB(self.obj_id)

    dist_y = abs(self.p.getAABB(self.obj3_id)[1][1] - AABB_obj[0][1])
    dist_z = abs(self.p.getAABB(self.obj3_id)[0][2] - AABB_obj[0][2])
    dist_x = abs(self.p.getAABB(self.obj3_id)[0][0] - AABB_obj[0][0])
 
    target_head = np.array([self.p.getAABB(self.obj3_id)[1][1],self.p.getAABB(self.obj1_id)[1][2]])
    current_head = np.array([AABB_obj[0][1],AABB_obj[0][2]])
    dist_target_cur_head = np.linalg.norm(target_head-current_head)
    dist_target_cur = np.exp(-dist_target_cur_head*100.0) * 0.1
    reward = reward_orn + dist_target_cur 

    self.termination_flag = False
    self.success_flag = False

    if cur_orn < math.pi/36.0 and dist_z < 0.01 and dist_y < 0.005:# - 0.3:
      reward = 5.0
      self.success_flag = True
      self.termination_flag = True

    self._env_step += 1
    if self._env_step >= self._maxSteps:
      self.termination_flag = True
      self._env_step = 0

    if dist_target_cur_head > 0.04:
      reward = -1.
      self.termination_flag = True
      self.success_flag = False

    current_pos = cur_center
    AABB_top_base = self.p.getAABB(self.obj3_id)
    return reward, self.termination_flag, self.success_flag
