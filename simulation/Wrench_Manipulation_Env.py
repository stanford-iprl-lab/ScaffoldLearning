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

from math import sin,cos,acos

import robot_Wrench as robot
from matplotlib import pyplot as plt
import scipy.spatial.distance 
import scipy.ndimage
from scipy import ndimage

def check_outside_tray(obj_pos, tray_bbox):
  diff = tray_bbox - obj_pos
  sign = np.sign(diff[0,:] * diff[1, :])[:2]
  return np.any(sign > 0) 


class RobotEnv():
  def __init__(self,
               worker_id,
               p_id,
               actionRepeat=80,
               isEnableSelfCollision=True,
               renders=False,
               maxSteps=20,
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
    self._maxSteps = 20#maxSteps
    self._isDiscrete = False
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180 
    self._cam_pitch = -40
    self._dv = dv
    self.p = p_id
    self.delta_t = dt
    self.p.setTimeStep(self.delta_t)
    self.fixture_offset = fixture_offset

    self.p.setPhysicsEngineParameter(enableConeFriction=1)
    self.p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
    self.p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)
    self.p.setPhysicsEngineParameter(numSolverIterations=40)
    self.p.setPhysicsEngineParameter(numSubSteps=40)
    self.p.setPhysicsEngineParameter(constraintSolverType=self.p.CONSTRAINT_SOLVER_LCP_DANTZIG,globalCFM=0.000001)
    self.p.setPhysicsEngineParameter(enableFileCaching=0)

    self.p.setTimeStep(1 / 100.0)
    self.p.setGravity(0,0,-9.81)

    self._blockRandom = blockRandom
    self._cameraRandom = cameraRandom
    self._width = width
    self._height = height
    self._isTest = isTest
    self._wid = worker_id
    self.termination_flag = False
    self.success_flag = False

    self.start_pos = start_pos
    self.robot = robot

    self.resource_dir = "../resource"
    self.initPose_dir = os.path.join(self.resource_dir,"initPose","wrenchPose.npy")
    self.initGripper_dir = os.path.join(self.resource_dir,"initPose","wrenchGripper.npy")
    self.texture_dir = os.path.join(self.resource_dir,"texture")
    self.cameraPose_dir = os.path.join(self.resource_dir,"cameraPose")

    qlist = np.load(self.initPose_dir)
    self.q_null = qlist[-1]
  
    self.urdf_dir = os.path.join(self.resource_dir,"urdf")
    self.p.loadURDF(os.path.join(self.urdf_dir,"plane.urdf"),[0,0,0])
    self.env_textid = self.p.loadTexture(os.path.join(self.texture_dir,"texture1.jpg"))
    self.p.changeVisualShape(0,-1,textureUniqueId=self.env_textid)

    self._env_step = 0
    #### table initialization    
    self.table_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[0.3,0.5,0.15])
    self.table_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[0.3,0.5,0.15])
    mass = 0
    self.table_id = self.p.createMultiBody(mass,baseCollisionShapeIndex=self.table_c,baseVisualShapeIndex=self.table_v,basePosition=(0.5,0.0,0.2))
    self.table_color = [128/255.0,128/255.0,128/255.0,1.0]
    self.p.changeVisualShape(self.table_id,-1,rgbaColor=self.table_color)

    ####  robot initialization
    self.robot = robot.Robot(pybullet_api=self.p,urdf_path=self.urdf_dir)
    self.red_color = [0.9254901, 0.243137, 0.086274509,1.0]
    self.blue_color = [0.12156, 0.3804, 0.745, 1.0]
    self.yellow_color = [0.949, 0.878, 0.0392, 1.0]

    self.init_obj()
    self.reset()
  

  def init_obj(self):

    table_z = self.p.getAABB(self.table_id)[1][2]
 
    self.obj_position = [0.44+0.038, -0.07+0.083, table_z+0.04]
 
    self.obj_position[0] = 0.43
    self.obj_position[0] += np.random.uniform(low=-0.02,high=0.02)
    self.obj_position[1] += np.random.uniform(low=-0.02,high=0.02)
    self.obj_position[2] -= 0.04
 
    self.obj_x = self.obj_position[0]
    self.obj_y = self.obj_position[1]
    self.obj_z = self.obj_position[2]

    self.obj_orientation = [0,0,0,1.0]#[0, 0, -0.1494381, 0.9887711]
    self.obj_scaling = 1.0
    self.obj_id = self.p.loadURDF( os.path.join(self.urdf_dir, "obj_libs/bottles/b7/b7.urdf"),basePosition=self.obj_position,baseOrientation=self.obj_orientation,globalScaling=self.obj_scaling,useFixedBase=True)
    self.p.changeVisualShape( self.obj_id, -1, rgbaColor=self.blue_color,specularColor=[1.,1.,1.])
    self.p.changeVisualShape( self.obj_id, 0, rgbaColor=self.yellow_color,specularColor=[1.,1.,1.])
    self.use_fixture = True

    self.p.resetJointState(self.obj_id,0,targetValue=-0.1,targetVelocity=0.0)

    self.bolt_z = self.p.getAABB(self.obj_id,-1)[1][2]

    obj_friction_ceof = 0.5
    self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
    self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
    self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
    self.p.changeDynamics(self.obj_id, -1, linearDamping=0.1)
    self.p.changeDynamics(self.obj_id, -1, angularDamping=0.1)
    self.p.changeDynamics(self.obj_id, -1, contactStiffness=300.0, contactDamping=0.1)

    table_friction_ceof = 0.4
    self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
    self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
    self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
    self.p.changeDynamics(self.table_id, -1, contactStiffness=1.0, contactDamping=0.9)


  def obj_reset(self):
    self.p.resetJointState(self.obj_id,0,targetValue=-0.1,targetVelocity=0.0)

    table_z = self.p.getAABB(self.table_id)[1][2]
    self.obj_position = [0.44+0.038, -0.07+0.083, table_z+0.04]
    self.obj_position[0] = 0.42
    self.obj_position[1] = -0.03
    self.obj_position[0] += np.random.uniform(low=-0.04,high=0.04)
    self.obj_position[1] += np.random.uniform(low=-0.04,high=0.04)
    self.obj_position[2] -= 0.04
 
    self.obj_x = self.obj_position[0]
    self.obj_y = self.obj_position[1]
    self.obj_z = self.obj_position[2]

    self.obj_orientation = [0,0,0,1.0]#[0, 0, -0.1494381, 0.9887711]
    self.obj_scaling = 1.0 #/ 6.26 * 3.1

    self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)
    self.bolt_z = self.p.getAABB(self.obj_id,-1)[1][2]


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

    qlist = np.load(self.initPose_dir)
    glist = np.load(self.initGripper_dir)
    num_q = len(qlist[0])

    self.null_q = [1.5320041040300532, -1.2410604956227453, -1.338379970868218, -2.301559526826164, 0.23437008617841384, 1.8328313603162587, 1.5954970526882803]
    self.robot.setJointValue(self.null_q,210)

    target_pos = [self.obj_x + 0.013, self.obj_y - 0.006, self.bolt_z - 0.001]
    target_orn = [0.07791876168003176, -0.041181656673171036, 0.9967247218368238, 0.004453411965720604]

    predict_pose = self.robot.IK_wrench(target_pos,target_orn,self.null_q) 
    self._env_step = 0 
    self.robot.setJointValue(predict_pose,210)

    self.p.resetJointState(self.obj_id,0,0.15,0.0)

    for i in range(10):
      self.robot.wrench_Control(target_pos,target_orn,self.null_q,210)
    
    for i in range(20):
      self.p.stepSimulation()

    initial_q_list = self.robot.getJointValue()
    initial_q_list[-1] -= 0.1
    self.robot.setJointValue(initial_q_list,210)

    wrench_orn = self.robot.getWrenchLeftTipOrn()
    wrench_euler = self.p.getEulerFromQuaternion(wrench_orn)

    self.start_orn = self.p.getJointState(self.obj_id,0)[0]
    cur_pos = self.robot.getWrenchTipPos()
    target_pos = self.p.getLinkState(self.obj_id,0)[0]
    self.prev_dist = np.linalg.norm(target_pos - cur_pos)
    self.prev_orn = np.copy(self.start_orn)
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

  def angleaxis2quaternion(self,angleaxis):
    angle = np.linalg.norm(angleaxis)
    axis = angleaxis / (angle + 0.00001)
    q0 = cos(angle/2)
    qx,qy,qz = axis * sin(angle/2) 
    return np.array([qx,qy,qz,q0])

  def quaternion2angleaxis(self,quater):
    angle = 2 * acos(quater[3])
    axis = quater[:3]/(sin(angle/2)+0.00001)
    angleaxis = axis * angle
    return np.array(angleaxis)

  def step(self, action):
    next_pos = np.array(self.robot.getEndEffectorPos()) + np.array(action)[:3]
    next_cur = np.array(self.robot.getEndEffectorOrn())
    next_cur = np.array(self.p.getEulerFromQuaternion(self.robot.getEndEffectorOrn()))
    next_cur[0] += action[3]
    next_cur[1] += action[4]
    next_cur[2] += action[5]
    orn_next = self.p.getQuaternionFromEuler(next_cur)
    for _ in range(4):
        self.robot.operationSpacePositionControl(next_pos,orn=orn_next,null_pose=None,gripperPos=220)
    observation, seg = self._get_observation_imgseg()
    reward,done,suc = self._reward()
    return observation, reward, done, suc


  def _reward(self):
    self.termination_flag = False
    self.success_flag = False

    reward = 0
    cur_pos_L = self.robot.getWrenchLeftTipPos()    
    cur_pos_L[2] = self.p.getAABB(self.robot.robotId, self.robot.wrench_left_tip_index)[0][2] 

    target_pos = np.array(self.p.getLinkState(self.obj_id,0)[0])
    dist_L = np.linalg.norm(target_pos - cur_pos_L)

    cur_pos_R = self.robot.getWrenchRightTipPos()    
    cur_pos_R[2] = self.p.getAABB(self.robot.robotId, self.robot.wrench_right_tip_index)[0][2] 
    dist_R = np.linalg.norm(target_pos - cur_pos_R)
    
    dist = 0.5 * dist_L + 0.5 * dist_R
    cur_orn = self.p.getJointState(self.obj_id,0)[0]
    reward_orn = self.prev_orn - cur_orn
    self.prev_orn = cur_orn

    next_cur = np.array(self.p.getEulerFromQuaternion(self.robot.getWrenchLeftTipOrn()))
    
    reward = reward_orn * 100.0
    self._env_step += 1

    if self.start_orn - cur_orn > 30/180.0*math.pi:
      self.termination_flag = True
      self.success_flag = True
      reward = 5.0
    if dist > 0.04:
      self.termination_flag = True
      self.success_flag = False
      reward = -1.0
    if self._env_step >= self._maxSteps:
      self.termination_flag = True
      self._env_step = 0
    return reward, self.termination_flag, self.success_flag
