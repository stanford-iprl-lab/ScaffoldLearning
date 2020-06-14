import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import math, os
import cv2
import torchvision.transforms as transforms

import imageio

os.environ["OMP_NUM_THREADS"] = "1"
device=torch.device("cuda")

np.set_printoptions(precision=4,suppress=True)
simulation_dir = '../simulation'
sys.path.insert(0, simulation_dir)

from Wrench_Manipulation_Env import RobotEnv
ExName = "plugging"
sys.path.insert(0,'/juno/u/lins2/bullet3/build_cmake/examples/pybullet')


def v_wrap(np_array,dtype=np.float32):
  if np_array.dtype != dtype:
    np_array = np_array.astype(dtype)
  return torch.from_numpy(np_array).to(device)

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, bdone, gamma):
  if done:
    v_s_ = 0.
  else:
    v_s_ = lnet.forward(v_wrap(s_[None,:]))[-1].data.cpu().numpy()[0,0]

  buffer_v_target = []
  for r, termination in zip(br[::-1], bdone[::-1]):
    if termination:
      v_s_ = 0
    v_s_ = r + gamma * v_s_
    buffer_v_target.append(v_s_)
  buffer_v_target.reverse()

  loss = lnet.loss_func(
    v_wrap(np.vstack(bs)),
    v_wrap(np.vstack(ba)),
    v_wrap(np.array(buffer_v_target)[:, None]))

  opt.zero_grad()
  loss.backward()
  
  nn.utils.clip_grad_norm(lnet.parameters(),1.0)
  for lp, gp in zip(lnet.parameters(), gnet.parameters()):
    gp._grad = lp.grad
  opt.step()

  # pull global parameters
  lnet.load_state_dict(gnet.state_dict())

import pybullet

MAX_EP = 15000
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.01)
        nn.init.constant_(layer.bias, 0.)


class ACNet(nn.Module):
  def __init__(self):
    super(ACNet, self).__init__()
    self.distribution = torch.distributions.Normal
    self.block1 = nn.Sequential(
      nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=True),
      nn.ReLU(),
      nn.BatchNorm2d(32),
    )
    # 60, 80
    self.block2 = nn.Sequential(
      nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=True),
      nn.ReLU(),
      nn.BatchNorm2d(32),
    )
    # 30, 40
    self.block3 = nn.Sequential(
      nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=True),
      nn.ReLU(),
      nn.BatchNorm2d(64),
    )
    # 15, 20
    self.block4 = nn.Sequential(
      nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=True),
      nn.ReLU(),
      nn.BatchNorm2d(64),
    )
    # 8, 10
    self.block5 = nn.Sequential(
      nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=True),
      nn.ReLU(),
      nn.BatchNorm2d(128),
    )
    # 4, 5
    self.block6 = nn.Sequential(
      nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=True),
      nn.ReLU(),
      nn.BatchNorm2d(128),
    )
    # 2, 3
    self.fc_a = nn.Sequential(
          nn.Linear(2 * 3 * 128, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 24),
          nn.ReLU()
    )

    self.fc_s = nn.Sequential(
          nn.Linear(2 * 3 * 128, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 24),
          nn.ReLU()
    )


    self.fc_v = nn.Sequential(
          nn.Linear(2 * 3 * 128, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 24),
          nn.ReLU()
    )


    self.mu_layer = nn.Linear(24,6)
    self.sigma_layer = nn.Linear(24,6)
    self.v_layer = nn.Linear(24,1)
  
    set_init([self.mu_layer, self.sigma_layer, self.v_layer]) 

  def forward(self, im):
    im = im.view(-1, 120, 160, 3)
    im = im.permute(0,3,1,2)
    im = self.block1(im)
    im = self.block2(im)
    im = self.block3(im)
    im = self.block4(im)
    im = self.block5(im)
    im = self.block6(im)
    im = im.reshape(-1, 2 * 3 * 128)
    x_a = self.fc_a(im)
    mu = self.mu_layer(x_a)
    mu = F.tanh(mu)
    x_s = self.fc_s(im)
    sigma = self.sigma_layer(x_s)
    sigma = F.softplus(sigma) * 0.06 + 0.005
    x_v= self.fc_v(im)
    values = self.v_layer(x_v)
    return mu, sigma, values

  def choose_action(self, s):
    self.training = False
    mu, sigma, _ = self.forward(s)
    m = self.distribution(mu.view(-1,).data, sigma.view(-1,).data)
    return m.sample().cpu().numpy(), mu.cpu().detach().numpy(), sigma.cpu().detach().numpy()

  def loss_func(self, s, a, v_t):
    self.train()
    mu, sigma, values = self.forward(s)
    td = v_t - values
    c_loss = td.pow(2)

    m = self.distribution(mu, sigma)
    log_prob = m.log_prob(a) 
    entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale) 
    exp_v = log_prob * td.detach() + ENTROPY_BETA * entropy
    a_loss = -exp_v
    total_loss = (a_loss + c_loss).mean()
    return total_loss

class Worker(mp.Process):
  def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, wid, SAVE_TOP_DIR):
    super(Worker, self).__init__()
    print("wid %d" % wid)
    self.wid = wid
    self.step = 0

    self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
    self.gnet, self.opt = gnet, opt
    self.random_seed = 42 + self.wid + int(np.log(self.wid * 100 + 1))
    print("random_seed",self.random_seed,"self.wid",self.wid)
    np.random.seed(self.random_seed)
  
    self.lnet = ACNet().to(device)
    self.init_step = 0
    self.SAVE_TOP_DIR = SAVE_TOP_DIR

  def run(self):
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    mean = np.reshape(mean,(1,1,3))
    std = np.reshape(std,(1,1,3))
    self.start_pos = [-0.1,-0.4,0.5]
    self.dt = 1./30.0
    if self.wid == 0:
      self.p_id = pybullet.connect(pybullet.GUI)
    else:
      self.p_id = pybullet.connect(pybullet.DIRECT)


    action_dir = os.path.join(self.SAVE_TOP_DIR,"action.npy")
    fixture_action = np.zeros((3,))
  
    self.env = RobotEnv(worker_id=self.wid,p_id=pybullet,dt=self.dt,maxSteps=20,fixture_offset=fixture_action)


    total_step = 1 + self.init_step
    suc_check = 0
    reward_check = 0
    episode_check = 0
    sigma_check1 = 0
    sigma_check2 = 0
    total_episode = 0

    buffer_s, buffer_a, buffer_r, buffer_done  = [], [], [], []
    while total_step < MAX_EP:
      observation = self.env.reset()
      observation = observation/255.0
      observation = (observation - mean)/std

      observation = np.reshape(observation,(-1,))
 
      while True:
        action, mu_r, sigma_r = self.lnet.choose_action(v_wrap(observation[None,:]))
        action[:3] = action[:3].clip(-0.03,0.03)   
        action[3:] = action[3:].clip(-0.05,0.05)   
#
#        if action[2] > 0.005:
#w          action[2] = 0.005
        
        observation_next, reward, done, suc = self.env.step(action)
        observation_next = observation_next/255.0
        observation_next = (observation_next - mean)/std

        recordGif = False
        if recordGif and total_step > 10:
          imageio.mimsave('pokingSthSlightly.gif',self.env.obs_list)
          return
         
        observation_next = np.reshape(observation_next,(-1,))

        buffer_s.append(observation)
        buffer_r.append(reward)
        buffer_a.append(action)
        buffer_done.append(done)

        if total_step % (UPDATE_GLOBAL_ITER + self.wid) == 0 or done:
          push_and_pull(self.opt, self.lnet, self.gnet, done, observation_next, buffer_s, buffer_a, buffer_r, buffer_done, GAMMA)
          buffer_s, buffer_a, buffer_r, buffer_done  = [], [], [], []
 
        if done:
          suc_check += suc
          episode_check += 1
          total_episode += 1

        observation = observation_next
        total_step += 1
        reward_check += reward

        if total_step % 100 == 0:
          current_performance = float(suc_check)/episode_check
          avg_sigma1 = sigma_check1 / 100.0
          avg_sigma2 = sigma_check2 / 100.0
          if self.wid == 0:
            print(self.SAVE_TOP_DIR,"total step %d, avg suc %f, avg reward %f" % (total_step, suc_check / 100.0, reward_check / 100.0))
          save_path = os.path.join(self.SAVE_TOP_DIR,str(total_step)+'model.pth.tar')
          if self.wid == 0 and int(total_step) % 1000 == 0:
            print("saving to",save_path)
            torch.save(self.gnet.state_dict(), save_path)
          suc_check = 0
          episode_check = 0
          sigma_check1 = 0.0
          sigma_check2 = 0.0

        if done:
          break

    reward_dir = os.path.join(self.SAVE_TOP_DIR,"reward.txt")
    np.savetxt(reward_dir,np.array([reward_check/100.0]),fmt='%f')

    print("finising the learning!")
    torch.cuda.empty_cache()
    print("empyting the cache!")
    sys.exit()
    os._exit(1)


if __name__ == "__main__":
  ExName = 'optimal'#sys.argv[1]
  #print(ExName)
  SAVE_TOP_DIR = os.path.join('./wrench/',ExName)
  if not os.path.exists(SAVE_TOP_DIR):
    os.makedirs(SAVE_TOP_DIR)

  mp.set_start_method('spawn')  
  gnet = ACNet()  # global network
 
  ## loading
  Load_model_id = '2000'
  Load_path = os.path.join(SAVE_TOP_DIR,Load_model_id + 'model.pth.tar')
  #checkpoint = torch.load(Load_path)
  #gnet.load_state_dict(checkpoint)

  gnet.to(device)
  gnet.share_memory()
 
  opt = SharedAdam(gnet.parameters(),lr=0.0001)
  global_ep, global_ep_r, res_queue = mp.Value('i',0), mp.Value('d',0.), mp.Queue()

  workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, SAVE_TOP_DIR) for i in range(1)]
  [w.start() for w in workers]
  res = []

  for worker in workers:
    worker.init_step = 0

  [w.join() for w in workers]

