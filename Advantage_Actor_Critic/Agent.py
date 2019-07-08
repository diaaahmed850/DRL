import copy
import glob
import os,sys
from os.path import dirname, abspath
import time
from collections import deque
import datetime
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from baselines.common import plot_util as pu
sys.path.insert(0,dirname(dirname(dirname(abspath(__file__)))))
Model_dir = dirname(dirname(abspath(__file__)))+'/Advantage_Actor_Critic'
sys.path.insert(0,Model_dir)
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.envs import make_vec_envs,VecPyTorch
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule,get_render_func


from arguments import get_args
import shutil
import imageio
from skimage.transform import resize



args = get_args()
args.det = not args.non_det




def trainAdvantageActorCritic():
    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    args.log_dir = os.path.expanduser(args.log_dir)

    plot_path = os.path.join(Model_dir,args.save_dir)
    plot_path = os.path.join(plot_path, args.algo)
    plot_path =os.path.join(plot_path, args.env_name)
    plot_path=os.path.join(plot_path,args.train_type)
    plot_path=os.path.join(plot_path, args.folder)
    plot_path=os.path.join(plot_path,args.log_dir)

    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)
    os.makedirs(plot_path)


    """
    try:
        
        os.makedirs(args.log_dir)
        #print("tryyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
    except OSError:
        #print("exceptttttttttttttttttttttttttttttt")
        #files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        #print(OSError)
        #for f in files:
            #os.remove(f)
        print("eh fe eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeehh")
        pass

    #eval_log_dir = args.log_dir + "_eval"
    eval_log_dir=args.log_dir
    """
    
    """
    try:
        os.makedirs(eval_log_dir)
        print("tryyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
    except OSError:
        print("exceptttttttttttttttttttttttttttttt")
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        print(OSError)
        for f in files:
            os.remove(f)
    """
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0]
                                       for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(Model_dir,args.save_dir)
            save_path = os.path.join(save_path, args.algo)
            save_path =os.path.join(save_path, args.env_name)
            save_path=os.path.join(save_path,args.train_type)
            save_path=os.path.join(save_path, args.folder)
         

            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path,  args.env_name+'_'+args.train_type+ ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, plot_path, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.tensor([[0.0] if done_ else [1.0]
                                           for done_ in done],
                                           dtype=torch.float32,
                                           device=device)
                
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))

def testAdvantageActorCritic():
    env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                            None, None, device='cpu',
                            allow_early_resets=False)

    # Get a render function
    render_func = get_render_func(env)

    # We need to use the same statistics for normalization as used in training
    save_path = os.path.join(Model_dir,args.save_dir)
    save_path = os.path.join(save_path, args.algo)
    save_path =os.path.join(save_path, args.env_name)
    save_path=os.path.join(save_path,args.train_type)
    save_path=os.path.join(save_path, args.folder)
    try:
        os.mkdir(save_path+'/gifs')
    except:
        pass
    """
    if args.view == 'try':
        save_path=os.path.join(save_path, args.folder)
    elif args.view == 'present':
        save_path=os.path.join(save_path, 'Final')
    """
    actor_critic, ob_rms = \
                torch.load( os.path.join(save_path,  args.env_name+'_'+args.train_type+ ".pt"))

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    if render_func is not None:
        render_func('human')

    obs = env.reset()
    frames=[]
    total_reward=0.0
    e=1
    if args.save_fig:
        frames.append(np.array(env.venv.get_images()[0]))

    if args.env_name.find('Bullet') > -1:
        import pybullet as p

        torsoId = -1
        for i in range(p.getNumBodies()):
            if (p.getBodyInfo(i)[0].decode() == "torso"):
                torsoId = i

    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)
        if args.save_fig:
            total_reward=total_reward+reward.item()
            frames.append(np.array(env.venv.get_images()[0]))

        masks.fill_(0.0 if done else 1.0)

        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if render_func is not None:
            render_func('human')
        if done and args.save_fig:
            generate_gif(e,frames,total_reward,save_path)
            total_reward=0.0
            frames=[]
            e=e+1

def generate_gif(num, frames_for_gif,score,path):
    """
    Args:
        frame_number: Integer, determining the number of the current frame
        frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
        reward: Integer, Total reward of the episode that es ouputted as a gif
        path: String, path where gif is saved
    """
    print(len(frames_for_gif))
    
    for idx, frame_idx in enumerate(frames_for_gif): 
        frames_for_gif[idx] = resize(frame_idx, (480, 480, 3), 
                             preserve_range=True, order=0).astype(np.uint8)
    imageio.mimsave(path+"/gifs/"+"0"+str(num)+"-"+args.env_name+"-"+args.train_type+"-"+str(int(score))+".mp4", 
                    frames_for_gif)#, duration=1/30)
    """
    if(len(frames_for_gif)>=8000):
        print(len(frames_for_gif[0:int(len(frames_for_gif)/2)]))
        print(len(frames_for_gif[int(len(frames_for_gif)/2):]))
        f1=frames_for_gif[0:int(len(frames_for_gif)/2)]
        f2=frames_for_gif[int(len(frames_for_gif)/2):]
        print("dgdsgsdgdsgdsgdsgsddsgsgsdsgsgsdg")
        imageio.mimsave(path+"/gifs/"+"0"+str(num)+"-"+args.env_name+"-"+args.train_type+"-"+str(int(score))+"-part1.mpeg", 
                    f1)#, duration=1/30)
        
        imageio.mimsave(path+"/gifs/"+"0"+str(num)+"-"+args.env_name+"-"+args.train_type+"-"+str(int(score))+"part2.mpeg", 
                    f2)#, duration=1/30)
        
    else:
        imageio.mimsave(path+"/gifs/"+"0"+str(num)+"-"+args.env_name+"-"+args.train_type+"-"+str(int(score))+".mp4", 
                    frames_for_gif)#, duration=1/30)
    """

    



def plotAdvantageActorCritic():
    plot_path = os.path.join(Model_dir,args.save_dir)
    plot_path = os.path.join(plot_path, args.algo)
    plot_path =os.path.join(plot_path, args.env_name)
    plot_path=os.path.join(plot_path,args.train_type)
    plot_path=os.path.join(plot_path, args.folder)
    plot_path=os.path.join(plot_path,args.log_dir)
    results = pu.load_results(plot_path)
    fig = pu.plot_results(results, average_group=True, split_fn=lambda _: '', shaded_std=False)

    ax = plt.gca()
    plt.xlabel('Number of Steps', ha='center', va='center', ma='left',rotation=0)
    plt.tight_layout()
    plt.ylabel('Rewards', ha='center', va='center', ma='left',rotation=90)
    plt.tight_layout()
    plt.savefig(plot_path+'/'+args.env_name+"-"+args.train_type+'-'+'rewards_plot.png')
    
 


#trainAdvantageActorCritic()

    