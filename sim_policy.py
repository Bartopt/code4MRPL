import os, shutil
import os.path as osp
import pickle
import json
import numpy as np
import click
import torch

from rlkit.core import logger
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv #CameraWrapper
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.agent import PEARLAgent
from configs.default import default_config
from launch_experiment import deep_update_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util import rollout


def sim_policy(variant, path_to_exp, num_trajs=1, deterministic=False, save_video=False, sparse=False,
               dump_eval_paths=False, demo_num=[1], not_only_demo=True, mean=False):
    '''
    simulate a trained policy adapting to a new task
    optionally save videos of the trajectories - requires ffmpeg

    :variant: experiment configuration dict
    :path_to_exp: path to exp folder
    :num_trajs: number of trajectories to simulate per task (default 1)
    :deterministic: if the policy is deterministic (default stochastic)
    :save_video: whether to generate and save a video (default False)
    :sparse: True when env is sparse-point-robot, otherwise False
    :dump_eval_paths: whether to save the paths for visualization, only useful for point-robot
    :demo_num: how much demo provide for agent, it can be decimal but cannot be 0
    :not_only_demo: Whether to use only demos sampling z, True for not only (default True)
    '''

    # create multi-task environment and sample tasks
    # env = CameraWrapper(NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params'])), variant['util_params']['gpu_id'])
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    eval_tasks=list(tasks[-variant['n_eval_tasks']:])
    print('testing on {} test tasks, {} trajectories each'.format(len(eval_tasks), num_trajs))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=obs_dim + action_dim + reward_dim,
        output_size=context_encoder,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    # deterministic eval
    if deterministic:
        agent = MakeDeterministic(agent)

    # load trained weights (otherwise simulate random policy)
    context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth')))
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth')))

    # logger setting, more complete is using setup_logger
    logger.set_snapshot_dir(path_to_exp)

    # loop through tasks collecting rollouts
    res = {'Demo0': {'all_sucs': [], 'all_rets': [], 'demo': 0}}
    if len(demo_num) > 0:
        for d in demo_num:
            res['Demo{}'.format(d)] = {'all_sucs': [], 'all_rets': [], 'demo': d}

    video_frames = []
    for idx in eval_tasks:
        env.reset_task(idx)
        agent.clear_z()
        all_paths = {}
        for k in res.keys():
            all_paths[k] = []

        #
        print('No demonstration')
        for n in range(num_trajs):
            path = rollout(env, agent, max_path_length=variant['algo_params']['max_path_length'], accum_context=True,
                           save_frames=save_video)
            all_paths['Demo0'].append(path)
            # to make a fair comparison with the demons situation
            agent.infer_posterior(agent.context)

        #
        if len(demo_num) > 0:
            print('extract latest agent policy as demonstration')
            Demos_context = {}
            for k in res.keys():
                if k == 'Demo0':
                    continue
                Demos_context[k] = agent.context[:, -int(variant['algo_params']['max_path_length'] * res[k]['demo']):, :]

            for k, v in Demos_context.items():
                print('{} Demonstration'.format(str(res[k]['demo'])))
                # select several latest replay buffer as expert demonstrations
                agent.context = v
                # second act with env
                for n in range(num_trajs):
                    agent.infer_posterior(agent.context)
                    path = rollout(env, agent, max_path_length=variant['algo_params']['max_path_length'],
                                   accum_context=not_only_demo, save_frames=save_video)
                    all_paths[k].append(path)

        if sparse:
            for k, v in all_paths.items():
                for p in v:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

        for k, v in res.items():
            res[k]['all_rets'].append([sum(p['rewards']) for p in all_paths[k]])
            try:
                res[k]['all_sucs'].append([np.array(p['env_infos'][-1]['IsSuc']) for p in all_paths[k]])
            except:
                pass

    if save_video:
        # save frames to file temporarily
        temp_dir = os.path.join(path_to_exp, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        for i, frm in enumerate(video_frames):
            frm.save(os.path.join(temp_dir, '%06d.jpg' % i))

        video_filename=os.path.join(path_to_exp, 'video.mp4'.format(idx))
        # run ffmpeg to make the video
        os.system('ffmpeg -i {}/%06d.jpg -vcodec mpeg4 {}'.format(temp_dir, video_filename))
        # delete the frames
        shutil.rmtree(temp_dir)

    # compute average returns across tasks
    n = min([len(a) for a in res['Demo0']['all_rets']])
    rets = {}
    sucs = {}
    for k in res.keys():
        rets[k] = [a[:n] for a in res[k]['all_rets']]
        if mean:
            rets[k] = np.mean(np.stack(rets[k]), axis=0)
        else:
            rets[k] = np.stack(rets[k])
        try:
            sucs[k] = [a[:n] for a in res[k]['all_sucs']]
            if mean:
                sucs[k] = np.mean(np.stack(sucs[k]), axis=0)
            else:
                sucs[k] = np.stack(sucs[k])
        except:
            pass

    # for i, ret in enumerate(rets):
    #     print('trajectory {}, avg return: {} , avg success rate: {} \n'.format(i, ret, sucs[i]))

    if not os.path.exists(path_to_exp + '/eval_trajectories'):
        os.makedirs(path_to_exp + '/eval_trajectories')

    for k in res.keys():
        if sparse:
            with open(path_to_exp + '/eval_trajectories/sret_demo{}_acctext{}_mean{}.pkl'.format(
                    str(res[k]['demo']), str(not_only_demo), str(mean)), 'wb') as handle:
                pickle.dump(rets[k], handle)
        else:
            with open(path_to_exp + '/eval_trajectories/ret_demo{}_acctext{}_mean{}.pkl'.format(
                    str(res[k]['demo']), str(not_only_demo), str(mean)), 'wb') as handle:
                pickle.dump(rets[k], handle)
        with open(path_to_exp + '/eval_trajectories/suc_demo{}_acctext{}_mean{}.pkl'.format(
                    str(res[k]['demo']), str(not_only_demo), str(mean)), 'wb') as handle:
                pickle.dump(sucs[k], handle)


@click.command()
@click.argument('config', default=None)
@click.argument('path', default=None)
@click.option('--num_trajs', default=3)
@click.option('--deterministic', is_flag=True, default=False)
@click.option('--video', is_flag=True, default=False)
def main(config, path, num_trajs, deterministic, video):
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    sim_policy(variant, path, num_trajs, deterministic, video)


if __name__ == "__main__":
    main()
