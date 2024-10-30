# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from easydict import EasyDict
from gym import wrappers
from isaacgym import gymapi
from isaacgym import gymutil
from omegaconf import DictConfig, OmegaConf
from rl_games.torch_runner import Runner

from phc.env.tasks import humanoid_amp_task
from phc.env.tasks.humanoid_amp import HumanoidAMP
from phc.learning import amp_models, network_builder
from phc.learning import amp_network_builder
from phc.learning import amp_network_mcp_builder
from phc.learning import amp_network_pnn_builder
from phc.learning import amp_players
from phc.learning import im_amp_players
from phc.utils.config import set_np_formatting, set_seed, SIM_TIMESTEP
from phc.utils.flags import flags
from phc.utils.parse_task import parse_task
from poselib.poselib.skeleton.skeleton3d import SkeletonState
from rl_games.algos_torch import torch_ext, model_builder
from rl_games.common import env_configurations, vecenv, object_factory
from rl_games.common.algo_observer import AlgoObserver
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import hydra
import json
import numpy as np
import os
import os.path as osp
import sys
import time
import torch

from utils.bsmg_xror_utils import open_xror
from utils.pose_utils import expm_to_quat, unity_to_zup

sys.path.append(os.getcwd())


args = None
cfg = None
cfg_train = None
COLLECT_Z = False


class MyPlayer(im_amp_players.IMAMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)

        d = open_xror(f"data/sample_data/4233b6fe-1fa4-4c48-8259-2e202d902531.xror")

        three_p = d["gt_3p_np"].reshape(-1, 3, 6)
        three_p = three_p[::2]

        xyzs = three_p[..., :3]
        expm = three_p[..., 3:]
        quat = expm_to_quat(expm)
        xyzs, quat = unity_to_zup(xyzs, quat)

        self.max_steps = three_p.shape[0] - 1

        # centering
        xyzs[..., [0, 1]] -= xyzs[0, 0, [0, 1]]

        # Local angle corrections
        my_lhand_rot = Rotation.from_quat(quat[..., 1, :].reshape((-1, 4)))

        my_lhand_eul = my_lhand_rot.as_euler("ZYX", degrees=True)
        my_lhand_eul[..., -1] -= 90
        my_lhand_rot = Rotation.from_euler("ZYX", my_lhand_eul, degrees=True)

        my_lhand_eul = my_lhand_rot.as_euler("XYZ", degrees=True)
        my_lhand_eul[..., -1] -= 90
        my_lhand_rot = Rotation.from_euler("XYZ", my_lhand_eul, degrees=True)

        quat[..., 1, :] = my_lhand_rot.as_quat()

        my_rhand_rot = Rotation.from_quat(quat[..., 2, :].reshape((-1, 4)))

        my_rhand_eul = my_rhand_rot.as_euler("ZYX", degrees=True)
        my_rhand_eul[..., -1] += 90
        my_rhand_rot = Rotation.from_euler("ZYX", my_rhand_eul, degrees=True)

        my_rhand_eul = my_rhand_rot.as_euler("XYZ", degrees=True)
        my_rhand_eul[..., -1] += 90
        my_rhand_rot = Rotation.from_euler("XYZ", my_rhand_eul, degrees=True)

        quat[..., 2, :] = my_rhand_rot.as_quat()

        xyzs *= 1.5044 / np.median(xyzs[:100, 0, 2])

        n = three_p.shape[0]

        custom_3p_ref_pos = torch.zeros((n, 24, 3), dtype=torch.float, device=self.device)
        custom_3p_ref_pos[:, [13, 18, 23]] = torch.tensor(xyzs.reshape(-1, 3, 3), dtype=torch.float, device=self.device)

        custom_3p_ref_rot = torch.zeros((n, 24, 4), dtype=torch.float, device=self.device)
        custom_3p_ref_rot[:, [13, 18, 23]] = torch.tensor(quat.reshape(-1, 3, 4), dtype=torch.float, device=self.device)

        self.env.task.custom_3p_ref_pos = custom_3p_ref_pos
        self.env.task.custom_3p_ref_rot = custom_3p_ref_rot

    def run(self):
        # basename = os.path.splitext(os.path.basename(cfg["curr_motion_path"]))[0]
        basename = "hehe"

        rb_pos_history = []
        rb_rot_history = []
        rb_vel_history = []
        rb_ang_history = []
        reward_history = []
        ref_rb_pos_history = []
        ref_rb_rot_history = []
        ref_rb_vel_history = []
        ref_rb_ang_history = []
        torque_history = []
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = 1
        # n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for t in range(n_games):
            if games_played >= n_games:
                break
            obs_dict = self.env_reset()

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            _state_init = self.env.task._state_init
            self.env.task._state_init = HumanoidAMP.StateInit.Start

            with torch.no_grad():
                for n in tqdm(range(self.max_steps)):
                    # for n in tqdm(range(100)):
                    obs_dict = self.env_reset(done_indices)

                    if COLLECT_Z:
                        z = self.get_z(obs_dict)

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)

                    obs_dict, r, done, info = self.env_step(self.env, action)

                    cr += r
                    steps += 1

                    if COLLECT_Z:
                        info["z"] = z
                    done = self._post_step(info, done.clone())
                    done *= 0

                    if render:
                        self.env.render(mode="human")
                        time.sleep(self.render_sleep)

                    for _ in range(2):
                        rb_pos_history.append(self.env.task._rigid_body_pos.detach().cpu().numpy())
                        rb_rot_history.append(self.env.task._rigid_body_rot.detach().cpu().numpy())
                        rb_vel_history.append(self.env.task._rigid_body_vel.detach().cpu().numpy())
                        rb_ang_history.append(self.env.task._rigid_body_ang_vel.detach().cpu().numpy())
                        ref_rb_pos_history.append(self.env.task.ref_body_pos.detach().cpu().numpy())
                        ref_rb_rot_history.append(self.env.task.ref_body_rot.detach().cpu().numpy())
                        ref_rb_vel_history.append(self.env.task.ref_body_vel.detach().cpu().numpy())
                        ref_rb_ang_history.append(self.env.task.ref_body_ang_vel.detach().cpu().numpy())
                        reward_history.append(r)
                        # torque_history.append(self.env.task.torques.detach().cpu().numpy())

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if "battle_won" in info:
                                print_game_res = True
                                game_res = info.get("battle_won", 0.5)
                            if "scores" in info:
                                print_game_res = True
                                game_res = info.get("scores", 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print(
                                    "reward:",
                                    cur_rewards / done_count,
                                    "steps:",
                                    cur_steps / done_count,
                                    "w:",
                                    game_res,
                                )
                            else:
                                print(
                                    "reward:",
                                    cur_rewards / done_count,
                                    "steps:",
                                    cur_steps / done_count,
                                )

                        sum_game_res += game_res
                        # if batch_size//self.num_agents == 1 or games_played >= n_games:
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        pos_stacked = np.stack(rb_pos_history, axis=1)
        rot_stacked = np.stack(rb_rot_history, axis=1)
        vel_stacked = np.stack(rb_vel_history, axis=1)
        ang_stacked = np.stack(rb_ang_history, axis=1)
        ref_rb_pos_stacked = np.stack(ref_rb_pos_history, axis=1)
        ref_rb_rot_stacked = np.stack(ref_rb_rot_history, axis=1)
        ref_rb_vel_stacked = np.stack(ref_rb_vel_history, axis=1)
        ref_rb_ang_stacked = np.stack(ref_rb_ang_history, axis=1)
        reward_stacked = np.stack(reward_history, axis=1)
        # torque_stacked = np.stack(torque_history, axis=1)
        cumulative_rewards = np.sum(reward_stacked, axis=1).reshape(-1)
        elite_idxs = np.argsort(cumulative_rewards)[-10:]
        d = {
            # "epoch": self.epoch_num,
            # "out_name": cfg["out_name"],
            "rb_pos": pos_stacked[[0]],
            "rb_rot": rot_stacked[[0]],
            "rb_vel": vel_stacked[[0]],
            "rb_ang": ang_stacked[[0]],
            "ref_rb_pos": ref_rb_pos_stacked[[0]],
            "ref_rb_rot": ref_rb_rot_stacked[[0]],
            "ref_rb_vel": ref_rb_vel_stacked[[0]],
            "ref_rb_ang": ref_rb_ang_stacked[[0]],
            "rewards": reward_stacked[[0]],
            # "torques": torque_stacked[[0]],
            "cumulative_rewards": cumulative_rewards,
        }
        torch.save(d, f"out/posrot.pkl")

        rot_stacked_for_json = torch.as_tensor(rot_stacked[0])
        pos_stacked_for_json = torch.as_tensor(pos_stacked[0])

        # Convert to Unity JSON compatible format, use local quaternions
        sk_tree = self.env.task.skeleton_trees[0]
        sk_state = SkeletonState.from_rotation_and_root_translation(sk_tree, rot_stacked_for_json, pos_stacked_for_json[:, 0], is_local=False)
        global_translation = sk_state.global_translation.cpu().detach().numpy()
        local_rotation = sk_state.local_rotation.cpu().detach().numpy()

        # zup-to-unity conversion
        global_translation[..., 1] *= -1
        global_translation[..., [0, 1, 2]] = global_translation[..., [1, 2, 0]]
        local_rotation[..., [0, 2]] *= -1
        local_rotation[..., [0, 1, 2, 3]] = local_rotation[..., [1, 2, 0, 3]]

        my_json_dict = {"frames": []}
        for i in range(pos_stacked_for_json.shape[0]):
            frame_dict = {"rootXyz": global_translation[i, 0].tolist(), "jointLocalQuat": local_rotation[i].tolist()}
            my_json_dict["frames"].append(frame_dict)
        with open(f"out/posrot.json", "w") as f:
            json.dump(my_json_dict, f)

        return


class MyRunner(Runner):

    def __init__(self, algo_observer=None):
        super().__init__(algo_observer)
        self.algo_factory = object_factory.ObjectFactory()
        # self.algo_factory.register_builder("a2c_continuous", lambda **kwargs: MyAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder("a2c_continuous", lambda **kwargs: MyPlayer(**kwargs))

        self.model_builder = model_builder.ModelBuilder()
        self.network_builder = network_builder.NetworkBuilder()

        self.algo_observer = algo_observer

        torch.backends.cudnn.benchmark = True

    def run(self, args):
        self.load_path = f"data/policies/tracker.pkl"

        if args["train"]:
            self.run_train()
        elif args["play"]:
            print("Started to play")
            player = self.create_player()
            d = torch_ext.load_checkpoint(self.load_path)
            checkpoint = d["state"]
            # player.set_full_state_weights(checkpoint)
            player.model.load_state_dict(checkpoint["model"])
            if player.normalize_input:
                player.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
            player.epoch_num = checkpoint["epoch"]
            if player._normalize_amp_input:
                player._amp_input_mean_std.load_state_dict(checkpoint["amp_input_mean_std"])
                if player._normalize_input:
                    player.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
            # player.restore(self.load_path)

            player.run()
        else:
            self.run_train()


def parse_sim_params(cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = cfg.sim.slices

    if cfg.sim.use_flex:
        if cfg.sim.pipeline in ["gpu"]:
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.use_flex.shape_collision_margin = 0.01
        sim_params.use_flex.num_outer_iterations = 4
        sim_params.use_flex.num_inner_iterations = 10
    else:  # use gymapi.SIM_PHYSX
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
        sim_params.physx.num_subscenes = cfg.sim.subscenes
        if flags.test and not flags.im_eval:
            sim_params.physx.max_gpu_contact_pairs = 4 * 1024 * 1024
        else:
            sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024

    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if not cfg.sim.use_flex and cfg.sim.physx.num_threads > 0:
        sim_params.physx.num_threads = cfg.sim.physx.num_threads

    return sim_params


def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train["params"]["config"].get("multi_gpu", False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train["params"]["seed"] = cfg_train["params"]["seed"] + rank

        args.device = "cuda"
        args.device_id = rank
        args.rl_device = "cuda:" + str(rank)

        cfg["rank"] = rank
        cfg["rl_device"] = "cuda:" + str(rank)

    sim_params = parse_sim_params(cfg)
    args = EasyDict(
        {
            "task": cfg.env.task,
            "device_id": cfg.device_id,
            "rl_device": cfg.rl_device,
            "physics_engine": (gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX),
            "headless": cfg.headless,
            "device": cfg.device,
        }
    )  #### ZL: patch
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print(env.num_envs)
    print(env.num_actions)
    print(env.num_obs)
    print(env.num_states)

    frames = kwargs.pop("frames", 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):

    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and "consecutive_successes" in infos:
                cons_successes = infos["consecutive_successes"].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and "successes" in infos:
                successes = infos["successes"].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar("successes/consecutive_successes/mean", mean_con_successes, frame)
            self.writer.add_scalar("successes/consecutive_successes/iter", mean_con_successes, epoch_num)
            self.writer.add_scalar("successes/consecutive_successes/time", mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.use_global_obs = self.env.num_states > 0

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space
        info["amp_observation_space"] = self.env.amp_observation_space

        info["enc_amp_observation_space"] = self.env.enc_amp_observation_space

        if isinstance(self.env.task, humanoid_amp_task.HumanoidAMPTask):
            info["task_obs_size"] = self.env.task.get_task_obs_size()
        else:
            info["task_obs_size"] = 0

        if self.use_global_obs:
            info["state_space"] = self.env.state_space
            print(info["action_space"], info["observation_space"], info["state_space"])
        else:
            print(info["action_space"], info["observation_space"])

        return info


vecenv.register(
    "RLGPU",
    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
)
env_configurations.register(
    "rlgpu",
    {
        "env_creator": lambda **kwargs: create_rlgpu_env(**kwargs),
        "vecenv_type": "RLGPU",
    },
)


def build_alg_runner(algo_observer):
    runner = MyRunner(algo_observer)
    runner.player_factory.register_builder("amp_discrete", lambda **kwargs: amp_players.AMPPlayerDiscrete(**kwargs))

    runner.algo_factory.register_builder(
        # "amp", lambda **kwargs: amp_agent.AMPAgent(**kwargs)
        "amp",
        lambda **kwargs: MyAgent(**kwargs),
    )
    runner.player_factory.register_builder("amp", lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))

    runner.model_builder.model_factory.register_builder("amp", lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
    runner.model_builder.network_factory.register_builder("amp", lambda **kwargs: amp_network_builder.AMPBuilder())
    runner.model_builder.network_factory.register_builder("amp_mcp", lambda **kwargs: amp_network_mcp_builder.AMPMCPBuilder())
    runner.model_builder.network_factory.register_builder("amp_pnn", lambda **kwargs: amp_network_pnn_builder.AMPPNNBuilder())

    runner.algo_factory.register_builder("im_amp", lambda **kwargs: MyAgent(**kwargs))
    runner.player_factory.register_builder("im_amp", lambda **kwargs: MyPlayer(**kwargs))

    return runner


@hydra.main(
    version_base=None,
    config_path="./data/cfg",
    config_name="config",
)
def main(cfg_hydra: DictConfig) -> None:
    global cfg_train
    global cfg

    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    set_np_formatting()
    (
        flags.debug,
        flags.follow,
        flags.fixed,
        flags.divide_group,
        flags.no_collision_check,
        flags.fixed_path,
        flags.real_path,
        flags.show_traj,
        flags.server_mode,
        flags.slow,
        flags.real_traj,
        flags.im_eval,
        flags.no_virtual_display,
        flags.render_o3d,
    ) = (
        cfg.debug,
        cfg.follow,
        False,
        False,
        False,
        False,
        False,
        True,
        cfg.server_mode,
        False,
        False,
        cfg.im_eval,
        cfg.no_virtual_display,
        cfg.render_o3d,
    )

    flags.test = cfg.test
    flags.add_proj = cfg.add_proj
    flags.has_eval = cfg.has_eval
    flags.trigger_input = False

    if cfg.server_mode:
        flags.follow = cfg.follow = True
        flags.fixed = cfg.fixed = True
        flags.no_collision_check = True
        flags.show_traj = True
        cfg["env"]["episode_length"] = 99999999999999

    if cfg.real_traj:
        cfg["env"]["episode_length"] = 99999999999999
        flags.real_traj = True

    cfg.train = not cfg.test
    set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

    # Create default directories for weights and statistics
    cfg_train = cfg.learning
    cfg["env"]["motion_file"] = f"data/sample_data/amass_isaac_standing_upright_slim.pkl"
    cfg["env"]["num_envs"] = 1
    algo_observer = RLGPUAlgoObserver()
    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(cfg)

    return


if __name__ == "__main__":
    main()
