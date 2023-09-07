import copy
import os
import argparse
import logging
# this needs to be setup right after the first logging import to ensure everything works as expected
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

import psutil
import numpy as np
import time
import json
import datetime
import torch
from torch_ac.utils.penv import ParallelEnv
import torch_ac
import sys
import wandb

import utils
from model import ACModel
import early_stopping
import metadata


def setup(args: argparse.Namespace):
    # Set run dir
    date = datetime.datetime.now().strftime("%y%m%dT%H%M%S")
    default_model_name = f"{args.env}_{args.algo}_{date}"

    model_name = args.model or default_model_name
    model_name = model_name.replace(" ", "_")
    model_name = model_name.replace("/", "_")
    model_dir = os.path.abspath(utils.get_model_dir(model_name))
    args.output_dirpath = model_dir
    if not os.path.exists(args.output_dirpath):
        os.makedirs(args.output_dirpath)

    # add the file based handler to the logger
    fh = logging.FileHandler(filename=os.path.join(args.output_dirpath, 'log.txt'))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"))
    logging.getLogger().addHandler(fh)

    try:
        # attempt to get the slurm job id and log it
        logging.info("Slurm JobId: {}".format(os.environ['SLURM_JOB_ID']))
        args.slurm_job_id = os.environ['SLURM_JOB_ID']
    except KeyError:
        pass

    try:
        # attempt to get the hostname and log it
        import socket
        hn = socket.gethostname()
        logging.info("Job running on host: {}".format(hn))
        args.hostname = hn
    except RuntimeError:
        pass

    # Set device
    args.device = str(utils.device)
    logging.info("{}\n".format(" ".join(sys.argv)))
    logging.info(args)

    if args.seed is None or args.seed <= 0:
        args.seed = np.random.randint(0, 2**32 - 1)

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # write the args configuration to disk
    logging.info("writing args to config.json")
    with open(os.path.join(args.output_dirpath, 'config.json'), 'w') as fh:
        json.dump(vars(args), fh, ensure_ascii=True, indent=2)

    # wandb.login(key="")
    wandb.init(
        # set the wandb project where this run will be logged
        project="rl-starter-files",
        name=args.model,
        dir=model_dir,
        # mode='offline',

        # track hyperparameters and run metadata
        config=vars(args)
    )

    args.mem = args.recurrence > 1

    # Load environments
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    logging.info("Environments loaded")

    # Load training status
    try:
        status = utils.get_status(args.output_dirpath)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    logging.info("Training status loaded")

    # Load model
    acmodel = ACModel(envs[0].observation_space, envs[0].action_space, args.mem, args.text)
    if "vocab" in status:
        acmodel.preprocess_obss.vocab.load_vocab(status["vocab"])
    logging.info("Observations preprocessor loaded")
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(utils.device)
    logging.info("Model loaded")
    # logging.info("{}\n".format(acmodel))

    # Load algo
    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, utils.device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, acmodel.preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, utils.device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, acmodel.preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    logging.info("Optimizer loaded")

    return status, envs, acmodel, algo


def eval_model(args: argparse.Namespace, acmodel, eval_env, epoch: int, train_stats: metadata.TrainingStats):
    # run one evaluation episode

    acmodel.eval()
    acmodel.to(utils.device)
    # wrap model into agent
    eval_agent = utils.Agent(acmodel, num_envs=args.procs)
    # logging.info("Eval Agent loaded")

    # Initialize logs
    eval_logs = {"eval_num_frames_per_episode": [], "eval_return_per_episode": []}

    # Run agent
    eval_start_time = time.time()

    eval_obss = eval_env.reset()

    eval_log_done_counter = 0
    eval_log_episode_return = torch.zeros(args.procs, device=utils.device)
    eval_log_episode_num_frames = torch.zeros(args.procs, device=utils.device)

    while eval_log_done_counter < args.eval_episodes:
        eval_actions = eval_agent.get_actions(eval_obss)
        eval_obss, eval_rewards, eval_terminateds, eval_truncateds, _ = eval_env.step(eval_actions)
        eval_dones = tuple(a | b for a, b in zip(eval_terminateds, eval_truncateds))
        eval_agent.analyze_feedbacks(eval_rewards, eval_dones)

        eval_log_episode_return += torch.tensor(eval_rewards, device=utils.device, dtype=torch.float)
        eval_log_episode_num_frames += torch.ones(args.procs, device=utils.device)

        for i, eval_done in enumerate(eval_dones):
            if eval_done:
                eval_log_done_counter += 1
                eval_logs["eval_return_per_episode"].append(eval_log_episode_return[i].item())
                eval_logs["eval_num_frames_per_episode"].append(eval_log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(eval_dones, device=utils.device, dtype=torch.float)
        eval_log_episode_return *= mask
        eval_log_episode_num_frames *= mask

    eval_end_time = time.time()

    # log the results
    eval_num_frames = sum(eval_logs["eval_num_frames_per_episode"])
    eval_fps = eval_num_frames / (eval_end_time - eval_start_time)
    eval_duration = int(eval_end_time - eval_start_time)
    eval_return_per_episode = utils.synthesize(eval_logs["eval_return_per_episode"])
    eval_num_frames_per_episode = utils.synthesize(eval_logs["eval_num_frames_per_episode"])

    eval_header = ["eval_update", "eval_frames", "eval_FPS", "eval_duration"]
    eval_data = [eval_log_done_counter, eval_fps, eval_fps, eval_duration]
    eval_header += ["eval_return_" + key for key in eval_return_per_episode.keys()]
    eval_data += eval_return_per_episode.values()
    eval_header += ["eval_num_frames_" + key for key in eval_num_frames_per_episode.keys()]
    eval_data += eval_num_frames_per_episode.values()

    wandb_input_dict = dict()
    for field, value in zip(eval_header, eval_data):
        if isinstance(value, torch.Tensor):
            value = value.item()
        wandb_input_dict[field] = value
    wandb.log(wandb_input_dict)

    for key in wandb_input_dict.keys():
        if not (key.endswith('_min') or key.endswith('_max') or key.endswith('_std')):
            train_stats.add(epoch, key, wandb_input_dict[key])
    # train_stats.add_dict(epoch, wandb_input_dict)

    del eval_agent


def train_epoch(args, algo, num_frames: int, update: int, epoch: int, train_stats: metadata.TrainingStats):
    epoch_num_frames = 0

    algo.acmodel.train()

    while epoch_num_frames < args.eval_interval:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        epoch_num_frames += logs["num_frames"]

        fps = logs["num_frames"] / (update_end_time - update_start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        #rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS"]
        data = [update, num_frames, fps]
        # header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        # data += rreturn_per_episode.values()
        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        # https://docs.wandb.ai/guides/track/log/logging-faqs
        wandb_input_dict = dict()
        for field, value in zip(header, data):
            if isinstance(value, torch.Tensor):
                value = value.item()
            wandb_input_dict[field] = value

        for key in wandb_input_dict.keys():
            if not (key.endswith('_min') or key.endswith('_max') or key.endswith('_std')):
                train_stats.append_accumulate(key, wandb_input_dict[key])

        if update % args.log_interval == 0:
            logging.info(
                "U {} | F {:06} | FPS {:04.0f} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))
            wandb.log(wandb_input_dict)

            # log loss and current GPU utilization
            cpu_mem_percent_used = psutil.virtual_memory().percent
            gpu_mem_percent_used, memory_total_info = utils.get_gpu_memory()
            gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
            logging.info('  cpu_mem: {:2.1f}%  gpu_mem: {}% of {}MiB'.format(cpu_mem_percent_used, gpu_mem_percent_used, memory_total_info))

        update += 1

    train_stats.close_accumulate(epoch, 'update', method='avg')
    train_stats.close_accumulate(epoch, 'frames', method='avg')
    train_stats.close_all_accumulate(epoch=epoch, method='mean')
    return num_frames, update


def main(args: argparse.Namespace):
    status, envs, acmodel, algo = setup(args)  # updates args in place

    # Train model
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    train_stats = metadata.TrainingStats()
    epoch = -1

    plateau_scheduler = early_stopping.EarlyStoppingOnPlateau(mode='max',patience=args.patience, threshold=args.eps)

    # pre-build the eval envs (so they can be reused)
    eval_envs = []
    for i in range(args.procs):
        eval_env = utils.make_env(args.env, args.seed + 10000 * i)
        eval_envs.append(eval_env)
    eval_env = ParallelEnv(eval_envs)
    logging.info("Eval Environments loaded")

    #while num_frames < args.frames:
    while not plateau_scheduler.is_done():
        epoch += 1
        # train for an epoch
        num_frames, update = train_epoch(args, algo, num_frames, update, epoch, train_stats)

        # run eval (save needs to happen first, as this loads the saved model)
        eval_model(args, acmodel, eval_env, epoch, train_stats)

        eval_reward = train_stats.get_epoch('eval_return_mean', epoch=epoch)
        plateau_scheduler.step(eval_reward)

        train_stats.add_global('num_epochs_trained', epoch)

        if plateau_scheduler.is_equiv_to_best_epoch:
            logging.info('Updating best model with epoch: {}'.format(epoch))

            # save the model
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(acmodel.preprocess_obss, "vocab"):
                status["vocab"] = acmodel.preprocess_obss.vocab.vocab
            utils.save_status(status, args.output_dirpath)
            logging.info("Status saved")

            # update the global metrics with the best epoch
            train_stats.update_global(epoch)

        # generate all the plotting
        train_stats.plot_all_metrics(output_dirpath=args.output_dirpath)
        # write copy of current metadata metrics to disk
        train_stats.export(args.output_dirpath)

    wall_time = time.time() - start_time
    train_stats.add_global('wall_time', wall_time)
    logging.info("Total WallTime: {}seconds".format(train_stats.get_global('wall_time')))
    train_stats.export(args.output_dirpath)  # update metrics data on disk

    # close out wandb logging
    wandb.finish()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--algo", required=True,
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=3208920712,
                        help="random seed (default: 3208920712)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="number of updates between two logs (default: 10)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="number of episodes of evaluation (default: 100)")
    parser.add_argument("--eval-interval", type=int, default=50000,
                        help="number of updates between two evaluations (i.e. how big is an epoch)")

    # Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 1)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")

    parser.add_argument('--eps', default=1e-4, type=float, help='eps value for determining early stopping metric equivalence.')
    parser.add_argument('--patience', default=20, type=int, help='number of epochs past optimal to explore before early stopping terminates training.')


    args = parser.parse_args()

    main(args)


