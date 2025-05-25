import os
import datetime
import gymnasium as gym
import ale_py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo, AtariPreprocessing, FrameStackObservation
from CNN import network, replay_buffer, agent
import argparse
import torch
import wandb


# argparse
parser = argparse.ArgumentParser(description="DQN Agent for Breakout")
parser.add_argument(
    "--load_model",
    action="store_true",
    help="Flag to indicate whether to load the model",
)
parser.add_argument("--set_params", action="store_true", help="Flag to set parameters")

parser.add_argument(
    "--model_path", type=str, default="model.pth", help="Path to the model file"
)
parser.add_argument(
    "--max_step", type=int, default=int(1e4 * 1000), help="Maximum number of steps"
)
parser.add_argument("--update_freq", type=int, default=10, help="Update frequency")
parser.add_argument("--sync_freq", type=int, default=10000, help="Sync frequency")
parser.add_argument("--action_space", type=int, default=4, help="Action space size")

parser.add_argument(
    "--buffer_size", type=int, default=1000000, help="Replay buffer size"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--epsilon_start", type=float, default=1.0, help="Initial epsilon for exploration"
)
parser.add_argument(
    "--epsilon_end", type=float, default=0.1, help="Final epsilon for exploration"
)
parser.add_argument(
    "--epsilon_decay_steps", type=float, default=250000, help="Epsilon decay rate"
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.00025,
    help="Learning rate for the optimizer",
)
parser.add_argument(
    "--gamma", type=float, default=0.99, help="Discount factor for future rewards"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use for training (cpu or cuda)",
)

args = parser.parse_args()


# wandb
wandb.login()
wandb.init(project="DQN_Breakout")

wandb.config.update(
    {
        "max_step": args.max_step,
        "update_freq": args.update_freq,
        "sync_freq": args.sync_freq,
        "action_space": args.action_space,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "device": args.device,
    }
)


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":

    timestamp = get_timestamp()
    save_dir = f"../results/results_{timestamp}/"
    os.makedirs(save_dir, exist_ok=True)

    base_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)

    # original (210, 160, 3)
    # preprocess (84, 84, 4)

    env = AtariPreprocessing(
        base_env,
        noop_max=30,  # リセット後に 0～30 no-op
        grayscale_obs=True,
        frame_skip=4,
        scale_obs=True,  # 0–255→0.0–1.0 float
        screen_size=84,  # リサイズ先
    )

    # ③ フレームスタック（最新 4 フレームをチャンネル方向に結合）
    env = FrameStackObservation(env, stack_size=4)

    env = RecordVideo(
        env, video_folder=save_dir + "video/", episode_trigger=lambda e: e % 1000 == 0
    )
    # Observation Space :Boinput(0, 255, (210, 160, 3), uint8)

    MAX_STEP = args.max_step

    #  最低ライン：10 M frames（≈2.5 Mステップ）

    # 本気ライン：50 M frames（≈12.5 Mステップ）

    # 到達感を得るなら：20 M frames（≈5 Mステップ）以上を目安に
    update_freq = args.update_freq
    sync_freq = args.sync_freq
    action_space = 4
    dqn_agent = agent(action_space=action_space, timesteps=MAX_STEP)

    if args.set_params:
        dqn_agent.set_paramaters(
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps,
            gamma=args.gamma,
            lr=args.learning_rate,
            device=torch.device(args.device),
        )

    if args.load_model:
        model_path = args.model_path
        dqn_agent.load_model(model_path)
        dqn_agent.net_sync()
        print(f"Model loaded from {model_path}")
    else:
        print("Training from scratch, no model loaded.")

    reward_list = []
    step_list = []
    episode_list = []

    episode = 0
    steps = 0

    try:
        while steps < MAX_STEP:
            episode += 1
            obs, info = env.reset()

            reward_sum = 0
            done = False

            while not done and steps < MAX_STEP:
                action = (
                    dqn_agent.select_action(obs)
                    if steps > 3
                    else env.action_space.sample()
                )
                next_obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                reward_sum += reward

                dqn_agent.add(obs, action, reward, next_obs, terminated)
                obs = next_obs

                if steps > 0 and steps % update_freq == 0:
                    dqn_agent.update()

                if steps > 0 and steps % sync_freq == 0:
                    dqn_agent.net_sync()
                if steps % 10000 == 0:
                    print(f"Step {steps}: epsilon = {dqn_agent.epsilon:.3f}")
                wandb.log({"step": steps, "epsilon": dqn_agent.epsilon})
                wandb.log({"action": action})

                done = bool(terminated or truncated)

            reward_list.append(reward_sum)
            episode_list.append(episode)
            step_list.append(steps)
            print(f"Episode {episode} | Steps {steps} | Return {reward_sum}")
            wandb.log({"episode": episode, "reward": reward_sum})
            wandb.log({"step": steps})

    except KeyboardInterrupt:
        print("Interrupted by user, saving...")

    finally:

        wandb.finish()
        model_path = os.path.join(save_dir, "model.pth")
        dqn_agent.save_model(model_path)

        params = {
            "MAX_STEP": MAX_STEP,
            "update_freq": update_freq,
            "action_space": action_space,
            "buffer_size": dqn_agent.replay.buffer.maxlen,
            "batch_size": dqn_agent.batch_size,
            "epsilon_start": dqn_agent.epsilon_start,
            "epsilon_end": dqn_agent.epsilon_end,
            "epsilon_decay": dqn_agent.epsilon_decay,
            "epsilon_decay_steps": dqn_agent.epsilon_decay_steps,
            "gamma": dqn_agent.gamma,
            "device": dqn_agent.device,
            "learning_rate": dqn_agent.lr,
        }
        with open(os.path.join(save_dir, "params.txt"), "w", encoding="utf-8") as f:
            for k, v in params.items():
                f.write(f"{k}: {v}\n")

        plt.figure(figsize=(8, 4))
        plt.plot(episode_list, reward_list, label="Return")
        ma = pd.Series(reward_list).rolling(window=20, min_periods=1).mean()
        plt.plot(episode_list, ma, label="20-episode MA")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(save_dir, "return_plot.png")
        plt.savefig(fig_path)

        df = pd.DataFrame(
            {"episode": episode_list, "step": step_list, "reward": reward_list}
        )
        csv_path = os.path.join(save_dir, "rewards.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")

        env.close()
        print(f"All results saved to {save_dir}")
