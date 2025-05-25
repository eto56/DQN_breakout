import os
import datetime
import gymnasium as gym
import ale_py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from CNN import network, replay_buffer, agent

# ヘルパー関数: タイムスタンプ文字列取得
def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# メイン
if __name__ == '__main__':
    # 時間情報を用いたディレクトリ作成\    
    timestamp = get_timestamp()
    save_dir = f'../results/results_{timestamp}/'
    os.makedirs(save_dir, exist_ok=True)

    # 環境とエージェント設定
    base_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    
    env = RecordVideo(base_env, video_folder=save_dir+"video/", episode_trigger=lambda e: e % 1000 == 0)

    MAX_STEP = int(1e4 * 1000)


    #  最低ライン：10 M frames（≈2.5 Mステップ）

    # 本気ライン：50 M frames（≈12.5 Mステップ）

    # 到達感を得るなら：20 M frames（≈5 Mステップ）以上を目安に
    update_freq = 10
    sync_freq = 10000
    action_space = 4
    dqn_agent = agent(action_space=action_space, timesteps=MAX_STEP)
    model_path =  'model.pth'

    dqn_agent.load_model(model_path)
    dqn_agent.net_sync()

    # 学習ロギング用リスト
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
                # アクション選択
                action = dqn_agent.select_action(obs) if steps > 0 else env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                reward_sum += reward

                # リプレイ
                dqn_agent.add(obs, action, reward, next_obs, terminated)
                obs = next_obs

                # 更新
                if steps > 0 and steps % update_freq == 0:
                    dqn_agent.update()
                
                if steps >0 and steps % sync_freq == 0:
                    dqn_agent.net_sync()
                if steps % 10000 == 0:
                    print(f"Step {steps}: epsilon = {dqn_agent.epsilon:.3f}")

                done = bool(terminated or truncated)

            reward_list.append(reward_sum)
            episode_list.append(episode)
            step_list.append(steps)
            print(f"Episode {episode} | Steps {steps} | Return {reward_sum}")

    except KeyboardInterrupt:
        print("Interrupted by user, saving..." )

    finally:
        # モデル保存
        model_path = os.path.join(save_dir, 'model.pth')
        dqn_agent.save_model(model_path)

        # パラメータ保存
        params = {
            'MAX_STEP': MAX_STEP,
            'update_freq': update_freq,
            'action_space': action_space,
            'buffer_size': dqn_agent.replay.buffer.maxlen,
            'batch_size': dqn_agent.batch_size,
            'epsilon_start': dqn_agent.epsilon_start,
            'epsilon_end': dqn_agent.epsilon_end,
            'epsilon_decay': dqn_agent.epsilon_decay,
            'learning_rate': dqn_agent.lr
        }
        with open(os.path.join(save_dir, 'params.txt'), 'w', encoding='utf-8') as f:
            for k, v in params.items():
                f.write(f"{k}: {v}\n")

        # プロット保存
        plt.figure(figsize=(8, 4))
        plt.plot(episode_list, reward_list, label='Return')
        ma = pd.Series(reward_list).rolling(window=20, min_periods=1).mean()
        plt.plot(episode_list, ma, label='20-episode MA')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(save_dir, 'return_plot.png')
        plt.savefig(fig_path)

        # CSV保存
        df = pd.DataFrame({'episode': episode_list, 'step': step_list, 'reward': reward_list})
        csv_path = os.path.join(save_dir, 'rewards.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')

        # 環境閉じる
        env.close()
        print(f"All results saved to {save_dir}")
