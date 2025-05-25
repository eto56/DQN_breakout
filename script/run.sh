#!/bin/bash
#SBATCH -p dgx-a100-80g    
#SBATCH -G 1    
#SBATCH -t 3-0  
#SBATCH -J run
#SBATCH -o ../log/stdout.%J   
#SBATCH -e ../log/stderr.%J 
#SBATCH --mail-type=ALL          # when you want to get notifications. You can select one from [BEGIN , END , FAIL , REQUEUE , ALL] 
#SBATCH --mail-user="matsukawa@mi.t.u-tokyo.ac.jp" # Email address which receives notifications

model_path="model.pth"

max_step=10000000

update_freq=10
sync_freq=10000
action_space=4
buffer_size=1000000
batch_size=32
epsilon_start=1.0
epsilon_end=0.1
epsilon_decay_steps=1000000
learning_rate=0.00025
gamma=0.99
output_txt="output.txt"

python main.py  --set_params \
  --model_path  "$model_path" \
  --max_step     "$max_step"    \
  --update_freq  "$update_freq" \
  --sync_freq    "$sync_freq"   \
  --action_space "$action_space"\
  --buffer_size  "$buffer_size" \
  --batch_size   "$batch_size"  \
  --epsilon_start "$epsilon_start" \
  --epsilon_end   "$epsilon_end"   \
  --epsilon_decay_steps "$epsilon_decay_steps" \
  --learning_rate "$learning_rate" \
  --gamma        "$gamma" 



# parser.add_argument('--load_model', action='store_true', help='Flag to indicate whether to load the model')
# parser.add_argument('--set_params', action='store_true', help='Flag to set parameters')

# parser.add_argument('--model_path', type=str, default='model.pth', help='Path to the model file')
# parser.add_argument('--max_step', type=int, default=int(1e4 * 1000), help='Maximum number of steps')
# parser.add_argument('--update_freq', type=int, default=10, help='Update frequency')
# parser.add_argument('--sync_freq', type=int, default=10000, help='Sync frequency')
# parser.add_argument('--action_space', type=int, default=4, help='Action space size')

# parser.add_argument('--buffer_size', type=int, default=1000000, help='Replay buffer size')
# parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
# parser.add_argument('--epsilon_start', type=float, default=1.0, help='Initial epsilon for exploration')
# parser.add_argument('--epsilon_end', type=float, default=0.1, help='Final epsilon for exploration')
# parser.add_argument('--epsilon_decay_steps', type=float, default=250000, help='Epsilon decay rate')
# parser.add_argument('--learning_rate', type=float, default=0.00025, help='Learning rate for the optimizer')
# parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
# parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cpu or cuda)')


