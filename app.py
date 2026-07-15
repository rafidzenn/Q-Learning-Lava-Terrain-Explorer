import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
import io
from PIL import Image

GRID_SIZE = 5
NUM_LAVA = 3
NUM_GAS = 2
NUM_CRIERS = 2
MAX_EPISODES = 100
MAX_STEPS = 100
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.8
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.980

grid = np.zeros((GRID_SIZE, GRID_SIZE))
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))

def place_hazards():
    global grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    placed = set()
    def get_unique_position():
        while True:
            pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if pos not in placed and pos != (GRID_SIZE-1, GRID_SIZE-1):
                placed.add(pos)
                return pos
    for _ in range(NUM_LAVA):
        x, y = get_unique_position()
        grid[x, y] = 1
    for _ in range(NUM_GAS):
        x, y = get_unique_position()
        grid[x, y] = 2
    for _ in range(NUM_CRIERS):
        x, y = get_unique_position()
        grid[x, y] = 3

def get_reward(state):
    x, y = state
    if grid[x, y] == 1: return -100
    elif grid[x, y] == 2: return -50
    elif grid[x, y] == 3: return -80
    else: return 10

def get_possible_actions():
    return ['up', 'down', 'left', 'right']

def move(state, action):
    x, y = state
    if action == 'up': x = max(x-1, 0)
    elif action == 'down': x = min(x+1, GRID_SIZE-1)
    elif action == 'left': y = max(y-1, 0)
    elif action == 'right': y = min(y+1, GRID_SIZE-1)
    return (x, y)

def q_learning():
    global Q_table, EPSILON
    EPSILON = 1.8
    Q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    place_hazards()
    episodes = 0
    total_rewards = []
    while episodes < MAX_EPISODES:
        state = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < MAX_STEPS:
            steps += 1
            if random.uniform(0, 1) < EPSILON:
                action = random.choice(get_possible_actions())
            else:
                action = get_possible_actions()[np.argmax(Q_table[state[0], state[1]])]
            next_state = move(state, action)
            reward = get_reward(next_state)
            total_reward += reward
            action_index = get_possible_actions().index(action)
            best_next = np.max(Q_table[next_state[0], next_state[1]])
            Q_table[state[0], state[1], action_index] = \
                (1 - ALPHA) * Q_table[state[0], state[1], action_index] + \
                ALPHA * (reward + GAMMA * best_next)
            state = next_state
            if state == (GRID_SIZE-1, GRID_SIZE-1):
                done = True
        episodes += 1
        total_rewards.append(total_reward)
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(EPSILON, EPSILON_MIN)
    return total_rewards

def run_simulation():
    total_rewards = q_learning()
    path = [(0, 0)]
    state = (0, 0)
    for _ in range(50):
        action_idx = np.argmax(Q_table[state[0], state[1]])
        action = get_possible_actions()[action_idx]
        next_state = move(state, action)
        path.append(next_state)
        state = next_state
        if state == (GRID_SIZE-1, GRID_SIZE-1):
            break

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cmap = plt.cm.get_cmap('viridis', 4)
    im = axes[0].imshow(grid.T, cmap=cmap, vmin=0, vmax=3)
    xs, ys = zip(*path)
    axes[0].plot(ys, xs, 'r-o', linewidth=2, markersize=8)
    axes[0].plot(ys[0], xs[0], 'go', markersize=12, label='Start')
    axes[0].plot(ys[-1], xs[-1], 'b*', markersize=15, label='End')
    plt.colorbar(im, ax=axes[0], ticks=[0, 1, 2, 3])
    axes[0].set_title("Agent's Navigation Path")
    axes[0].legend()

    axes[1].plot(total_rewards)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Reward')
    axes[1].set_title('Training Rewards per Episode')
    axes[1].grid(True)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

with gr.Blocks(title="Autonomous Volcanic Terrain Explorer") as demo:
    gr.Markdown("# 🌋 Autonomous Exploration System for Simulated Volcanic Terrain")
    gr.Markdown("Uses **Q-Learning (MDP)** to train a robot to navigate volcanic terrain with lava, gas, and craters.")
    gr.Markdown("Click **Run Simulation** to train the agent and visualize its path!")
    run_btn = gr.Button("🚀 Run Simulation", variant="primary")
    output_img = gr.Image(label="Results", type="pil")
    run_btn.click(fn=run_simulation, outputs=output_img)

demo.launch()
