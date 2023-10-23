import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
# Change this to file's current directory
cur_dir = "C:\\Users\\Orr Bezalely\\Desktop\\KekeCompetition-main\\Keke_JS\\Experiments"
# Change experiment variable to either 1, 2, or 3 depending on experiment number (3 representing scaled-down env)
experiment = 2
# Change moving_avg variable to True or False depending on whether you want a smoothed or raw graph respectively
moving_avg = True


# Based on code from https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/
def moving_average(lst, window_size):
    arr = lst[:window_size // 2] + lst + lst[-window_size//2:]
    i = 0
    moving_averages = []
    while i < len(arr) - window_size:
        window_average = np.sum(arr[i:i + window_size]) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages



# Sets directory of files, line colours, styles and line thickness of lines based on experiment number
if experiment == 1:
    inside_dir = "\\EXP1"
    episodes = "15,000"
    colours = ["blue", "black", "orange", "green", "red"]
    style = ["solid", (0,(4,10)), "solid", (0,(4,10)), (7,(4,10))]
    line_thickness = [2.5,1,2.5,1,1]
elif experiment == 2:
    inside_dir = "\\EXP2"
    episodes = "15,000"
    #  betweenness, closeness, degree, handcrafted, optimal, ozgur, Q-learning, random
    colours = ["red", "green", "orange", "purple", "blue", "black", "brown", "cyan", "violet", "indigo", "yellow"]
    style = ["solid", "solid", "dashed", "solid", "solid", (0,(4,10)), "dashed", (0,(4,10)), "--", "--", "--"]
    line_thickness = [2.5,2.5,1,2.5,2.5,1,1,1,1,1,1]
elif experiment == 3:
    inside_dir = "\\EXP3"
    episodes = "10,000"
    colours = ["blue", "black", "orange", "green", "red"]
    style = ["solid", (0,(4,10)), "solid", (0,(4,10)), (7,(4,10))]
    line_thickness = [2.5,1,2.5,1,1]
else:
    assert(False)

full_dir = cur_dir + inside_dir
directory = os.fsencode(full_dir)

# Sets moving average to either 5 or 1 depending on whether the moving average wanted
if moving_avg:
    moving_average_num = 5
else:
    moving_average_num = 1

# Plots graph and lines
fig, ax = plt.subplots(figsize=(11, 9))
c_i = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    f = open(full_dir + "\\" + filename, "r")
    average_rewards_lst_q = json.load(f)
    f.close()
    n_episodes_q = len(average_rewards_lst_q[0])
    average_rewards_q = np.array(average_rewards_lst_q).mean(axis=0)
    new_rewards_lst = np.array(moving_average(list(average_rewards_q), moving_average_num))
    ax.plot(range(n_episodes_q), new_rewards_lst, label=filename, c=colours[c_i], linestyle=style[c_i], linewidth=line_thickness[c_i])
    c_i += 1

# Sets graph title
if moving_avg:
    ax.set_title(f"Experiment {experiment} - Average Rewards of Various Agents Over {episodes} Episodes in Augmented Two-Room Environment During Training (Averaged Over 100 Trials, {moving_average_num}-Point Moving Average)",  wrap=True, fontsize=20)
else:
    ax.set_title(f"Experiment {experiment} - Average Rewards of Various Agents Over {episodes} Episodes in Augmented Two-Room Environment During Training (Averaged Over 100 Trials)",  wrap=True, fontsize=20)

# Sets graph's axis labels
ax.set_xlabel("Episodes", fontsize=16)
ax.set_ylabel("Average Rewards", fontsize=16)

# Sets graph's legend
if experiment == 2 or experiment == 3:
    ax.legend(bbox_to_anchor=(0.6,0.7), fancybox=True, shadow=True, fontsize=12)
else:
    ax.legend(loc="center right", fancybox=True, shadow=True, fontsize=12)

plt.grid()
plt.show()
