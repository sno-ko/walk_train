import json
import numpy as np
import matplotlib.pyplot as plt
import sys

def min_ylim(list):
    min_value = min(list)
    if min_value >= 0:
        return min_value * 0.8
    else:
        return min_value * 1.8


args = sys.argv
if len(args) == 1:
    path = "data/learn_data.json"
else:
    path = args[1]

fig = plt.figure(figsize=(6, 6))
fig.canvas.set_window_title("@74_suke")
ax_1 = fig.add_subplot(2, 1, 1)
ax_2 = fig.add_subplot(2, 1, 2)
ax_3 = ax_2.twinx()

ax_1.set_title("best score")
ax_1.set_xlabel("generation")
ax_1.set_ylabel("score")

ax_2.set_title("cosine similarity")
ax_2.set_xlabel("generation")
ax_2.set_ylabel("cosSim_mean")

ax_3.set_ylabel("cosSim_variance")

with open(path, "r") as file:
    dict = json.load(file)

score_log = np.array(dict["score_log"])
genera_listA = np.array(range(len(score_log)))

simMean_log = np.array(dict["simMean_log"])
simVari_log = np.array(dict["simVari_log"])
genera_listB = np.array(range(len(simMean_log)))

ax_1.plot(genera_listA, score_log, label="score", marker="o", markersize=1.5, linewidth=1)
ax_2.plot(genera_listB, simMean_log, color="red", label="cosSim_mean", marker="o", markersize=1.5, linewidth=1)
ax_3.plot(genera_listB, simVari_log, color="green", label="cosSim_variance", marker="o", markersize=1.5, linewidth=1)

ylim_1 = min_ylim(score_log)
ylim_2 = min_ylim(simMean_log)
ylim_3 = min_ylim(simVari_log)

ax_1.set_xlim(min(genera_listA), max(genera_listA)-1.01)
ax_1.set_ylim(ylim_1, np.percentile(score_log, 75)*2)

ax_2.set_xlim(min(genera_listB), max(genera_listB)-1.01)
ax_2.set_ylim(ylim_2, np.percentile(simMean_log, 75)*2)
ax_3.set_xlim(min(genera_listB), max(genera_listB)-1.01)
ax_3.set_ylim(ylim_3, np.percentile(simVari_log, 75)*2)

ax_1.legend()
ax_2.legend(loc=(0.01, 0.9))
ax_3.legend(loc=(0.01, 0.8))

print("generation_spawnClone:", dict["genera_clone"])

plt.tight_layout()
plt.show()
