import os
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

interbal = 5

fig = plt.figure(figsize=(4, 6))
fig.canvas.set_window_title("@74_suke")
ax_1 = fig.add_subplot(2, 1, 1)
ax_2 = fig.add_subplot(2, 1, 2)
ax_3 = ax_2.twinx()

ax_1.set_title("score transition")
ax_1.set_xlabel("generation")
ax_1.set_ylabel("score")

ax_2.set_title("cosine similarity")
ax_2.set_xlabel("generation")
ax_2.set_ylabel("cosSim_average")

ax_3.set_ylabel("cosSim_variance")
plt.tight_layout()

with open(path, "r") as file:
    dict = json.load(file)

score_log = np.array(dict["score_log"])
genera_listA = np.array(range(len(score_log)))

simAve_log = np.array(dict["simAve_log"])
simVari_log = np.array(dict["simVari_log"])
genera_listB = np.array(range(len(simAve_log)))

lines_1, = ax_1.plot(genera_listA, score_log, label="score", marker="o", markersize=1.5, linewidth=1)
lines_2, = ax_2.plot(genera_listB, simAve_log, color="red", label="cosSim_average", marker="o", markersize=1.5, linewidth=1)
lines_3, = ax_3.plot(genera_listB, simVari_log, color="green", label="cosSim_variance", marker="o", markersize=1.5, linewidth=1)

xlim_1 = min(genera_listA)
ylim_1 = min_ylim(score_log)

xlim_2 = min(genera_listB)
ylim_2 = min_ylim(simAve_log)
ylim_3 = min_ylim(simVari_log)

ax_1.set_xlim(xlim_1, max(genera_listA)+0.5)
ax_1.set_ylim(ylim_1, np.percentile(score_log, 75)*2)

ax_2.set_xlim(xlim_2, max(genera_listB)+0.5)
ax_2.set_ylim(ylim_2, np.percentile(simAve_log, 75)*2)
ax_3.set_xlim(xlim_2, max(genera_listB)+0.5)
ax_3.set_ylim(ylim_3, np.percentile(simVari_log, 75)*2)

ax_1.legend()
ax_2.legend(loc=(0.01, 0.9))
ax_3.legend(loc=(0.01, 0.8))

print("generation_spawnClone:", ", ".join([str(genera) for genera in dict["genera_clone"]]), end="")
len_generClone = len(dict["genera_clone"])

old_mtime = os.stat(path).st_mtime
while True:
    mtime = os.stat(path).st_mtime
    if mtime != old_mtime:
        with open(path, "r") as file:
            dict = json.load(file)
        score_log = np.array(dict["score_log"])
        genera_listA = np.array(range(len(score_log)))
        simAve_log = np.array(dict["simAve_log"])
        simVari_log = np.array(dict["simVari_log"])
        genera_listB = np.array(range(len(simAve_log)))

        lines_1.set_data(genera_listA, score_log)
        lines_2.set_data(genera_listB, simAve_log)
        lines_3.set_data(genera_listB, simVari_log)

        genera_maxB = max(genera_listB)
        ax_1.set_xlim(xlim_1, max(genera_listA)+0.5)
        ax_1.set_ylim(min_ylim(score_log), np.percentile(score_log, 75)*2)

        ax_2.set_xlim(xlim_2, genera_maxB+0.5)
        ax_2.set_ylim(min_ylim(simAve_log), np.percentile(simAve_log, 75)*2)
        ax_3.set_xlim(xlim_2, genera_maxB+0.5)
        ax_3.set_ylim(min_ylim(simVari_log), np.percentile(simVari_log, 75)*2)

        old_mtime = mtime
        if len(dict["genera_clone"]) != len_generClone:
            for genera in dict["genera_clone"][len_generClone:]:
                print(" ,", genera, end="")
            len_generClone = len(dict["genera_clone"])
    plt.pause(interbal)
