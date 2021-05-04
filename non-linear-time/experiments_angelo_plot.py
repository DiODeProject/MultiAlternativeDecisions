import numpy as np
import matplotlib.pyplot as plt

x =[.3, .4, .5, .6] 
y =[2.141, 1.735, 1.548, 1.464] 

yerr = [0.201, 0.181, 0.168, 0.155]

fig, ax = plt.subplots()

ax.errorbar(x, y, yerr=yerr, color='k', linewidth=2)

ax.set_xlabel("stimuli's magnitude", fontsize=20)
ax.set_ylabel('reaction time', fontsize=20)
ax.set_xticks(x)
ticksize=15
ax.xaxis.set_tick_params(labelsize=ticksize)
ax.yaxis.set_tick_params(labelsize=ticksize)
fig.savefig("experimental_results.pdf", bbox_inches='tight')
