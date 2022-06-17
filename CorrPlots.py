import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


flatchain = pd.read_csv(
    "/Users/buuch/Dropbox (Privat)/_UZH/_MasterThesis/MicrobialGrowth/csv files/PMsyn_flatchain.csv")
var4 = ["mu_a","sigma_a","mu_b","sigma_b"]
var3 = ["o2","u","v"]

sns.set_palette(["#7a7aff"])
g = sns.PairGrid(flatchain,corner=True)
g.map_lower(sns.scatterplot,color="C1")
g.map_lower(sns.kdeplot,levels=5,color="firebrick",alpha=0.6)
g.map_diag(sns.histplot,color="C1")
#g.set(xticklabels=[]) # remove tick labels for full corrplots
#g.set(yticklabels=[])
axes = []
for ax in g.axes.ravel():
    if ax != None:
        ax.xaxis.set_label_text("")
        ax.yaxis.set_label_text("")
        axes.append(ax)


# add xlabels ans ylabels
#axes[1].yaxis.set_label_text(r"$u\:[\mu m]$")
#axes[3].yaxis.set_label_text(r"$v\:[\mu m]$")
#axes[3].xaxis.set_label_text(r"$\omega_2\:[\frac{1}{h}]$")
#axes[4].xaxis.set_label_text(r"$u\:[\mu m]$")
#axes[5].xaxis.set_label_text(r"$v\:[\mu m]$")

#axes[6].yaxis.set_label_text(r"$\sigma_\beta$")
#axes[6].xaxis.set_label_text(r"$\mu_\alpha\:[\frac{1}{h}]$")
#axes[7].xaxis.set_label_text(r"$\sigma_\alpha$")
#axes[8].xaxis.set_label_text(r"$\mu_\beta$")
#axes[9].xaxis.set_label_text(r"$\sigma_\beta$")

# adding all fixed values to histogram
#axes[0].axvline(x=1.4,c="firebrick",linewidth=2)
#axes[2].axvline(x=0.05,c="firebrick",linewidth=2)
#axes[5].axvline(x=0.49,c="firebrick",linewidth=2)
#axes[9].axvline(x=0.002,c="firebrick",linewidth=2)

# adding the crosses for the fixed values
#axes[1].scatter(x=1.4,y=0.05,color="firebrick",s=100,marker="+",linewidth=2)
#axes[3].scatter(x=1.4,y=0.5,color="firebrick",s=100,marker="+",linewidth=2)
#axes[6].scatter(x=1.4,y=0.002,color="firebrick",s=100,marker="+",linewidth=2)
#axes[4].scatter(x=0.05,y=0.5,color="firebrick",s=100,marker="+",linewidth=2)
#axes[7].scatter(x=0.05,y=0.002,color="firebrick",s=100,marker="+",linewidth=2)
#axes[8].scatter(x=0.49,y=0.002,color="firebrick",s=100,marker="+",linewidth=2)

plt.show()

g.savefig("PMsyn_corr17.png")