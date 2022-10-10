import os
import json
import numpy as np
import matplotlib.pyplot as plt


logFileDir = "./logs/BASEMODEL_train_20221006-104034.log"

lossDict = {"g_adv_loss_latent" : [],
            "g_sty_loss_latent": [],
            "g_ds_loss_latent": [],
            "g_cyc_loss_latent": [],
            "g_loss_latent": [],
            "g_adv_loss_ref": [],
            "g_sty_loss_ref": [],
            "g_ds_loss_ref": [],
            "g_cyc_loss_ref": [],
            "g_loss_ref": [],
            "d_adv_loss_latent": [],
            "d_loss_latent": [],
            "d_adv_loss_ref": [],
            "d_loss_ref": []}

xs = []
x0 = 1
if os.path.splitext(logFileDir)[1] == ".log":
  
  with open(logFileDir, "r") as logf:
    for ln in logf:
      try:
        v0 = float(ln.strip().split(":")[-1].replace('"','').strip(","))
      except:
        continue

      k0 = ln.strip().split(":")[0].replace('"','')
      if k0 == "iter":
        xs.append(x0)
        x0 += 1
      if k0 in lossDict.keys():
        lossDict[k0].append(v0)
else:
  raise ValueError


for k0, llist in lossDict.items():
  plt.figure(figsize = (21,7))
  plt.xlim([0, len(xs)])
  plt.ylim([min(llist), max(llist)])
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  #npllist = np.array(llist)
  #npllist /= np.max(npllist)
  plt.plot(xs, llist, label = k0, lw = 2)
  plt.grid()
  plt.legend(loc="lower right")
  plt.savefig(f"./testModel1/logGraph/log_{k0}.png")
  plt.close()

