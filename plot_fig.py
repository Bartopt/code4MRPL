import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
fontSize = 20

def load_ret(expdir, ret_file):
    with open(os.path.join(expdir, '{}.pkl'.format(ret_file)), 'rb') as f:
        data = pickle.load(f)
    return data

def PlotStd(x, data, color, label, xOffset=0, alpha=0.1):
    # todo: using seaborn instead
    m = np.mean(data, axis=0)
    # std = np.std(data, axis=0)
    # r1 = m + std
    # r2 = m - std
    # plt.plot(x + xOffset, m, color=color, linewidth=4, label=label)
    # plt.fill_between(x + xOffset, r1, r2, color=color, alpha=alpha)
    plt.plot(x + xOffset, m, color=color, linewidth=4, label=label)

# edit path
expdir1 = 'output/pih-meta/MRPL-2/eval_trajectories/' # directory to load data from

y1 = load_ret(expdir1, 'ret_demo1_acctextTrue_meanFalse').squeeze()
y2 = load_ret(expdir1, 'ret_demo0_acctextTrue_meanFalse').squeeze()


plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams.update({"font.size": fontSize})
plt.xlabel("Test-time rollout numbers", fontsize=fontSize+10)
plt.ylabel("Return", fontsize=fontSize+10)

x = np.arange(y1.shape[1])
PlotStd(x, y1, 'limegreen', 'MRPL', 1, alpha=0.1)
PlotStd(x, y2, 'red', 'MRPL-NoDemo', 1, alpha=0.1)
plt.legend()
plt.show()
# f = plt.gcf()
# f.savefig('tmp.png')
