import os
from typing import List
import data
import pickle
from cycler import cycler

import scipy.io
import matplotlib.pyplot as plt
import matplotlib

# using Agg backend will not show figure
#matplotlib.use('Agg')

import numpy as np


def wrap_list(lst, items_per_line=10):
    """insert new line characters every items per line, to make printing of long lists nicer"""
    lines = []
    for i in range(0, len(lst), items_per_line):
        chunk = lst[i:i + items_per_line]
        line = ", ".join("{!r}".format(x) for x in chunk)
        lines.append(line)
    return "[" + ",\n ".join(lines) + "]"


def plot_ax(words: List[str], k: int, ax,bbox = (0., 1.1, 1., .102) ):
    """plot a list of word over time, the provided axes"""
    tokens = [vocab.index(w) for w in words]
    betas = [beta[k, :, x] for x in tokens]
    ax.set_prop_cycle(colorcycle)
    for i, comp in enumerate(betas):
        ax.plot(range(T), comp, label=words[i], lw=2, linestyle='--', marker='o', markersize=4)
    ax.legend(frameon=False, bbox_to_anchor=bbox, loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)
    print('np.arange(T)[0::2]: ', np.arange(T)[0::2])
    ax.set_xticks(np.arange(T)[0::2])
    ax.set_xticklabels(timelist[0::2])
    ax.set_title('Topic #'+str(k), fontsize=12)


def plot_words(words: List[str], k: int,bbox = (0., 1.1, 1., .102) ):
    """print a list of word over time in a new figure"""
    tokens = [vocab.index(w) for w in words]
    betas = [beta[k, :, x] for x in tokens]
    fig, ax = plt.subplots()
    ax.set_prop_cycle(colorcycle)
    for i, comp in enumerate(betas):
        ax.plot(range(T), comp, label=words[i], lw=2, linestyle='--', marker='o', markersize=4)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.5 ])

    ax.legend(frameon=False, bbox_to_anchor=bbox, loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)
    print('np.arange(T)[0::2]: ', np.arange(T)[0::2])
    ax.set_xticks(np.arange(T)[0::2])
    ax.set_xticklabels(timelist[0::2])
    ax.set_title('Topic #'+str(k), fontsize=12)


# trained topics
beta_file = './results/detm_un_K_50_Htheta_800_Optim_adam_Clip_2.0_\
ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_4_minDF_15_trainEmbeddings_0_beta'

beta = scipy.io.loadmat(beta_file)['values']  # K x T x V topic x time x vocab ( e.g. 50 x 14 x 10113)
print('beta: ', beta.shape)

# time stamps of documents e.g. ['2007', '2008', '2009', '2010',...]
with open('scripts/split_paragraph_False/min_df_15/timestamps.pkl', 'rb') as f:
    timelist = pickle.load(f)
print('timelist: ', timelist)
T = len(timelist)
ticks = [str(x) for x in timelist]
print('ticks: ', ticks)

# get dictionary of vocab  {  token <int>:word <str>, ...}
data_file = 'scripts/split_paragraph_False/min_df_15'
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)

# color order for multiple lines plot
colorcycle = cycler('color', ['blue', 'green', 'red', 'cyan', 'magenta',  'black', 'purple',
                                    'brown', 'orange', 'teal', 'coral', 'lime', 'lavender',
                                    'turquoise', 'tan', 'salmon', 'gold'])


# print out top words of topics
num_words = 100  # print out for each topic
time_index = [5]  # index of time to print out
num_topics = 50


keyword = 'xenon'  # 'axion', 'xenon', '1T'
keyword_max_prob = 0
keyword_topic = None

for k in range(num_topics):
    print((' -- topic {} -- '.format(k))*20)
    for t in time_index:
        gamma = beta[k, t, :]
        top_words = list(gamma.argsort()[-num_words+1:][::-1])  # get indices of sorted array
        topic_words = [vocab[a] for a in top_words]

        for a in list(gamma.argsort()):
            if keyword in vocab[a]:
                #print("found {} with prob {} ".format(vocab[a], gamma[a]))
                if gamma[a] >= keyword_max_prob:
                    keyword_max_prob = gamma[a]
                    keyword_topic = k


        #print('.. Time: {} ===> {}'.format(t, topic_words))
        print('time ', t, wrap_list(topic_words))

        plot_words([keyword],k)

    print('\n')

print('for the keyword {}, the max probability is {},'
      ' found in topic: {} at time: {} '.format(keyword,keyword_max_prob,keyword_topic,time_index) )


# ----- print top words of a chosen topic over all times --------#
chosenTopic = 46
print('Topic number ...'.format(chosenTopic))
num_words = 10
for t in range(14):
    gamma = beta[chosenTopic, t, :]
    top_words = list(gamma.argsort()[-num_words+1:][::-1])
    topic_words = [vocab[a] for a in top_words]
    print('Time: {} ===> {}'.format(t, topic_words))


# words_x = ['diphotons']
# tokens_x = [vocab.index(w) for w in words_x]
# betas_x = [beta[46, :, x] for x in tokens_x]
# for i, comp in enumerate(betas_x):
#     plt.plot(comp, label=words_x[i], lw=2, linestyle='--', marker='o', markersize=4)
# plt.legend(frameon=False, bbox_to_anchor=(0., -0.3, 1., .102), loc='lower left',
#                ncol=3, mode="expand", borderaxespad=0.)
# plt.set_xticks(np.arange(T)[0::2])
# plt.set_xticklabels(timelist[0::2])
# plt.set_title('Topic #46 - show only diphoton', fontsize=12)
# plt.show()


# --------- create a figure with 6 panels  ---------#
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
ax1, ax2, ax3, ax4, ax5, ax6= axes.flatten()
ticks = [str(x) for x in timelist]

# --------- plot panel by panel, supply the list of words manually  ---------#
words_0 = [ 'inflaton',  'two-higgs-doublet', 'topology','lep2','superluminal']
#,'observability','superluminal','superconductors','light-by-light','saxion','femtoscopic', 'deuteron',
plot_ax(words_0, 0, ax1)


words33 = ['750','excess','flavour','elastic',
           'normalization']
#'weyl','bottomonium','bispectrum','diphoton']
plot_ax(words33, 33, ax2)

words34 = ['750','curvaton','terrestrial', 'high-luminosity', 'rainbow-ladder', 'scale-invariant']
# ,'icecube', 'reggeon', 'baryogenesis', 'supercooled', 'coleman-weinberg', 'soft-photon']
#,'photon-photon','tev-scale','three-loop','monojet',
plot_ax(words34, 34, ax3)

words38 = ['higgs',     'superparticle','inelastic','neutronization',
]
#'weyl','bottomonium','bispectrum','diphoton','quark-antiquark', 'neutrino-proton','qcd','singlet','radiation']
plot_ax(words38, 38, ax4,bbox = (0., -0.4, 1., .102))


words41 = [ 'string','susy','one-loop','two-loop','dirac','star'
]
plot_ax(words41, 41, ax5,bbox = (0., -0.4, 1., .102))
# highest higgs 43

words48 = ['cosmic', 'hadronic',
  'dipole',  'gravity',
 'massless',  'proton']
# 'higgs' sharpest 45,
plot_ax(words48, 48, ax6, bbox = (0., -0.4, 1., .102))


# w48 = [
#  'quintom',  'antiproton',
#  'higgs',  'isotopic']
# # 'perseus',  'majorana','semi-leptonic', 'soliton',  'kimber-martin-ryskin',  'p-wave']
# #'lattice',  'branching',   'yukawa', 'singlet', 'd3-branes', '3-body', 'twist-four','electron-nucleon', 'nodal'
# plot_ax(w48, 48, ax7, bbox = (0., -0.4, 1., .102))

plt.savefig( beta_file + '.png',bbox_inches='tight')
plt.show()
command = 'open ' + beta_file + '.png'
os.system(command)


# --------- create a figure with 6 panels  ---------#
fig2, axes2 = plt.subplots(nrows=2, ncols=3, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
ax21, ax22, ax23, ax24, ax25, ax26 = axes.flatten()
ticks = [str(x) for x in timelist]

# --------- plot panel by panel, supply the list of words manually  ---------#
words_0 = [ 'inflaton',  'two-higgs-doublet', 'topology','lep2','superluminal']
#,'observability','superluminal','superconductors','light-by-light','saxion','femtoscopic', 'deuteron',
plot_ax(words_0, 0, ax21)
