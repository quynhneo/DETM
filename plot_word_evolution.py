import scipy.io
import matplotlib.pyplot as plt
from cycler import cycler
import data
import pickle
import numpy as np
beta_file = './results/detm_un_K_50_Htheta_800_Optim_adam_Clip_2.0\
_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_4_minDF_30_trainEmbeddings_1_beta'

beta = scipy.io.loadmat(beta_file)['values']  # K x T x V topic x time x vocab ( e.g. 50 x 14 x 10113)
print('beta: ', beta.shape)

with open('scripts/split_paragraph_False/min_df_30/timestamps.pkl', 'rb') as f:
    timelist = pickle.load(f)
print('timelist: ', timelist)
T = len(timelist)
ticks = [str(x) for x in timelist]
print('ticks: ', ticks)

## get vocab
data_file = 'scripts/split_paragraph_False/min_df_30'
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)

## plot topics
num_words = 20
times = [0, 7, 13]
num_topics = 50
for k in range(num_topics):
    print(('topic {}'.format(k))*20)
    for t in times:
        gamma = beta[k, t, :]
        top_words = list(gamma.argsort()[-num_words+1:][::-1])  #get indice of sorted array
        topic_words = [vocab[a] for a in top_words]
        for a in list(gamma.argsort()):
            if 'diphoton' in vocab[a]:
                print("found {} with prob {} ".format(vocab[a], gamma[a]))

        print('.. Time: {} ===> {}'.format(t, topic_words))
    print('\n')

print('Topic Climate Change...')
num_words = 10
for t in range(14):
    gamma = beta[46, t, :]
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

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
ticks = [str(x) for x in timelist]
#plt.xticks(np.arange(T)[0::2], timelist[0::2])

words_1 = ['temperature', 'magnetic', 'phase', 'potential',  'gravitational', 'transition', 'string',
           'inflation', 'cosmological']
tokens_1 = [vocab.index(w) for w in words_1]
betas_1 = [beta[0, :, x] for x in tokens_1]
for i, comp in enumerate(betas_1):
    ax1.plot(range(T), comp, label=words_1[i], lw=2, linestyle='--', marker='o', markersize=4)
ax1.legend(frameon=False, bbox_to_anchor=(0., 1.1, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)

print('np.arange(T)[0::2]: ', np.arange(T)[0::2])
ax1.set_xticks(np.arange(T)[0::2])
ax1.set_xticklabels(timelist[0::2])
ax1.set_title('Topic #0', fontsize=12)


# words_5 = ['neutrino', 'symmetry', 'gauge', 'scalar', 'masses', 'mixing',
           #'supersymmetry', 'flavor', 'dark', 'cp', 'higgs']
words_5 = ['diphotons','photon']
tokens_5 = [vocab.index(w) for w in words_5]
betas_5 = [beta[3, :, x] for x in tokens_5]
# ax2.set_prop_cycle(cycler('color', ['blue', 'green', 'red', 'cyan', 'magenta',  'black', 'purple',
#                                     'brown', 'orange', 'teal', 'coral', 'lime', 'lavender',
#                                     'turquoise', 'tan', 'salmon', 'gold']))
for i, comp in enumerate(betas_5):
    ax2.plot(comp, label=words_5[i], lw=2, linestyle='--', marker='o', markersize=4)

ax2.legend(frameon=False,bbox_to_anchor=(0., 1.1, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)
ax2.set_xticks(np.arange(T)[0::2])
ax2.set_xticklabels(timelist[0::2])
ax2.set_title('Topic #3  ', fontsize=12)


words_11= ['neutrino', 'dark', 'neutrinos', 'cosmic', 'solar', 'oscillations', 'atmospheric' ]
tokens_11 = [vocab.index(w) for w in words_11]
betas_11 = [beta[22, :, x] for x in tokens_11]
for i, comp in enumerate(betas_11):
    ax3.plot(comp, label=words_11[i], lw=2, linestyle='--', marker='o', markersize=4)
ax3.legend(frameon=False,bbox_to_anchor=(0., 1.1, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)
ax3.set_xticks(np.arange(T)[0::2])
ax3.set_xticklabels(timelist[0::2])
ax3.set_title('Topic #22', fontsize=12)


words_13 = ['gauge', 'effective', 'qcd', 'terms', 'method', 'corrections','renormalization']
tokens_13 = [vocab.index(w) for w in words_13]
betas_13 = [beta[43, :, x] for x in tokens_13]
for i, comp in enumerate(betas_13):
    ax4.plot(comp, label=words_13[i], lw=2, linestyle='--', marker='o', markersize=4)
ax4.legend(frameon=False,bbox_to_anchor=(0., -0.3, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)
ax4.set_xticks(np.arange(T)[0::2])
ax4.set_xticklabels(timelist[0::2])
ax4.set_title('Topic #43', fontsize=12)


#words_28 = ['higgs', 'lhc', 'boson', 'collider', 'decay', 'production', 'decays', 'electroweak', 'mssm']
words_28 = ['diphotons','photon']
tokens_28 = [vocab.index(w) for w in words_28]
betas_28 = [beta[46, :, x] for x in tokens_28]
for i, comp in enumerate(betas_28):
    ax5.plot(comp, label=words_28[i], lw=2, linestyle='--', marker='o', markersize=4)
ax5.legend(frameon=False,bbox_to_anchor=(0., -0.3, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)
ax5.set_xticks(np.arange(T)[0::2])
ax5.set_xticklabels(timelist[0::2])
ax5.set_title('Topic #46', fontsize=12)


words_30 = ['quark', 'decay', 'pi', 'decays', 'states',  'form', 'chiral','qcd','meson', 'pion','lattice']
tokens_30 = [vocab.index(w) for w in words_30]
betas_30 = [beta[47, :, x] for x in tokens_30]
ax6.set_prop_cycle(cycler('color', ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple',
                                    'brown', 'orange', 'teal', 'coral', 'lime', 'lavender',
                                    'turquoise', 'tan', 'salmon', 'gold']))

for i, comp in enumerate(betas_30):
    ax6.plot(comp, label=words_30[i], lw=2, linestyle='--', marker='o', markersize=4)
ax6.legend(frameon=False,bbox_to_anchor=(0., -0.35, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)
ax6.set_xticks(np.arange(T)[0::2])
ax6.set_xticklabels(timelist[0::2])
ax6.set_title('Topic #47', fontsize=12)
"""
words_46 = ['mass', 'neutrino', 'quark', 'energy', 'decay',  'pi', 'standard']
tokens_46 = [vocab.index(w) for w in words_46]
betas_46 = [beta[46, :, x] for x in tokens_46]
for i, comp in enumerate(betas_46):
    ax7.plot(comp, label=words_46[i], lw=2, linestyle='--', marker='o', markersize=4)
ax7.legend(frameon=False)
ax7.set_xticks(np.arange(T)[0::2])
ax7.set_xticklabels(timelist[0::2])
ax7.set_title('Topic 19', fontsize=12)


words_49= ['magnetic', 'phase', 'spin',  'topological', 'temperature', 'transition', 'field', 'properties']
tokens_49 = [vocab.index(w) for w in words_49]
betas_49 = [beta[49, :, x] for x in tokens_49]
for i, comp in enumerate(betas_49):
    ax8.plot(comp, label=words_49[i], lw=2, linestyle='--', marker='o', markersize=4)
ax8.legend(frameon=False)
ax8.set_title('Topic #20', fontsize=12)
ax8.set_xticks(np.arange(T)[0::2])
ax8.set_xticklabels(timelist[0::2])
"""
plt.savefig( beta_file + '.png',bbox_inches='tight')
#plt.show()
