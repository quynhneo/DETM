# DETM modified to train on arxiv abstracts 

We modify the code to train on abstracts of papers from arxiv.org.

In `main.py`:

We try to match the settings from Dieng's paper (https://arxiv.org/abs/1907.05545), fix run time errors, add comments.

In `scripts/data_undebates`:

We modify so that it can process the meta data json file contains arxiv abstracts, with the option to select a category (default category: high energy physics phenomenonlogy `hep-ph`).

New file: `requirements.txt`.

## Results
The plot below shows results for DETM on [`hep-ph`](https://arxiv.org/archive/hep-ph) (high energy physics phenomenology) category. Six topics out of 50 are shown here. For each topics, probabilities of some selected words are plotted against time (2007-2020). In topics #33 and #34, peak probability of the word `750` coincides with the [flurry of papers](https://en.wikipedia.org/wiki/750_GeV_diphoton_excess) on a possible discovery of new physics around 2015-2016, which turned out to be just a statistical fluke. Topic 38 shows the increase in `higgs` around the time of the  discovery of Higgs boson in 2012.


![result](https://github.com/quynhneo/detm-arxiv/blob/master/detm_un_K_50_Htheta_800_Optim_adam_Clip_2.0_ThetaAct_relu_Lr_0.001_Bsz_200_RhoSize_300_L_4_minDF_15_trainEmbeddings_0_beta.png)

Other steps involving, from getting data, preprocessing, embedding words and runtime, etc can be found here: https://github.com/quynhneo/detm-arxiv
