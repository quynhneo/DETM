# DETM modified to train on arxiv abstracts 

We modify the code to train on abstracts of papers from arxiv.org.

In `main.py`:

We try to match the settings from Dieng's paper (https://arxiv.org/abs/1907.05545), fix run time errors, add comments.

In `scripts/data_undebates`:

We modify so that it can process the meta data json file contains arxiv abstracts, with the option to select a category (default category: high energy physics phenomenonlogy `hep-ph`).

New file: `requirements.txt`.


Other steps involving, from getting data, preprocessing, embedding words and results can be found here: https://github.com/quynhneo/detm-arxiv

