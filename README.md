# XLSum
XLSum: Fine-tuning [XLNet](https://arxiv.org/abs/1906.08237) for Extractive Summarization

📍 **XLNet** is Google's recent unsupervised language model based on the generalized permutation modeling objective. The XLNet model is auto-regressive and uses [Transformer-XL](https://arxiv.org/abs/1901.02860) as its basis, which means that there is no limit to the length of the text sequence that XLNet can process. The original code for XLNet could be found in the below links.

* [XLNet Tensorflow Version](https://github.com/zihangdai/xlnet)
* [XLNet PyTorch Version](https://github.com/huggingface/pytorch-transformers)

**Python Requirements** : This code is written in Python 3.6

## Data Preprocessing

XLSum uses CNN/Daily Mail articles to fine-tune the model. Please follow the instructions below to download the data, and to preprocess the data for model training using the provided code.

#### **Step 1**
Download the CNN and Daily Mail `stories` directory from this [link](https://cs.nyu.edu/~kcho/DMQA/), the DeepMind Q&A Dataset. Unzip all of the story files into the `raw_data/raw_files` directory.

#### **Step 2**
Run the following code:
```
python preprocess.py
```
The files will be saved to `xlnet_data`. You can delete all other files after preprocessing is done.
