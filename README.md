# ROAR - An NLP/Speech Processing Toolkit for Indic Languages

## Introduction

ROAR is a toolkit for Indic languages that provides a unified interface for various NLP and Speech Processing tasks. It is built on top of [PyTorch](https://pytorch.org/) (WIP). It is based on a lean fork of Nvidia's [NeMo](https://github.com/NVIDIA/NeMo) toolkit. Majorly targeting Text-to-Speech Synthesis ROAR aims to streamline developer workflow from implementing new models with pre-implemented modularized functions that can be re-used across projects/models.

## Instructions

To setup the toolkit, clone the repository and follow the instructions given below.

```
conda create -n roar python=3.10
pip install Cython
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

TinyLLaMa specific

```
conda install xformers -c xformers
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../.. && rm -rf flash-attention
```

After installing the requirements, 

### Step 1:

```
python scripts/dataset_processing/tts/extract_sup_data.py \
    manifest_filepath=<manifest_filepath> \
    sup_data_path=<sup_data_path> \
    dataset.sample_rate=<sample_rate> \
    dataset.n_fft=<n_fft> \
    dataset.win_length=<win_length> \
    dataset.hop_length=<hop_length> \
```

### Step 2:

```
python examples/tts/fastpitch.py \
    --config-name=fastpitch_22050_align \
    train_dataset=<train_dataset_path> \
    validation_datasets=<validation_dataset_path> \
    sup_data_types="['align_prior_matrix', 'pitch', 'speaker_id']" \
    sup_data_path=<sup_data_path> \
    pitch_mean=<pitch_mean> \
    pitch_std=<pitch_std> \
    +exp_manager.create_wandb_logger=True \
    +exp_manager.wandb_logger_kwargs.name="tutorial-FastPitch-multispeaker" \
    +exp_manager.wandb_logger_kwargs.project="NeMo" \
```

A more in-depth tutorial will be added soon.

## ToDo:
- [x] Core NeMo Fork
- [x] NeMo Utils Fork
- [x] Vanilla MHA
- [x] Flash Attention
- [x] TransformerLayer
- [x] One TTS Alignment
- [x] Fastpitch
- [x] HiFi-GAN
- [x] JETS
- [x] ConformerLayer
- [x] Rotary Position Embeddings
- [ ] TinyLLaMa

---
**Disclaimer**: All the code in this repository, forked from NeMo, follows the licensing of the original NeMo repository. The code is provided as-is for research purposes only and without any guarantees. Please contact the original authors for any commercial use.