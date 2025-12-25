# [ICLR'25] MDSGen: Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Generation
<br>

**[Trung X. Pham](https://trungpx.github.io/)<sup>1\*</sup>, [Tri Ton](https://triton99.github.io/)<sup>1\*</sup>, [Chang D. Yoo](https://sanctusfactory.com/family.php)<sup>1†</sup>** 
<br>
<sup>1</sup>KAIST, South Korea
<br>
\*Co-first authors (equal contribution), †Corresponding authors

<p align="center">
        <a href="https://triton99.github.io/mdsgen-site/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2410.02130" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2312.13528-b31b1b.svg">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/triton99/MDSGen">
</p>

## 📣 News
- **[02/2025]**: Code released.
- **[01/2025]**: MDSGen accepted to ICLR 2025 🎉.
- **[10/2024]**: Paper uploaded to arXiv. Check out the manuscript [here](https://arxiv.org/abs/2410.02130).(https://arxiv.org/abs/2410.02130).

## To-Dos
- [x] Release model weights on Google Drive.
- [x] Release inference code
- [ ] Release training code & dataset preparation

## ⚙️ Environmental Setups
1. Clone MDSGen.
```bash
git clone https://github.com/triton99/MDSGen
cd MDSGen
```

2. Create the environment.
```bash
conda create -n mdsgen python=3.8
conda activate mdsgen 
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
pip install -e .

```

## 📁 Data Preparations
T.B.D

## 🚀 Getting Started
### Trainning
T.B.D

### Download Checkpoints
The pretrained checkpoints can be downloaded on [Google Drive](https://drive.google.com/file/d/1XZoK-uU0mTWeRLilLZ_FtysRgearBxOp/view?usp=sharing).

### Inference
Before running inference, extract CAVP features from the video described in the [Diff-Foley](https://github.com/luosiallen/Diff-Foley/blob/main/inference/diff_foley_inference.ipynb).

To run the inference code, you can use the following command:
```bash
python inference.py --ckpt ./ckpts/mdsgen_audioldm.pt --video_feat_path ./sources/__2MwJ2uHu0_000004.npz

```

## 📖 Citing MDSGen

If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:

```bibtex
@inproceedings{pham2024mdsgenfastefficientmasked,
  title     = {MDSGen: Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Generation},
  author    = {Trung X. Pham and Tri Ton and Chang D. Yoo},
  year      = {2025},
  booktitle = {International Conference on Learning Representations},
}
```

## 🤗 Acknowledgements

Our code is based on [MDT](https://github.com/sail-sg/MDT?tab=readme-ov-file) and [Diff-Foley](https://github.com/luosiallen/Diff-Foley). We thank the authors for their excellent work!