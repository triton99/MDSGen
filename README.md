<h1 align="center"> MDSGen: <br>Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Generation
</h1>

<div align="center">
  <a href="https://trungpx.github.io/" target="_blank">Trung X.&nbsp;Pham</a><sup>1*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://triton99.github.io/" target="_blank">Tri&nbsp;Ton</a><sup>1*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://sanctusfactory.com/family.php" target="_blank">Chang D.&nbsp;Yoo</a><sup>1</sup> &ensp; </b> &ensp;
  <br>
  <sup>1</sup> KAIST &emsp; <br>
  <sup>*</sup>Equal Contribution &emsp; <br>
  <sup></sup> ICLR 2025 &emsp; <br>
</div>

### [Project Page](https://triton99.github.io/mdsgen-site/) | [Arxiv](https://arxiv.org/abs/2410.02130)

## TODO
- [ ] Release model weights on Google Drive.
- [x] Release inference code
- [ ] Release training code & dataset preparation

## Getting Started

### Installation

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

### Download Checkpoints

The pretrained checkpoints can be downloaded on Google Drive.

### Inference

```bash
python inference.py --ckpt ./ckpts/mdsgen_audioldm.pt --video_feat_path ./sources/__2MwJ2uHu0_000004.npz

```

## Citing MDSGen

If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:

```bibtex
@article{pham2024mdsgenfastefficientmasked,
  author    = {Trung X. Pham and Tri Ton and Chang D. Yoo},
  title     = {MDSGen: Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Generation},
  journal   = {arXiv preprint arXiv:2410.02130},
  year      = {2024}
}
```

## Acknowledgements

Our code is based on [MDT](https://github.com/sail-sg/MDT?tab=readme-ov-file) and [Diff-Foley](https://github.com/luosiallen/Diff-Foley). We thank the authors for their excellent work!


