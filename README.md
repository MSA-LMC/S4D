# [Static for Dynamic: Towards a Deeper Understanding of Dynamic Facial Expressions Using Static Expression Data](https://arxiv.org/pdf/2409.06154)

## News

[2025.9.15] The paper is accepted by IEEE Transactions on Affective Computing.

[2024.9.5] Code and pre-trained models will be released here.

## Pre-Train

```bash
cd pretrain/omnivision &&  OMP_NUM_THREADS=1 HYDRA_FULL_ERROR=1 python train_app_submitit.py +experiments=videomae/videomae_base_vox2_affectnet
```
## Fine-Tune

```bash
cd finetune && bash run.sh
```



## Results

<img width="1024" alt="image" src="https://github.com/user-attachments/assets/31b131e1-6530-4486-9bb4-a006fe464d32" />




## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MSA-LMC/S4D&type=Date)](https://star-history.com/#MSA-LMC/S4D&Date)
