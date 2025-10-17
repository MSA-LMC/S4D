# [Static for Dynamic: Towards a Deeper Understanding of Dynamic Facial Expressions Using Static Expression Data](https://arxiv.org/pdf/2409.06154)
<img width="1024" height="506" alt="image" src="https://github.com/user-attachments/assets/db750330-84e2-4128-96c3-77c4a8fdc76c" />

## 📰 News


[2025.9.17] The code and pre-trained models are available.

[2025.9.15] The paper is accepted by IEEE Transactions on Affective Computing.

~~[2024.9.5] Code and pre-trained models will be released here.~~

## 🚀 Main Results

<img width="1024" alt="image" src="https://github.com/user-attachments/assets/31b131e1-6530-4486-9bb4-a006fe464d32" />

<img width="1024" height="464" alt="image" src="https://github.com/user-attachments/assets/41904e7a-31cb-4025-badc-4fdc979b1763" />

<img width="1024" height="377" alt="image" src="https://github.com/user-attachments/assets/237962f6-4aa8-4855-b7d0-306df5d0ee73" />


## Pre-Train and Fine-Tune
1、 Download the pre-trained weights from [Huggingface](https://huggingface.co/cyinen/S4D), and move it to the [finetune/checkpoints/pretrain/voxceleb2+AffectNet] directory.

2、 Run the following command to pre-train or fine-tune the model on the target dataset.

```bash
# create the envs
conda create -n s4d python=3.9
conda activate s4d
pip install -r requirements.txt

# pre-train
cd pretrain/omnivision &&  OMP_NUM_THREADS=1 HYDRA_FULL_ERROR=1 python train_app_submitit.py +experiments=videomae/videomae_base_vox2_affectnet

# fine-tune
cd finetune && bash run.sh
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MSA-LMC/S4D&type=Date)](https://star-history.com/#MSA-LMC/S4D&Date)
