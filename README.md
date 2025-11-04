# [Static for Dynamic: Towards a Deeper Understanding of Dynamic Facial Expressions Using Static Expression Data](https://arxiv.org/pdf/2409.06154)
<img width="1024" height="506" alt="image" src="https://github.com/user-attachments/assets/db750330-84e2-4128-96c3-77c4a8fdc76c" />

## 📰 News

**[2025.9.17]** Our previous work [S2D](https://github.com/MSA-LMC/S2D/tree/main) has been recognized as a Highly Cited Paper by Clarivate.

**[2025.9.17]** The code and pre-trained models are available.

**[2025.9.15]** The paper is accepted by the IEEE Transactions on Affective Computing.

~~[2024.9.5] Code and pre-trained models will be released here.~~

## 🚀 Main Results

<img width="1024" alt="image" src="https://github.com/user-attachments/assets/31b131e1-6530-4486-9bb4-a006fe464d32" />

<img width="1024" height="464" alt="image" src="https://github.com/user-attachments/assets/41904e7a-31cb-4025-badc-4fdc979b1763" />

<img width="1024" height="377" alt="image" src="https://github.com/user-attachments/assets/237962f6-4aa8-4855-b7d0-306df5d0ee73" />


## Pre-Training and Fine-Tune
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

## ✏️ Citation

If you find this work helpful, please consider citing:
```bibtex
@ARTICLE{10663980,
  author={Chen, Yin and Li, Jia and Shan, Shiguang and Wang, Meng and Hong, Richang},
  journal={IEEE Transactions on Affective Computing}, 
  title={From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial Expression Recognition in Videos}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Adaptation models;Videos;Computational modeling;Feature extraction;Transformers;Task analysis;Face recognition;Dynamic facial expression recognition;emotion ambiguity;model adaptation;transfer learning},
  doi={10.1109/TAFFC.2024.3453443}}

@ARTICLE{11207542,
  author={Chen, Yin and Li, Jia and Zhang, Yu and Hu, Zhenzhen and Shan, Shiguang and Wang, Meng and Hong, Richang},
  journal={IEEE Transactions on Affective Computing}, 
  title={Static for Dynamic: Towards a Deeper Understanding of Dynamic Facial Expressions Using Static Expression Data}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Videos;Adaptation models;Face recognition;Transformers;Semantics;Multitasking;Computer vision;Spatiotemporal phenomena;Correlation;Emotion recognition;Dynamic facial expression recognition;mixture of experts;self-supervised learning;vision transformer},
  doi={10.1109/TAFFC.2025.3623135}}

}


```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MSA-LMC/S4D&type=Date)](https://star-history.com/#MSA-LMC/S4D&Date)
