# VQA-uncertainty

Official code for "[Exploring and exploiting model uncertainty for robust visual question answering](https://link.springer.com/article/10.1007/s00530-024-01560-0)" **(Multimedia 2024 Accepted)** 


>Visual Question Answering (VQA) methods have been widely demonstrated to exhibit bias in answering questions due to the distribution differences of answer samples between training and testing, resulting in resultant performance degradation. While numerous efforts have demonstrated promising results in overcoming language bias, broader implications (e.g., the trustworthiness of current VQA model predictions) of the problem remain unexplored. In this paper, we aim to provide a different viewpoint on the problem from the perspective of model uncertainty. In a series of empirical studies on the VQA-CP v2 dataset, we find that current VQA models are often biased towards making obviously incorrect answers with high confidence, i.e., being overconfident, which indicates high uncertainty. In light of this observation, we: (1) design a novel metric for monitoring model overconfidence, and (2) propose a model calibration method to address the overconfidence issue, thereby making the model more reliable and better at generalization. The calibration method explicitly imposes constraints on model predictions to make the model less confident during training. It has the advantage of being model-agnostic and computationally efficient. Experiments demonstrate that VQA approaches exhibiting overconfidence are usually negatively impacted in terms of generalization, and fortunately their performance and trustworthiness can be boosted by the adoption of our calibration method. 

> ![image](https://github.com/user-attachments/assets/ef546396-63ff-4a53-a88e-07fe81c927ac)

One merit of our method is that it is agnostic to model architectures. Therfore， as described in the paper, our method can be adapted to previous models such as [CSS](https://github.com/yanxinzju/CSS-VQA) , [SSL](https://github.com/CrossmodalGroup/SSL-VQA), [UpDn](https://github.com/chrisc36/bottom-up-attention-vqa), etc.

## Requirements
Python 3.8.8
Pytorch 1.9.0+cu111
Cuda 11.0
Gpu Nvidia 2080ti（11G）

## Data Setup
You can use
```
bash /tools/download.sh
```
to download the data <br> and the rest of the data and trained model can be obtained from [BaiduYun](https://pan.baidu.com/s/1oHdwYDSJXC1mlmvu8cQhKw)(passwd:3jot) or [MEGADrive](https://mega.nz/folder/0JBzGBZD#YGgonKMnwqmeSZmoV7hjMg) unzip feature1.zip and feature2.zip and merge them into data/rcnn_feature/ <br> use
```
bash CSS+conf/tools/process.sh 
```
to process the data <br>

## Training
```
cd CSS-VQA-master

# css
CUDA_VISIBLE_DEVICES=5 python main.py --dataset cpv2 --mode q_v_debias --debias learned_mixin --topq 1 --topv -1 --qvp 5 --output [t] --seed 0

# lmh

CUDA_VISIBLE_DEVICES=5 python main.py --dataset cpv2 --mode updn --debias learned_mixin --topq 1 --topv -1 --qvp 5 --output [lmh] --seed 0

# updn
CUDA_VISIBLE_DEVICES=5 python main.py --dataset cpv2 --mode updn --debias none --topq 1 --topv -1 --qvp 5 --output [updn] --seed 0

# rubi
CUDA_VISIBLE_DEVICES=5 python rubi_main.py --dataset cpv2 --mode updn --output [rubi] --seed 0
```

## Testing

```
CUDA_VISIBLE_DEVICES=7 python eval.py --dataset cpv2 --debias learned_mixin --model_state []
```

## Key implementation codes
The model implementation is mainly based on the loss function of calibration confidence, and the key code is [here](https://github.com/HCI-LMC/VQA-Uncertainty/blob/main/base_model.py#L19) and [here](https://github.com/HCI-LMC/VQA-Uncertainty/blob/main/base_model.py#L86). 
```
# L_tacs gt
def compute_self_loss(logits_neg, labels):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)

    qice_loss = neg_top_k.mean()
    return qice_loss

# L_conf max
def compute_loss(logits_neg, labels):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    prediction_max, pred_ans_ind = torch.topk(F.softmax(logits_neg, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)
    pre_ans_k = prediction_max.squeeze(1)
    # neg_top_k = neg_top_k.squeeze(1)
    qice_loss = neg_top_k.mean()
    pre_ans_k = pre_ans_k.tolist()
    neg_top_k = neg_top_k.tolist()
    return qice_loss ,pre_ans_k ,neg_top_k

loss = loss + compute_self_loss(logits, labels)
```

## Compute EOF
Taking Lxmert as an example, the EOF calculation method of the model is as follows:
```
python acc_per_type.py
```



## Citation

```bibtex
@article{zhang2024exploring,
  title={Exploring and exploiting model uncertainty for robust visual question answering},
  author={Zhang, Xuesong and He, Jun and Zhao, Jia and Hu, Zhenzhen and Yang, Xun and Li, Jia and Hong, Richang},
  journal={Multimedia Systems},
  volume={30},
  number={6},
  pages={1--14},
  year={2024},
  publisher={Springer}
}
  ```

### Acknowledgments
[CSS](https://github.com/yanxinzju/CSS-VQA) , [SSL](https://github.com/CrossmodalGroup/SSL-VQA), [UpDn](https://github.com/chrisc36/bottom-up-attention-vqa), etc. many thanks

