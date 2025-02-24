# NVC: Robust Unsupervised Domain Adaptation through Negative-View Regularization
Inspired by [negative augmentation](https://arxiv.org/abs/2110.07858), We propose Negative View-based Contrastive (NVC) loss which can be used in UDA setting and boosts UDA performance, leveraging the negative augmentation, especially patch-based shuffling (a.k.a P-Shuffle).
Our comprehensive experiments show that NVC prevents ViT-based models from heavily depending on local features within image patches and helps the models to attend to the semantically meaningful global features.

Our paper "Robust Unsupervised Domain Adaptation through Negative-View Regularization" is accepted to WACV 2024, and you can access to the paper with the links: [abstract](https://openaccess.thecvf.com/content/WACV2024/html/Jang_Robust_Unsupervised_Domain_Adaptation_Through_Negative-View_Regularization_WACV_2024_paper.html) and [paper](https://openaccess.thecvf.com/content/WACV2024/papers/Jang_Robust_Unsupervised_Domain_Adaptation_Through_Negative-View_Regularization_WACV_2024_paper.pdf).

The bibtex code of our paper is as follows:
```bash
@InProceedings{Jang_2024_WACV,
    author    = {Jang, Joonhyeok and Lee, Sunhyeok and Kim, Seonghak and Kim, Jung-un and Kim, Seonghyun and Kim, Daeshik},
    title     = {Robust Unsupervised Domain Adaptation Through Negative-View Regularization},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {2462-2471}
}
```

Note: The first author of this paper, Joonhyeok Jang, changed the name to Sehui Jang. (February 21th, 2025)

# Requirements
We used:
- python==3.8.5
- torch==1.11.0
- torchvision==0.12.0

You may have to install other additional libraries such as timm and prettytable, and you can install the additional libraries without specifying the version. That is, you can use any version as long as there is no problem with compatibility with python, torch, and torchvision.


# Datasets
## Existing UDA datasets
We conducted our experiments on Office-31, Office-Home, and VisDA-2017.

## New UDA datasets
As we mentioned in the preceding section, we introduce and release a new dataset [Retail-71](https://github.com/JoonHyeokJ/Retail-71).

There are a few ways in which you download Retail-71.
1. Please go into the file 'examples/script_retail71_sdat_nvc.sh'([link](https://github.com/JoonHyeokJ/NVC/blob/master/examples/script_retail71_sdat_nvc.sh)), and you can find the download link.
2. You can check the [repository of Retail-71](https://github.com/JoonHyeokJ/Retail-71), and then find the download link.
3. You can directly access the [Google drive-based download link](https://drive.google.com/file/d/1ySCLGlJ9KEo2dOTIpFs_kpFfI_pf1E8v/view?usp=sharing).


# Training
As mentioned in our paper, we appoint [SDAT](https://github.com/val-iisc/SDAT) to the baseline method, and we attach our method NVC loss to the baseline.

For training, go into the folder 'examples'([link](https://github.com/JoonHyeokJ/NVC/tree/master/examples)) and then refer to the shell script files(.sh).
- Office-31: script_office31_sdat_nvc.sh
- Office-Home: script_officehome_sdat_nvc.sh
- VisDA-2017: script_visda2017_sdat_nvc.sh
- Retail-71: script_retail71_sdat_nvc.sh

An example is as follows:
```bash
cd examples
CUDA_VISIBLE_DEVICES=0 python cdan_mcc_sdat_nvc.py /your/path/to/Retail-71/retail71/0    \\
    -d Retail -s train -t validation --epochs 25  -b 96 -i 555 -p 185    \\
    --per-class-eval --no-pool    \\
    --lr 0.015 --seed 0 -a vit_tiny_patch16_224  --rho 0.02    \\
    --denorm_and_toPIL --gaussian_blur    \\
    --triplet_type latentv2 --triplet_coef 0.0 --triplet_temp 0.5    \\
    --neg_aug_type shuffle --neg_aug_ratio 1.00 --neg_aug_patch_size 32    \\
    --log /your/target_directory/to/save/log_and_checkpoints/sdat_rs0 ;
```

Also, you can see the full list of arguments in [examples/cdan_mcc_sdat_nvc.py](https://github.com/JoonHyeokJ/NVC/blob/master/examples/cdan_mcc_sdat_nvc.py).

# Test
## Existing UDA datasets
In the case of Office-31, Office-Home, and VisDA-2017, you can check the test(validation) accuracy (i.e., UDA performance) when the training is ended. The accuracy is displayed on the console and written into the log file(.txt).

## Retail-71 dataset
The validation accuracy is displayed on the console and written into the log file(.txt), as other benchmarkes are.

Importantly, Retail-71 contains its test set consisting of three difficulties (i.e., easy, medium, and hard difficulty).

You can see [examples/script_retail71_sdat_nvc_test.sh](https://github.com/JoonHyeokJ/NVC/blob/master/examples/script_retail71_sdat_nvc_test.sh) and use the test scripts to test the trained model and compute the easy, medium, and hard accuracies.

## Test with negative views (for all datasets)
The term 'negative view' means the sample (i.e., image) transformed by negative augmentation such as P-Shuffle.
In our paper, we proposed to evaluate based on Negative Accuracy (ideally 100/{# of classes} (%)) and Average Negative Confidence Score (ideally 1/{# of classes}), and both are the metrics computed with negative view to evaluate how heavily the trained model depends on the local features.
For both metrics, higher value means that the model more heavily depends on the local features. Thus, lower value implies the model captures the contextual relation between image patches better, which is more desired case.

You can find the scripts in [examples/eval_w_negaug.sh](https://github.com/JoonHyeokJ/NVC/blob/master/examples/eval_w_negaug.sh).


# Pretrained models
Please go into the folder 'examples'([link](https://github.com/JoonHyeokJ/NVC/tree/master/examples)) and then refer to the shell script files(.sh).
- Office-31: script_office31_sdat_nvc.sh
- Office-Home: script_officehome_sdat_nvc.sh
- VisDA-2017: script_visda2017_sdat_nvc.sh
- Retail-71: script_retail71_sdat_nvc.sh or script_retail71_sdat_nvc_test.sh
