# PLAGen

## Abstract

Although current automatic report generation technologies have made significant progress, they are still defective in processing pixel-level information, which does not fully utilize the local pixel-level information and ignores the abnormal regions. To obtain visual information about abnormal areas and to exclude normal areas from interfering with report generation, we propose pixel-level information of localized and abnormal regions for the radiology report generation framework (PLAGen).  In the abnormal regions prominence module (ARPM), we use weak supervision to obtain the prominence map of the abnormal regions with a dark background, which reduces the background interference and then get the pixel-level information of the abnormal regions by the visual extractor. In the organ mask generation module (OMGM), we use semantic segmentation to segment the images into mask images of each organ to obtain localized pixel-level information and combine it with corresponding disease keywords to enhance the recognition of diseases. In this process, we introduce a cosine similarity function to constrain the consistency between the pixel-level information and the textual information. Additionally, position information is added as input to the multimodal alignment module (MAM), which enriches the cross-modal information and enhances the connections between local information. We perform a large number of experiments on the public IU X-ray dataset, and the results demonstrate that PLAGen performs better compared to existing state-of-the-art methods.

![](C:\Users\35106\Desktop\NLP\model.png)

## Updates

1. PLAGen now supports Multi-GPU (Distributed) and Mixed Precision Training, to support the new features, please ensure Pytorch Version >= 1.8 (Note there may be slight difference for the test results between Multi-GPU test and Single GPU test due to the DDP sampler.
2. We provide a separate test scripts to enable quick test in the trained dataset.
3. We recommend to re-generate the initial prototype matrix if you have your own data-precessing on the dataset, e.g, different image resolution or downsampled images.
4. We optimize and clean some parts of the code.

## Prerequisites

The following packages are required to run the scripts:
- [Python >= 3.6]
- [PyTorch >= 1.6]
- [Torchvision]
- [Pycocoevalcap]

* You can create the environment via conda:
```bash
conda env create --name [env_name] --file env.yml
```


## Download Trained Models
You can download the trained models [here](https://drive.google.com/drive/folders/1_y_6srL2ZnvDvE_I0YDvdgRzZCNrcMUf?usp=sharing).

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://openi.nlm.nih.gov/faq).

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/).

After downloading the datasets, put them in the directory `data`.

## Pseudo Label Generation
You can generate the pesudo label for each dataset by leveraging the automatic labeler  [ChexBert](https://github.com/stanfordmlgroup/CheXbert).

We also provide the generated labels in the files directory.

## Cross-modal Prototypes Initialization
The processed cross-modal prototypes are provided in the files directory.
For those who prefer to generate the prototype for initilization by their own, you should:
- Leverage the pretrained visual extractor (imagenet-pretrained) and Bert (ChexBert) to extract the visual and texual features.
- Concat the visual and texual features.
- Utilize K-mean algorithm to cluster to cross-modal features to 14 clusters.

The above procedure is elobarately described in our paper.

## Experiments on IU X-Ray
Our experiments on IU X-Ray were done on a machine with 1x2080Ti.

Run `bash run_iu_xray.sh` to train a model on the IU X-Ray data.

## Run on MIMIC-CXR
Our experiments on MIMIC-CXR were done on a machine with 4x2080Ti.

Run `bash run_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

