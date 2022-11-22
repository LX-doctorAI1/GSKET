# Knowledge Matters: Radiology Report Generation with General and Specific Knowledge

结合通用知识和特定知识的医学报告生成

## Requirements
- `Python >= 3.6`
- `Pytorch >= 1.7`
- `torchvison`
- [Microsoft COCO Caption Evaluation Tools](https://github.com/tylin/coco-caption)
- [CheXpert](https://github.com/stanfordmlgroup/chexpert-labeler)

`conda activate tencent`

## Data

Download IU and MIMIC-CXR datasets, and place them in `data` folder.

- IU dataset from [here](https://iuhealth.org/find-medical-services/x-rays)
- MIMIC-CXR dataset from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- we also release the /data_auxiliary/iu file of the iu dataset which may needed in the preprocessing stage


## Folder Structure
- config : setup training arguments and data path
- data : store IU and MIMIC dataset
- misc : build general knowledge and specific knowledge 
- models: basic model and all our models
- modules: 
    - the layer define of our model 
    - dataloader
    - loss function
    - metrics
    - tokenizer
    - some utils
- preprocess: data preprocess
- pycocoevalcap: Microsoft COCO Caption Evaluation Tools

## Training & Testing

The source code for training can be found here：

`main_basic.py`: basic model

`main.py`: model with knowledge

The values of all the hyperparameters can be found in the folder 'config'.

To run the command, you only need to specify the config file and the GPU ID and iteration version of the model to be used


Example：
`python main.py --cfg config/iu_retrieval.yml --gpu 0 --version 1`

## Citation
Shuxin Yang, Xian Wu, Shen Ge, S. Kevin Zhou, Li Xiao, Knowledge Matters: Radiology Report Generation with General and Specific Knowledge. Medical Image Analysis，2022


## Contact
If you have any problem with the code, please contact  Shuxin Yang(aspenstarss@gmail.com) or  Li Xiao(andrew.lxiao@gmail.com).

## Thanks
Joseph Paul Cohen, Joseph D. Viviano, Paul Bertin, Paul Morrison, Parsa Torabian, Matteo Guarrera, Matthew P Lungren, Akshay Chaudhari, Rupert Brooks, Mohammad Hashir, Hadrien Bertrand. TorchXRayVision: A library of chest X-ray datasets and models. Medical Imaging with Deep Learning. https://github.com/mlmed/torchxrayvision, 2020

