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

训练测试代码主入口：

`main_basic.py`: basic model

`main.py`: 使用知识的model

所有代码运行所需参数已经在`config`文件中设置好

运行命令只需指定配置文件和所需使用的gpu id和模型迭代版本

示例：
`python main.py --cfg config/iu_retrieval.yml --gpu 0 --version 1`

## Citation
Shuxin Yang, Xian Wu, Shen Ge, S. Kevin Zhou, Li Xiao,Knowledge Matters: Radiology Report Generation with General and Specific Knowledge.2022
