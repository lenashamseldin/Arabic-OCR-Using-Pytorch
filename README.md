# Arabic OCR - PyTorch

## Description
An API that takes a string of base64 encoded image as input and returns the predication and the probability using PyTorch and FastAPI. 


## Installation
```
git clone https://github.com/lenashamseldin/Arabic-OCR-Using-Pytorch.git
cd Arabic-OCR-Using-Pytorch
docker build -t arocr .
docker run -p 80:80 arocr
```
Open the [localhost url](http://0.0.0.0/docs#/default/pre_image_predict_post) from any browser

## Dataset
<details>
<summary><b>Arabic Letters Numbers OCR Dataset</b></summary>
[Download](https://www.kaggle.com/datasets/mahmoudreda55/arabic-letters-numbers-ocr/download?datasetVersionNumber=4)

Dataset consists of 29 letters.

```shell
-- Dataset
| --  أ    
|     | -- 0.png
|     | -- 1.png
|     | -- ...
| --  ب
|     | -- 0.png
|     | -- 1.png
|     | -- ...
| --  ت
|     | -- 0.png
|     | -- 1.png
|     | -- ...
...
```
</details>

## Training
* Install requirements
```
pip install -U --user -r requirements.txt
```
* Edit model in [notebook](https://github.com/lenashamseldin/Arabic-OCR-Using-Pytorch/blob/main/arLetters.ipynb)

## License
This project is available under the terms of [MIT License](https://choosealicense.com/licenses/mit/)
