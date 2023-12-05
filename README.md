# HiFiGAN

## Installation

```shell
pip install -r requirements.txt
```

Веса доступны по [ссылке](https://drive.google.com/file/d/10pMyN2wlvmbZLGA24-fbDmwozfunNCoW/view?usp=sharing)

```shell
gdown https://drive.google.com/uc?id=10pMyN2wlvmbZLGA24-fbDmwozfunNCoW
```

Загрузка данных

```shell
bash scripts/get_data.sh
```

## Training

``` shell
python train.py -c config.json
```

## Inference

```shell
python test.py --resume checkpoint.pth --config config.json --input /input/folder --output /output/folder
```
