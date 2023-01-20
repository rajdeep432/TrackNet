# TrackNet

Pytorch implementation based on [TrackNetv2](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2) (improvements in
progress).

## Installation

```
git clone https://github.com/mareksubocz/TrackNet
cd /TrackNet
pip install -r requirements.txt
```

## Training

```
python train.py
```

## Dataset Labelling

Keybindings:

- l : next frame
- h : previous frame
- o : annotate occluded ball
- m : annotate ball in motion (blurred)
- v : annotate well-visible ball
- n : fast-forward/pause video
- q : finish annotating

```
python labellingTool.py video.mp4
```

<p align="center">
  <img src="labelling_tool_demo.gif" alt="animated" />
</p>
<p align="center">
  <em>Labelling tool in use. Fast-forward function is distorted due to gif compression.</em>
</p>

## train.py Parameters cheatsheet

| Argument name      | Type  | Default value | Description |
|--------------------|-------|---------------|-------------|
|--weights           |str    |None           |Path to initial weights the model should be loaded with. If not specified, the model will be initialized with random weights.|
|--checkpoint        |str    |None           |Path to a checkpoint, chekpoint differs from weights due to including information about current loss, epoch and optimizer state.|
|--batch_size        |int    |2              |Batch size of the training dataset.|
|--val_batch_size    |int    |1              |Batch size of the validation dataset.|
|--shuffle           |bool   |True           |Should the dataset be shuffled before training?|
|--epochs            |int    |10             |Number of epochs.|
|--train_size        |float  |0.8            |Training dataset size.|
|--lr                |float  |0.01           |Learning rate.|
|--momentum          |float  |0.9            |Momentum.|
|--dataset           |str    |'dataset/'     |Path to dataset.|
|--device            |str    |'cpu'          |Device to use (cpu, cuda, mps).|
|--type              |str    |'auto'         |Type of dataset to create (auto, image, video). If auto, the dataset type will be inferred from the dataset directory, defaulting to image.|
|--save_period       |int    |10             |Save checkpoint every x epochs (disabled if <1).|
|--save_weights_only |bool   |False          |Save only weights, not the whole checkpoint|
|--save_path         |str    |'weights/'     |Path to save checkpoints at.|

## Docker

Steps for running an app:

```
docker build -t volleyball Dockerfile        
```

then use:

```
docker-compose -d up app 
```

check what's inside container with:
```
docker exec -it volleyball bash 
```


## Streamlit

For presentation purpose small demo page using <b>Streamlit</b>. Page is available under: http://demo

## Api

Proper endpoints for <b>Streamlit</b> demo App are created via FastAPI under: http://api