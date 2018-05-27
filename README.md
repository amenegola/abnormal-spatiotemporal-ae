# Abnormal Event Detection in Videos Using Spatiotemporal Autoencoder
This repository hosts the codes for "Abnormal Event Detection in Videos Using Spatiotemporal Autoencoder".
Paper can be found at [Springer](https://link.springer.com/chapter/10.1007/978-3-319-59081-3_23) and [arXiv](https://arxiv.org/abs/1701.01546).

Prerequisites:
- keras
- tensorflow
- h5py
- scikit-image
- scikit-learn
- sk-video
- tqdm (for progressbar)
- coloredlogs (optional, for colored terminal logs only)

You can use the `Dockerfile` provided to build the environment then enter the environment using `nvidia-docker run --rm -it -v HOST_FOLDER:/share DOCKER_IMAGE bash`.

First, you need to convert the videos to frame with `convert_video_to_frame.py`. First mkdir a `data` folder and put the folder floripa (request Drive tar.gz file from Author) in it. Run the script. For each dataset, put the training videos into `VIDEO_ROOT_PATH/DATASET_NAME/training_videos` and testing videos into `VIDEO_ROOT_PATH/DATASET_NAME/testing_videos`. Example structure of training videos for `avenue` dataset:
- `VIDEO_ROOT_PATH/avenue/training_videos`
  - `01.avi`
  - `02.avi`
  - ...
  - `16.avi`

Default configuration can be found at `config.yml`. Change it if necessary. 

To train the model, just run `python start_train.py`.

Once you have trained the model, you may now run `python start_test.py` after setting the parameters at the beginning of the file (Testing is not implemented yet).

Please cite the following paper if you use their code / paper:
```
@inbook{Chong2017,
  author    = {Chong, Yong Shean and
               Tay, Yong Haur},
  editor    = {Cong, Fengyu and
               Leung, Andrew and
               Wei, Qinglai},
  title     = {Abnormal Event Detection in Videos Using Spatiotemporal Autoencoder},
  bookTitle = {Advances in Neural Networks - ISNN 2017: 14th International Symposium, ISNN 2017, Sapporo, Hakodate, and Muroran, Hokkaido, Japan, June 21--26, 2017, Proceedings, Part II},
  year      = {2017},
  publisher = {Springer International Publishing},
  pages     = {189--196},
  isbn      = {978-3-319-59081-3},
  doi       = {10.1007/978-3-319-59081-3_23},
  url       = {https://doi.org/10.1007/978-3-319-59081-3_23}
}
```
