

# AILE: AI powered Light Echo detection module

This module uses YOLOv3, a popular real-time object detection system, to detect light echoes in survey images from modern telescopes. The module is implemented in python and can be easily integrated into detection pipelines. See this [article](https://doi.org/10.3847/1538-3881/ac9409) for more details.

Please cite our work if you use the content of this repository:

Li, X., Bianco, F.B., Dobler, G., Partoush, R., Rest, A., Acero-Cuellar, T., Clarke, R., Fortino, W.F., Khakpash, S. and Lian, M., 2022. Toward the Automated Detection of Light Echoes in Synoptic Surveys: Considerations on the Application of Deep Convolutional Neural Networks. The Astronomical Journal, 164(6), p.250.

@article{Li_2022,
doi = {10.3847/1538-3881/ac9409},
url = {https://dx.doi.org/10.3847/1538-3881/ac9409},
year = {2022},
month = {nov},
publisher = {The American Astronomical Society},
volume = {164},
number = {6},
pages = {250},
author = {Xiaolong Li and Federica B. Bianco and Gregory Dobler and Roee Partoush and Armin Rest and Tatiana Acero-Cuellar and Riley Clarke and Willow Fox Fortino and Somayeh Khakpash and Ming Lian},
title = {Toward the Automated Detection of Light Echoes in Synoptic Surveys: Considerations on the Application of Deep Convolutional Neural Networks},
journal = {The Astronomical Journal},
}

Dependencies

- python 3.6
- numpy 
- matplotlib
- tensorflow 2.x
- astropy
- opencv
- labelme

## Background

Light echoes (LEs) are the re!ections of astrophysical transients off of interstellar dust. They are fascinating astronomical phenomena that enable studies of the scattering dust as well as of the original transients. LEs, however, are rare and extremely dif"cult to detect as they appear as faint, diffuse, time-evolving features. The detection of LEs still largely relies on human inspection of images, a method unfeasible in the era of large synoptic surveys. We prepared a dataset from ATLAS telescope and assessing the ability of YOLOv3 for detection of LEs.

![LE_split.png](figures/LE_split.png "Light Echoes from ATLAS")


## Model
YOLOv3 use Darknet-53 as the backbone network to do feature extractions and make predictions at three scales (1/8, 1/16 and 1/32 the size of input image).
![yolo.png](figures/yolo.png "YOLOv3 archetecture")


The YOLOv3 minimize a loss function that consists of three parts: confidence loss $L_{conf}$, classification loss $L_{cls}$ and bounding-box regression loss $L_{box}$.

$$L_{total} = L_{conf} + L_{cls} + L_{box}$$

## Results
We train the model on Google Colab platform. We evaulate the model by precision, recall and F1 score. 

![yolo.png](figures/yolo_losscurve.png "YOLOv3 archetecture")


## Limitations

The model has been trained on a limited dataset from ATLAS telescope.
The input image is currently cropped to $576\times 576$. 


## Future Work

Improve the performance of the model by training on a larger and more diverse dataset.

Extend the module to handle multi-channel data.

Extend the module to handle time-series image data.

## References
- [YOLOv3](https://doi.org/10.48550/arXiv.1804.02767)

- [Toward the Automated Detection of Light Echoes in Synoptic Surveys: Considerations on the Application of Deep Convolutional Neural Networks](https://doi.org/10.3847/1538-3881/ac9409)


Contributions

If you would like to contribute to the module, please open a pull request with your changes.

