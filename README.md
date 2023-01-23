# Light Echo Detection Module

This module uses YOLOv3, a popular real-time object detection system, to detect light echoes in survey images from modern telescopes. The module is implemented in python and can be easily integrated into detection pipelines. Read article for more details.


Dependencies

- python 3.6
- tensorflow 2.x
- numpy 


## Background

Light echoes (LEs) are the re!ections of astrophysical transients off of interstellar dust. They are fascinating astronomical phenomena that enable studies of the scattering dust as well as of the original transients. LEs, however, are rare and extremely dif"cult to detect as they appear as faint, diffuse, time-evolving features. The detection of LEs still largely relies on human inspection of images, a method unfeasible in the era of large synoptic surveys.

![alt text](figures/LE_split.png "Light Echoes from ATLAS")



## Limitations

The model has been trained on a limited dataset  and may not perform well on images with different lighting conditions or camera settings.
The module is currently limited to detecting light echoes in 2D images.


## Future Work

Improve the performance of the model by training on a larger and more diverse dataset.

Extend the module to handle multi-channel data.

## References
YOLOv3


Contributions

If you would like to contribute to the module, please open a pull request with your changes.










