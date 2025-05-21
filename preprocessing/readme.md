Preprocessing Workflow Overview
1.	Crop and Prepare FITS Images
The original FITS files are cropped to 576×576 resolution. The resulting image arrays are saved in .npz format, along with the RA/Dec coordinates. A sky map is generated for visualization.
Notebook: fits_crop_split_skymap.ipynb

2.	Convert Annotations (COCO → YOLO Format)
Annotations are converted from COCO format to YOLO format for compatibility with training pipelines.
Notebook: coco_to_yolo.ipynb

3.	Data Augmentation & Dataset Split
Augmentations such as flipping and rotation are applied to the images. Corresponding YOLO-format annotation files are created. The dataset is then split into training and testing subsets.
Notebook: LE576_aug_train_test_split.ipynb

⸻

Additional Tools

Bounding Box Visualization
Provides a visualization of YOLO-format bounding boxes over the images.
Notebook: viewbox.ipynb

Augmentation with Albumentations
Additional data augmentation is performed using the Albumentations package.
Notebook: albumentations.ipynb

Cross Validation
Split train/test for cross validation
Notebook: cross_validation.ipynb


