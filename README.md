# DLIP

WIP

Data preparation for CV tasks (super-resolution, denoising, others):
- LR generation with openCV-based composable transformation (https://github.com/victorca25/opencv_transforms): (TBD)
- Extraction of realistic image kernels with KernelGAN (http://www.wisdom.weizmann.ac.il/~vision/kernelgan/): modified to work with Tensorflow 2.0 and PyTorch >= 1.3.0 without generating warnings, the resulting kernels are saved as numpy arrays (npy)
- Extraction of noise patches from real images (https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Image_Blind_Denoising_CVPR_2018_paper.pdf)

