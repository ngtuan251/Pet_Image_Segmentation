# Pet_Image_Segmentation
---

Suppose you want to know where an object is in the image, its shape, which pixel belongs to which object, and so on. In this situation, you'll want to segment the image, which means assigning a label to each pixel. The goal of image segmentation is to train a neural network to generate a pixel-by-pixel mask of an image. This helps in the interpretation of the image at a much lower level, namely the pixel level. Medical imaging, self-driving cars, and satellite imaging are just a few of the uses for image segmentation.

Spending hours learning so many Deep Learning courses and reading research papers, I realized that there is a very huge gap between creating a novel neural network architecture and using the model in an actual product. 

Hence, I devoted this project for structuring and developing production-ready Deep Learning code using OOP concepts to better my skills in building Machine Learning system in industy.

## Data
The dataset I will use is the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), created by Parkhi *et al*. The dataset consists of images, their corresponding labels, and pixel-wise masks. The masks are basically labels for each pixel. Each pixel is given one of three categories :

*   Class 1 : Pixel belonging to the pet.
*   Class 2 : Pixel bordering the pet.
*   Class 3 : None of the above/ Surrounding pixel.

## Result

![Segmentation Prediction]((https://raw.githubusercontent.com/ngtuan251/Pet_Image_Segmentation/master/images/segmentation_img.PNG)

## References

- [Image Segmentation - Tensorflow](https://www.tensorflow.org/tutorials/images/segmentation)


