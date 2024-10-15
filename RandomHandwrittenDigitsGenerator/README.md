# Random handwritten digit generator
A machine learning model designed to generate random handwritten digits. This project uses a generative adversarial network (GAN).

## Features
- The generator can generate solid handwritten digits most of the time.
- Generate whole numbers with n digits and optional thousands spacing.

## Model
### Generator
| Layer (Type)                    | Output Shape           | Parameters   |
|----------------------------------|------------------------|--------------|
| **Dense**                        | (None, 50,176)         | 10,035,200   |
| **BatchNormalization**           | (None, 50,176)         | 200,704      |
| **LeakyReLU**                    | (None, 50,176)         | 0            |
| **Reshape**                      | (None, 7, 7, 1,024)    | 0            |
| **Conv2DTranspose**              | (None, 7, 7, 512)      | 13,107,200   |
| **BatchNormalization**           | (None, 7, 7, 512)      | 2,048        |
| **LeakyReLU**                    | (None, 7, 7, 512)      | 0            |
| **Conv2DTranspose**              | (None, 14, 14, 256)    | 3,276,800    |
| **BatchNormalization**           | (None, 14, 14, 256)    | 1,024        |
| **LeakyReLU**                    | (None, 14, 14, 256)    | 0            |
| **Conv2DTranspose**              | (None, 14, 14, 128)    | 819,200      |
| **BatchNormalization**           | (None, 14, 14, 128)    | 512          |
| **LeakyReLU**                    | (None, 14, 14, 128)    | 0            |
| **Conv2DTranspose**              | (None, 28, 28, 1)      | 3,200        |

**Total Parameters**: 27,445,888 (104.70 MB)  
**Trainable Parameters**: 27,343,744 (104.31 MB)  
**Non-Trainable Parameters**: 102,144 (399.00 KB)

### Discriminator
| Layer (Type)                | Output Shape          | Parameters   |
|-----------------------------|-----------------------|--------------|
| **Conv2D**                   | (None, 14, 14, 128)   | 3,328        |
| **LeakyReLU**                | (None, 14, 14, 128)   | 0            |
| **Dropout**                  | (None, 14, 14, 128)   | 0            |
| **Conv2D**                   | (None, 7, 7, 256)     | 819,456      |
| **LeakyReLU**                | (None, 7, 7, 256)     | 0            |
| **Dropout**                  | (None, 7, 7, 256)     | 0            |
| **Conv2D**                   | (None, 4, 4, 512)     | 3,277,312    |
| **LeakyReLU**                | (None, 4, 4, 512)     | 0            |
| **Dropout**                  | (None, 4, 4, 512)     | 0            |
| **Flatten**                  | (None, 8,192)         | 0            |
| **Dense**                    | (None, 1)             | 8,193        |

**Total Parameters**: 4,108,289 (15.67 MB)  
**Trainable Parameters**: 4,108,289 (15.67 MB)  
**Non-Trainable Parameters**: 0

## Usage
1. #### Install dependencies
    Run ```pip install -r requirements.txt``` in your terminal to install the necessary python packages.

2. #### Train the model
    Run ```main.py``` to train the model on the MNIST dataset. You can adjust the number of epochs, noise dimension, etc. in ```data.py```. The model is automatically saved after training.

3. #### Generate random handwritten digits
    Use ```generate.py``` to generate a grid of digits or whole numbers as seen in the ```examples``` folder.

## Training progress
![Progress](./progress.gif)

## Training loss plot
![Loss Plot](losses_plot.png)

## Example evaluations
I generated 15 random numbers with increasing digits. They can be found in the ```examples``` folder, but here are some of them:

| ![Ex0](examples/generated_number_with_2_digits.png) | ![Ex1](examples/generated_number_with_4_digits.png) | ![Ex7](examples/generated_number_with_7_digits.png) |
| :--: | :--: | :--: |

We can observe that the generator is capable of generating convincing handwriting in some cases.

## License
This project is licensed under the Apache-2.0 License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgements
- <a href="https://www.tensorflow.org/" target="_blank">Tensorflow</a> and <a href="https://keras.io/" target="_blank">Keras</a> for deep learning frameworks
