# Machine Learning Project Collection
A comprehensive repository showcasing a variety of machine learning projects implemented in Python. Each project explores different algorithms, models, and techniques, highlighting practical implementations of machine learning concepts. This collection serves as a resource for experimentation and learning.

Each project is self-contained, complete with code, explanations, and results. Feel free to explore, fork, or contribute to any project that interests you!

## Table of Contents
1. [Projects](#projects)
    - [CAPTCHA Solver](#captcha-solver)
    - [Image Classifier](#image-classifier)
    - [Random Handwritten Digits Generator](#random-handwritten-digits-generator)
    - [Portrait Colorization](#portrait-colorization)
    - [Coming soon](#coming-soon)
2. [Contributing](#contributing)
3. [License](#license)

## Projects
### CAPTCHA Solver
This project implements a convolutional neural network (CNN) enhanced with a global attention mechanism to tackle the task of CAPTCHA solving. CAPTCHAs, which are commonly used to distinguish between humans and bots, typically consist of distorted characters that require pattern recognition to interpret correctly.

The CNN model processes images of CAPTCHAs (each containing 5 alphanumeric characters) and utilizes the attention layer to focus on specific parts of the image. This attention mechanism allows the network to learn where to focus, making it more robust in interpreting challenging CAPTCHAs. The model achieves an impressive accuracy of over 97%.

More information, including the dataset, model architecture, training process, and results, can be found in the ```CaptchaSolver``` folder.

#### Key features:
- Model: Convolutional neural network (CNN) with a global attention layer.
- Task: CAPTCHA solving (length 5).
- Accuracy: 97%+.
- Dataset: Collection of distorted CAPTCHA images that can be gererated with Python.
- Technologies: Tensorflow/Keras.
---

### Image Classifier
This project explores binary image classification using a convolutional neural network (CNN). The task is to distinguish between images of the actress Margot Robbie and actress Jaime Pressly, two individuals with similar appearances. Despite the challenge, the model is able to achieve moderate classification accuracy.

The model was trained on roughly 1,000 images (500 for each class) using supervised learning techniques. Due to the visual similarities between the two classes, this project serves as a great demonstration of the limitations and challenges faced by models in distinguishing between highly similar objects. Techniques such as data augmentation and dropout are used to prevent overfitting and improve the model's performance.

More details, including the dataset, model implementation, and results, can be found in the ```MargotRobbie_or_JaimePressly``` folder.

#### Key features:
- Model: Convolutional neural network (CNN).
- Task: Binary image classification (Margot Robbie vs Jaime Pressly).
- Dataset: 1,000 images (500 per class).
- Technologies: Tensorflow/Keras.
---

### Random Handwritten Digits Generator
This project focuses on generating random handwritten digits using a Generative Adversarial Network (GAN). The objective is to create realistic images of digits that mimic human handwriting. By training the GAN on a dataset of handwritten digits, the model learns to produce high-quality images that can be used for various applications, such as digit recognition and data augmentation. Despite the inherent challenges of generating diverse and coherent outputs, the model successfully achieves impressive results in creating realistic handwritten digits.

#### Key features:
- Model: Generative Adversarial Network (GAN).
- Task: Generate images that mimic handwritten digits.
- Dataset: MNIST digits.
- Technologies: Tensorflow/Keras.
---

### Portrait Colorization
This project uses a deep learning model to colorize grayscale portraits, leveraging a U-Net Generator and a PatchGAN Discriminator. The architecture is inspired by Pix2Pix, optimized to work in the La*b* color space, enabling the model to hallucinate only the color channels while preserving the structural integrity of the grayscale input.

The model underwent two training phases:
1. Pretraining on 100k images for 20 epochs using a combination of L1 and adversarial loss.
2. Fine-tuning on 500k images for 5 epochs with adversarial loss only.

The training datasets include FFHQ, CelebA-HQ, and Synthetic Faces High Quality (SFHQ), resized to 256x256 for uniformity.

#### Key Features:
- Model: U-Net Generator + PatchGAN Discriminator.
- Loss Function: L1 + adversarial loss (pretraining), adversarial loss (fine-tuning).
- Dataset: Over 525,000 images from FFHQ, CelebA-HQ, and SFHQ.
- Technologies: TensorFlow/Keras.
---

### Coming soon
---

## Contributing
Contributions to this repository are highly encouraged! Whether it's improving existing models, adding new projects, or fixing bugs, your input is valuable. To contribute:
1. Fork the repository.
2. Create a new branch with a descriptive name (e.g. ```feature/new_algorithm```).
3. Make your changes.
4. Submit a pull request with a detailed explanation of your changes.

If you're adding a new project, make sure to include a ```README.md``` file in the project folder with the follwing information:
- Project description.
- Dataset used.
- Model architecture.
- Results (accuracy, loss, etc.).

## License
This repository is licensed under the Apache-2.0 License - see the [LICENSE](../LICENSE) file for details.
