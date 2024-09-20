# Machine Learning Project Collection
A comprehensive repository showcasing a variety of machine learning projects implemented in Python. Each project explores different algorithms, models, and techniques, highlighting practical implementations of machine learning concepts. This collection serves as a resource for experimentation and learning.

Each project is self-contained, complete with code, explanations, and results. Feel free to explore, fork, or contribute to any project that interests you!

## Table of Contents
1. [Projects](#projects)
    - [CAPTCHA Solver](#captcha-solver)
    - [Image Classifier](#image-classifier)
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

### Image Classifier
This project explores binary image classification using a convolutional neural network (CNN). The task is to distinguish between images of the actress Margot Robbie and actress Jaime Pressly, two individuals with similar appearances. Despite the challenge, the model is able to achieve moderate classification accuracy.

The model was trained on roughly 1,000 images (500 for each class) using supervised learning techniques. Due to the visual similarities between the two classes, this project serves as a great demonstration of the limitations and challenges faced by models in distinguishing between highly similar objects. Techniques such as data augmentation and dropout are used to prevent overfitting and improve the model's performance.

More details, including the dataset, model implementation, and results, can be found in the ```MargotRobbie_or_JaimePressly``` folder.

#### Key features:
- Model: Convolutional neural network (CNN).
- Task: Binary image classification (Margot Robbie vs Jaime Pressly).
- Dataset: 1,000 images (500 per class).
- Technologies: Tensorflow/Keras.

### Coming soon

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
