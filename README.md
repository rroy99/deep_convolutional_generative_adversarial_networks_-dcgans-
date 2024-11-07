# Creating Anime Characters using Deep Convolutional Generative Adversarial Networks (DCGANs) and Keras 🎨🤖

<div align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.9+-blue?logo=tensorflow&logoColor=white" alt="TensorFlow Badge" />
  <img src="https://img.shields.io/badge/Keras-2.9+-purple?logo=keras&logoColor=white" alt="Keras Badge" />
  <img src="https://img.shields.io/badge/Python-3.8+-green?logo=python&logoColor=white" alt="Python Badge" />
  <img src="https://img.shields.io/badge/NumPy-1.21+-red?logo=numpy&logoColor=white" alt="NumPy Badge" />
  <img src="https://img.shields.io/badge/Matplotlib-3.5+-blue?logo=matplotlib&logoColor=white" alt="Matplotlib Badge" />
  <img src="https://img.shields.io/badge/Pandas-1.3.4+-orange?logo=pandas&logoColor=white" alt="Pandas Badge" />
  <img src="https://img.shields.io/badge/Seaborn-0.9.0+-purple?logo=seaborn&logoColor=white" alt="Seaborn Badge" />
  <img src="https://img.shields.io/badge/scikit--learn-0.20.1+-yellow?logo=scikit-learn&logoColor=white" alt="Scikit-learn Badge" />
  <img src="https://img.shields.io/badge/SkillsNetwork-Preinstalled-orange?logo=skillsnetwork&logoColor=white" alt="Skills Network Badge" />
</div>

## Documentation 📄

This project utilizes **Deep Convolutional Generative Adversarial Networks (DCGANs)** and **Keras** to generate unique anime characters. The **DCGAN** is a powerful deep learning architecture composed of two main components: the **Generator** and the **Discriminator**. The Generator creates fake data (anime character images in this case) while the Discriminator tries to differentiate between real and fake data. The goal is to train these two networks to work together, leading to the Generator producing more realistic anime characters over time.

### Objectives 🎯
- Understand the original formulation of **GANs**, and their two separately trained networks: **Generator** and **Discriminator**
- Implement **GANs** on simulated and real datasets
- Apply **DCGANs** to a dataset
- Understand how to train **DCGANs**
- Generate an image using a **DCGAN**
- Understand how changing the input of the latent space of **DCGANs** changes the generated image

### Dataset 📊
This project uses the **Anime Face dataset** from Kaggle, consisting of 20,000 anime character images. These images are used to train the GAN model, enabling the Generator to create realistic anime characters.

### Key Steps 🛠️:
1. **Generator Model**: Transposed convolution layers to upsample random noise into an image.
2. **Discriminator Model**: Convolutional layers to classify images as real or fake.
3. **Training Process**: Alternating between training the Generator and the Discriminator.
4. **Optimization**: **Adam optimizer** with binary cross-entropy loss to improve model performance.

## Technologies Used 💻

- **Pandas** for managing the data.
- **NumPy** for mathematical operations.
- **Scikit-learn** for machine learning and machine-learning-pipeline related functions.
- **Seaborn** for visualizing the data.
- **Matplotlib** for additional plotting tools.
- **Keras** for loading datasets.
- **TensorFlow** for machine learning and neural network related functions.
- **Python** for programming and scripting.
- **Kaggle Dataset (Anime Faces)**: The dataset used to train the model, containing 20,000 high-quality anime face images.

## Analysis of Technology Used 🔍

### 1. **TensorFlow & Keras**:
- **Why used**: TensorFlow provides an efficient framework for building and training neural networks, with Keras simplifying model definition and training.
- **Advantages**: These libraries offer high-level abstractions, making it easy to experiment with architectures like GANs. Keras integrates seamlessly with TensorFlow, allowing us to focus more on model design and less on the underlying details.
- **Challenges**: Working with GANs can be computationally intensive. Training deep models like DCGANs requires significant hardware resources (e.g., GPUs) and time, especially when working with large datasets.

### 2. **Convolutional Neural Networks (CNNs)**:
- **Why used**: DCGANs are built on the foundation of CNNs, which are especially good at processing image data. Convolutional layers help the models understand spatial hierarchies in images, making them suitable for generating and classifying image data.
- **Advantages**: CNNs capture spatial patterns effectively, which is crucial for generating high-quality images like anime faces.
- **Challenges**: Deep CNNs require large amounts of data and computation. Proper architecture and hyperparameter tuning are essential for stable training.

### 3. **GANs and DCGANs**:
- **Why used**: GANs are a powerful class of generative models that can learn to generate realistic data by training two networks against each other. DCGANs apply convolutional layers to GANs, making them particularly effective at generating images.
- **Advantages**: DCGANs improve on the basic GAN architecture by stabilizing training and making the generator capable of creating high-quality images.
- **Challenges**: GANs, including DCGANs, are prone to instability during training. It can be challenging to balance the Generator and Discriminator during training, requiring careful tuning of learning rates, optimizers, and architecture choices.

### 4. **Latent Space Exploration**:
- **Why used**: Latent variables are used as input for the Generator. By sampling different points from the latent space, we can generate diverse anime characters.
- **Advantages**: Latent space exploration allows for creativity and flexibility in generated outputs, enabling the generation of a wide variety of character designs.
- **Challenges**: Interpreting and manipulating the latent space to generate meaningful variations in output can be difficult, and it requires experimentation to find the right representations.


## Installation Steps 🛠️

1. **Clone the repository**  
   First, clone the repository to your local machine.

2. **Install dependencies**  
   Install the required libraries using `pip`. install the necessary libraries like TensorFlow, Keras, NumPy, Matplotlib, Seaborn, Scikit-learn and Pandas.

3. **Train the model**  
   Run the provided training script to train the Deep Convolutional GAN on the dataset.

4. **Generate new anime faces**  
   Once the model is trained, you can use it to generate new anime faces based on random noise inputs.

## Conclusion 🎉
This project demonstrates how DCGANs can be used to generate unique anime characters automatically. The combination of deep learning techniques like GANs, CNNs, and TensorFlow provides a robust solution to creating millions of characters for an online video game. Exploring the latent space opens up the possibility for infinite unique designs.
