# InfinityCoins
 A scratch GAN (Generative Adversarial Network) to generate images of mixed world coins using PyTorch


# User Notice:

 **This is my very first machine learning project. Since I am still learning machine learning and am at the start of it, this is a very early start and is a work in progress. Currently, I am suspecting that the GAN is mode collapsing after too many iterations. I am trying to fix this, so if you have any suggestions, create an issue, I would love to hear what your solution may be!**

 To attempt to solve mode collapse, I have currently tried these strategies:
 - Implementing a random seed through either numpy or torch.
 - Adding/decreasing layers within the neural net
 - Adding more randomness by flipping images during training through transformer
 
 What I have not tried to do yet to solve this issue and perhaps make the GAN more efficient:
 - Implement a different loss function (such as the Wasserstein loss function)
 - Implement mini batch discrimination
 - Experiment with different learning rates
 - Implement weight decay in optimizers.

# GAN Progress and Learning Demo:

**Epoch 1:**
![epoch_1](https://github.com/PatzEdi/InfinityCoins/blob/main/assets/gan_images_epoch_1.png?raw=true)

**Epoch 16**
![epoch_16](https://github.com/PatzEdi/InfinityCoins/blob/main/assets/gan_images_epoch_16.png?raw=true)

**Epoch 100-200 (sometimes the process takes quicker/longer to get to this stage)**
![epoch_Final](https://github.com/PatzEdi/InfinityCoins/blob/main/assets/generated_image.png?raw=true)

# Usage

 # Dataset Preprocessing
 - The data set used in this project was gathered using Kaggle, and can be found [here](https://www.kaggle.com/datasets/wanderdust/coin-images). Credits to [Pablo Lopez Santori](https://www.kaggle.com/wanderdust) for creating this dataset.
 - However, the dataset linked above was most likely used to train on classification tasks. What we needed, was a generative model to generate images of all these coins by mixing them together. So, I created [this script](/data_mixer.py) that automatically converges each file in each class folder into one singular folder. Credits to the [FinderZ](https://github.com/PatzEdi/FinderZ) library for making this really simple.
 - Let's look at this [script](/data_mixer.py). Make sure to install [FinderZ](https://github.com/PatzEdi/FinderZ) first, it is a file management library. You can see the **org_path** variable and the **mv_path** variable. The **org_path** variable takes in the path to the dataset linked above. So, it would be something like this: "[/path/before/this]/data/coins/coins/data/train". 
 - The **mv_path** variable takes in where to move each image within each class specified in the **org_path**. So, make it something like this: "[/path/before/this/]/data/coins/coins/data/train_mixed/mixed". Of course, make sure to create the train_mixed folder before hand, and the mixed folder, which is the class folder. So now, the data will be mixed into one singular folder.
 - Once you have the data mixed into one singular folder using [this script](/data_mixer.py), you can start to edit the training file variables to certain paths.

 # Training
 - In the [train script](/src/train/train.py), there are two main variables that you need to specify: **data_path** and **save_image_path**. Data path is the path that leads to train_mixed, **NOT** train_mixed/mixed. Keep in mind that the mixed folder is the class of the images, and the Dataloader from torch.utils.data must have a class to work with. The **save_image_path** is the path that you choose to save images to per epoch. For each epoch, an image is generated.
 - Once you have trained your model, two models will be saved for the discriminator and the generator. The saved models will have a .pth file extension. Once you have these, you can move on to the inference.
 
 # Inference
 - Open [inference.py](/src/inference/inference.py). Specify the same data_path (This does not really matter, as we no longer need to train). Specify the loading paths of the models in the **generator_model_path** and **discriminator_model_path** variables. 
 - Once you have specified your paths to each saved model, open [this script](/src/inference/simplified_inf.py). The first parameter is the path to where you want to save the image (which includes the file name and extension as well), while the second parameter is how many sub images you want for that saved image (default = 5). Please note that the main [inference.py](/src/inference/inference.py) must be in the same path as the [simplified_inf.py](/src/inference/simplified_inf.py), as the simplified_inf.py imports the inference.py.

# Enjoy
- Although the current model and training infrastructure may not be perfect and are suspected of leading to mode collapse, this is a work in progress. Certain patterns may be repetitive some times, but colors always vary. 

# How it works:
- The generative adversarial network takes in two layers: one that captures the whole image, and a segmentation mask that captures the center of the image. The segmentation mask is an attempt to increase the detail of the images generated within the coin.
- Using connected layers, a generator and a discriminator communicate like so: As the discriminator gets better and better at determining between fake and real images, the generator tries to fool the discriminator and creates a more realistic output through training.

# Methods used to train:
- For light training, I use an M1 Macbook Air for tests.
- For quicker and heavier training, I use https://runpod.io. I either use two NVIDIA RTX 3090 GPU's for training or one RTX 4090. Big credits to the cloud GPU team at [runpod](https://runpod.io)

# Precautions:
- When putting in bigger values for generated images during training, memory may be taken up to a large extent. Lack of memory will lead to washed out generations.
- When putting in bigger values for generated images, adjust the segmentation layer mask accordingly.

# Credits:
- [PyTorch](https://pytorch.org/)
- [FinderZ](https://github.com/PatzEdi/FinderZ)
- [World Coin Dataset](https://www.kaggle.com/datasets/wanderdust/coin-images)
- [Python Imaging Library](https://pypi.org/project/Pillow/)
- [numpy](https://numpy.org)
