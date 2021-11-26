# Skin Cancer Classifier - Modeled with a Convolutional Neural Network

This is a custom trained neural network model that classifies skin lesions as being benign or malignant. I documented my workflow in a jupyter notebook and put the entire training process inside a docker image. I also want to show how versatile a docker image can be, because this model is now self contained and can train without any dependencies on any machine that downloads it, including a cloud instance with powerful GPUs. The result
that the docker image produces is a CNN model that is saved on the local machine.  

My docker image can be accessed at: https://hub.docker.com/repository/docker/pavelbrn/tfdocker 
