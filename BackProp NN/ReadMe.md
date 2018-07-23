NN class created to use the Mean Squared Error and Cross Entropy Error functions. The mnist handwritten 
dataset: http://yann.lecun.com/exdb/mnist/ performs well, reaching around 90% accuracy on a 2 epoch, [28^2,30,10] configuration.


Unlike the mnist hand recognition dataset, this dataset performs horribly as compared. It signifies that there needs to
be an alternative approach to training said NN. (Convolutional NN)


Work in Progress:

Defined a Node class and neural network class. Node class initiated with a a random weight vector. The 
Network is defined by an array consisting of the number of nodes per layer. e.g. [10,20,10].

Uses the Mean squared error loss function to tend towards Minima. It's likely I will switch to the Cross entropy Error
function to prevent stalling.

DataSet: http://www.vision.caltech.edu/Image_Datasets/Caltech101/

The images are grayscaled and resized to a 32x32 square. Then a dictionary is created and saved into a pkl file.
The network.py file loads the dictionary and uses it to train the network.

Running the imageProcess.py file in the same directory as the folder containing the data set, will create the required pkl file.
