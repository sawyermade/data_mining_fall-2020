# Project 1: CNN MNIST
This assignment uses some of the MNIST dataset with 30,000 training (1/2 not
available) examples with 10,000 test images.  MNIST is 28x28 grayscale images of handwritten numbers from 0-9.
You need to run the code in the class file called  project1f20v1.pyPreview the document (as is) 3 times and report the average accuracy on
the test data. Then you can add another dense layer (or more) if you like and document that performance. That is not necessary
though.
Note: If you use an outside source (from the Internet, a book, etc.) you must reference it.  You and the source will then share credit with some points assigned to the source.

You MUST create a convolutional model with at least 2 convolutional layers and
at least 1 max or average pooling layer. You should have, for this assignment,
a dense layer before your outputs. Your goal is to achieve better accuracy
than this simple classifier. You MUST leave the train set (30K examples) and
test set (10K examples) unchanged. You can do whatever you like with validation
data. I have included 3K examples not in test or train for validation.
I used tensorflow more than keras in dealing with the data below. If
interested in what I am doing below you can start with the links below.

https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/overview.ipynb
https://www.tensorflow.org/datasets/catalog/overview

CNN Example using Keras:
https://www.tensorflow.org/tutorials/images/cnn

You MUST provide the python (py) file that starts with your netid. You MUST
put your name in the file. You MUST say what you did, what result you got
and why you think it was better. You MUST have in comments the results
you got as well as the previous requested information.
