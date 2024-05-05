# MAP-IT-to-Visualize-Representations
Code associated with the paper "MAP IT to Visualize Representations". This is code for the basic algorithm, in Matlab. The hope is to extend over time, also to other languages.  

MAP IT is inspired by t-SNE (please see https://lvdmaaten.github.io/tsne/) and assumes as input a matrix of pairwise transition probabilities in the input space. The code is currently not very efficient in terms of scalability. The transition matrix used in the paper for a subset of MNIST is made available and can be directly used. The corresponding labels are also made available. To visualize this subset of MNIST in a 2-dimensional plane, for instance using 7 neighbors to model neighborhood structure, please type 

Ydata = mapit_p(Pmnist2000, labels_mnist2000, 2, 7)

MAP IT to Visualize Representations

Robert Jenssen

ICLR 2024

https://openreview.net/pdf?id=OKf6JtXtoy 
