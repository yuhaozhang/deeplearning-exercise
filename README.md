Deep Learning Exercises
=============

My own playground for basic deep learning algorithms. This also includes **answers** for most exercises on the Stanford deep dearning online tutorials. These tutorials can be found at:
- [[1] Stanford UFLDL Tutorial (older version)](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)
- [[2] Stanford Deep Learning Tutorial (newer version)](http://ufldl.stanford.edu/tutorial/)

The base code of most algorithms comes from the above tutorials. The original version of the base code could be found [here](http://ufldl.stanford.edu/tutorial/StarterCode/). I made some minor adjustments to the base code (mainly in terms of directory structure and how the data is used), but these changes can be safely ignored. I wrote and ran these codes with `MATLAB 2014a`.

Content includes (its corresponding directory and online tutorial are highlighted in parenthesis):
- Basic Machine Learning Algorithms (`basic/`) `[2]`
	- Linear Regression
	- Logistic Regression
	- Softmax Regression
- Sparse AutoEncoder (`autoencoder/`) `[1]`
- PCA Whitening (`pca/`) `[2]`
- Reconstruction ICA / RICA (`rica/`) `[2]`
- Supervised Convolutional Neural Network (`cnn/`) `[2]`
- Self-taught Learning (`stl/`) `[2]`
- Deep Networks / Stacked AutoEncoder (`stackedae/`) `[1]`

Some other directories that are necessary to run the code:
- `common/`: Directory with data I/O codes and minFunc source code;
- `data/`: A data directory with the MNIST data in MATLAB readable format, which could be downloaded from [here](http://ufldl.stanford.edu/tutorial/StarterCode/).

I went through all these tutorials and validate these answers with the given reference results. I personally found these tutorials and exercises to be very useful to deepen one's understanding of deep learning algorithms, since it "pushes" one to think deeply in terms of many implementation and math details behind the concepts. I highly recommend to follow the newer version tutorial (`[2]`), while using the older version (`[1]`) as supplementary materials.