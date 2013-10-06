% The script loads the training data and show an image of a random index.
% This may be helpful for visualizing the data and show how to load the data in Matlab or Octave

train = load('mnist_train.txt');
idx = randi(size(train, 1));
img = uint8(reshape(train(idx , 2:end), 28, 28)');
imshow(img)
