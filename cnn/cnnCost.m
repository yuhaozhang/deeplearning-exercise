function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
activitions = cnnConvolve(filterDim, numFilters, images, Wc, bc);
activationsPooled = cnnPool(poolDim, activitions);

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
exp_term = exp(Wd * activationsPooled + repmat(bd, 1, 10)); % expand bd from 10 x 1 to 10 x 10
sum_exp = sum(exp_term, 1);
probs = bsxfun(@rdivide, exp_term, sum_exp);

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%
I = sub2ind(size(probs), labels', 1:size(probs,2));
cost = -sum(log(probs(I)));

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%

% Error formula: delta^(4)_i = - \sum_j{I(y == j) * (I(i==j) - a^(4)_i)}
class = (1:numClasses)';
% this label_mask_mat will have one 1 in each column, and the 1 appears in
% the label-th row
label_mask_mat = bsxfun(@eq, labels', repmat(class, 1, numImages));
% check whether these two matrices have equal size
assert(isequal(size(probs), size(label_mask_mat)));
% delta_4 is delta_d, which is the error of the softmax layer
delta_4 = - label_mask_mat .* (1 - probs);
% this is the reverse of label_mast_mat
label_mask_mat_reverse = ones(size(label_mask_mat)) - label_mask_mat;
delta_4 = delta_4 - label_mask_mat_reverse .* (-probs);

% delta_3 is the error of the hidden layer before the softmax, f(z) = z for
% this layer
delta_3 = Wd' * delta_4;
assert(isequal(size(delta_3), [outputDim * outputDim * numFilters, numImages]));
% backpropagate error through the pooling layer and compute gradients
for imageNum=1:numImages
    % reshape to a square shape: outputDim x outputDim x numFilters
    delta_source = reshape(delta_3(:, imageNum), [outputDim, outputDim, numFilters]);
    im = squeeze(images(:,:,imageNum));
    for filterNum=1:numFilters
        % delta_pool shape: convDim * convDim
        delta_pool = 1/(poolDim^2) * kron(delta_source(:,:,filterNum), ones(poolDim));
        delta_rot = rot90(delta_pool,2);
        Wc_grad(:,:,filterNum) = Wc_grad(:,:,filterNum) + conv2(im, delta_rot, 'valid');
        bc_grad(filterNum) = bc_grad(filterNum) + sum(sum(delta_pool));
    end
end

% Wc_grad = Wc_grad / numImages;
% bc_grad = bc_grad / numImages;

% vectorized for Wd gradients
Wd_grad = delta_4 * activationsPooled';
bd_grad = sum(delta_4, 2);

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
