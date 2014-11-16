function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

if size(labels, 1) ~= 1
    labels = labels';
end

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

stackNum = numel(stack);
activation = cell(size(stack));

%% forward propagate
input = data;
% stacked layers output
for d=1:stackNum
    activation{d}.z = bsxfun(@plus, stack{d}.w * input, stack{d}.b);
    activation{d}.a = sigmoid(activation{d}.z);
    input = activation{d}.a;
end
% softmax layer output
z_softmax = softmaxTheta * activation{stackNum}.a;
exp_term = exp(z_softmax);
sum_exp = sum(exp_term, 1);
probs = bsxfun(@rdivide, exp_term, sum_exp);

I = sub2ind(size(probs), labels, 1:size(probs,2));
cost = -sum(log(probs(I)));

% add the weight decay term for the softmax layer weights
cost = cost + lambda/2 * sum(sum(softmaxTheta.^2));

%% backpropagation

% First: compute gradient for the softmax layer
class = (1:numClasses)';
% this label_mask_mat will have one 1 in each column, and the 1 appears in the label-th row
label_mask_mat = bsxfun(@eq, labels, repmat(class, 1, M));
% check whether these two matrices have equal size
assert(isequal(size(probs), size(label_mask_mat)));
% delta_softmax is the error of the softmax layer
delta_softmax = - label_mask_mat .* (1 - probs);
% this is the reverse of label_mast_mat
label_mask_mat_reverse = ones(size(label_mask_mat)) - label_mask_mat;
delta_softmax = delta_softmax - label_mask_mat_reverse .* (-probs);

softmaxThetaGrad = delta_softmax * activation{stackNum}.a';
% add the weight decay term gradients
softmaxThetaGrad = softmaxThetaGrad + lambda * softmaxTheta;

% Second: stacked layers (not including the first layer)
delta_l = softmaxTheta' * delta_softmax .* (activation{stackNum}.a .* (1 - activation{stackNum}.a));
for d=stackNum:2
    stackgrad{d}.w = delta_l * activation{d-1}.a';
    stackgrad{d}.b = sum(delta_l,2);
    delta_l = stack{d}.w' * delta_l .* (activation{d-1}.a .* (1- activation{d-1}.a));
end

% Third: first layer (where input is the raw data input)
stackgrad{1}.w = delta_l * data';
stackgrad{1}.b = sum(delta_l,2);

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
