function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

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

[~,pred] = max(probs,[],1);
pred = pred';

% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
