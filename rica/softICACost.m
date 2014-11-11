%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
EPSILON = params.epsilon;
LAMBDA = params.lambda;
Wx = W * x;
Wx_smoothed = l1Smooth(Wx, EPSILON);
cost = LAMBDA * sum(sum(Wx_smoothed)) + 1/2 * sum(sum((W' * Wx - x).^2));

% gradient
% reconstruction term
WWx_minus_x = W' * Wx - x;
grad1 = 2 * W * WWx_minus_x * x' + 2 * Wx * WWx_minus_x';

% sparsity term, solve it using a neural network, and use sqrt(x^2 +
% epsilon) to replace |x|
grad2 = Wx ./ Wx_smoothed * x';
Wgrad = 1/2 * grad1 + LAMBDA * grad2;

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);

end

function y = l1Smooth(x, epsilon)
y = sqrt(x.^2 + epsilon);
end