function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  exp_m = exp(theta' * X); % 9 x m dimensions
  exp_m = [exp_m; ones(1, m)]; % expand the 10th class probabilities
  sum_exp = sum(exp_m, 1); % 1 x m
  P = bsxfun(@rdivide, exp_m, sum_exp); % 10 x m probability matrix
  I = sub2ind(size(P), y, 1:size(P,2));
  f = -sum(log(P(I)));
  
  class = (1:num_classes)';
  diff = bsxfun(@eq, y, repmat(class, 1, m)) - P; % I(y_i = k) - P(y_i = k | x_i; theta) : matrix
  g = - X * diff';
  g = g(:,1:num_classes-1);
  g=g(:); % make gradient a vector for minFunc

