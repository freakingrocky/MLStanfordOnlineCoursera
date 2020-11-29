function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Code From costFunction.m
h = sigmoid(X * theta);

unreg = (1/m) * sum(-y .* log(h) - (1 - y) .* log(1 - h));

ungrad = ((1/m) .* X' * (h - y));

% The norm operator simply returns the largest singular value
J = unreg + ((lambda/(2*m)).*norm(theta(2:end))^2);

% Real Code Starts From Here
grad = ungrad + ((lambda/m) .* theta);
grad(1) = ungrad(1);


% =============================================================

end
