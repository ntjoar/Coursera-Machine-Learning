function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Hypothesis
h = X * theta;

% Cost function
std_cost = sum((h - y) .^ 2);                 % Standard cost
reg_cost = lambda * (sum(theta(2:end) .^ 2)); % Regularization term
J = (std_cost + reg_cost) / (2 * m);          % Total cost

% Gradient descent
alpha = 1;                                 % Assumed since no parameter sets this value
grd = (alpha / m) * (X' *(h - y));         % Standard gradient
reg = (lambda / m) * [0; theta(2:end, :)]; % Regularization term
grad = grd + reg;                          % Total gradient

% =========================================================================

grad = grad(:);

end
