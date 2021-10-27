function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Cost function  
z = X * theta;                      % sigmoid(z)
y_1 = y' * log(sigmoid(z));         % Find y=1 log
y_0 = (1 - y)' * log(1-sigmoid(z)); % Find y=0 log
j = (-1) * (1 / m) * (y_1 + y_0);   % Put it all together

% Gradient descent
error = sigmoid(z) - y;      % Find error displacement given
g = (1 / m) * X' * error; % Get the gradient without updating theta values

% Regularized cost function is our old cost function plus our new one 
summation = sum(theta .^ 2) - (theta(1) .^ 2); % Get the summation from 1 to n, without theta_0
reg_cost = lambda / (2 * m) * (summation);     % Get the change needed
J = j + reg_cost;                              % Add the changes needed

% Regularized gradient descent
t = theta;                % Stores what theta we want to add by
scale = lambda / m;       % Scale lambda / m
grad = g + scale * theta; % Grad is just the old g + the new factor lambda / m times theta_j
grad(1) = g(1);           % Theta_0 stays the same, so reset this value to the original

% =============================================================

grad = grad(:);

end
