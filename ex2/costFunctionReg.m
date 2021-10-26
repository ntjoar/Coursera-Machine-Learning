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

[j g] = costFunction(theta, X, y); % All of our cost functions run off this one

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

end
