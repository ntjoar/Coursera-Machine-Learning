function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

% Cost function 
z = X * theta;                    % sigmoid(z)
y_1 = y' * log(sigmoid(z));       % Find y=1 log
y_0 = (1-y)' * log(1-sigmoid(z)); % Find y=0 log
J = (-1) * (1 / m) * (y_1 + y_0); % Put it all together

% Gradient descent
error = sigmoid(z) - y;      % Find error displacement given
grad = (1 / m) * X' * error; % Get the gradient without updating theta values

% =============================================================

end
