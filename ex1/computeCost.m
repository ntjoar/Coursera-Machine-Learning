function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = X * theta;          % Find our h vector, X is m x n, theta is n x 1
error = h - y;          % Find the error by doing h - y, y is m x 1 -> m x 1
error_sqr = error .^ 2; % Do an elementwise square as in the equation
q = sum(error_sqr);     % Find the sum of all elements
J = 1 / (2 * m) * q;    % Divide by 2 times the number of elements for fancy avg

% =========================================================================

end
