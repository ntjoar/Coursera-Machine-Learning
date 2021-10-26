function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    h = X * theta;                        % Find our h vector, X is m x n, theta is n x 1
    error = h - y;                        % Find the error by doing h - y, y is m x 1 -> m x 1
    delta = X' * error;                   % Find the delta by doing X transpose times error
    theta_change = alpha * 1 / m * delta; % Find our change by multiplying delta by 1/m and alpha
    theta = theta - theta_change;         % Subtract, remember that this is done simultaneously in our Gradient descent equation

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
