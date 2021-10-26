function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

X_sqr = X' * X;             % X^T times X
X_sqr_inv = pinv(X_sqr);    % (X^T times X)^(-1)
theta = X_sqr_inv * X' * y; % (X^T times X)^(-1) * X^T * y

% ============================================================

end
