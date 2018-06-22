function [J, grad] = costFunction(theta, X, y)
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
% number of training examples
m = length(y);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta

J = (-1/m)*sum(y.*log(sigmoid(X*theta)) + (1-y).*log(1-sigmoid(X*theta)));

% used to determine the gradient of cost function
temp = sigmoid(X*theta);
error = temp-y;
grad = (1/m)*(X'*error);

end
