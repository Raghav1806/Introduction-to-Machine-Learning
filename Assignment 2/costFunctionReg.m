function [J, grad] = costFunctionReg(theta, X, y, lambda)
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
z = X*theta;
h = sigmoid(z);
logisf = (-y)' * log(h) - (1-y)' * log(1-h);

J = ((1/m).*sum(logisf)) + (lambda/(2*m)).*sum(theta.^2);

k = length(theta)-1;
n = length(theta);

grad(1) = 1/m .* (sum(X'(1, :)*h - X'(1, :)*y));

for j = 2:n
  grad(j) = (1/m).*(sum(X'(j, :)*h - X'(j, :)*y) + lambda.*theta(j, 1)); 
end

end
