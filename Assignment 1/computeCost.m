function J = computeCost(X, y, theta)
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values

% number of training examples
m = length(y); 
% You need to return the following variables correctly 
J = 0;

% saving parameters of X, y, and theta 
[r1, c1] = size(X);
[r2, c2] = size(y);
[r3, c3] = size(theta);

% hypothesis function
mult = X*theta;

% computing the cost
for i = 1:r1
  for j = 1:c3
    J = J + 1/(2*m)*(mult(i)(j) - y(i)(j))**2;
  end
end

end
