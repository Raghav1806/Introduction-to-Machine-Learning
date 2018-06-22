function p = predict(Theta1, Theta2, X)
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to X data matrix
X = [ones(m, 1) X];

% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Map from layer 1 to layer 2
z1 = X*Theta1';
h1 = sigmoid(z1);

% Map from layer 2 to layer 3
% Add ones to h1 data matrix
h1 = [ones(m, 1) h1];
z2 = h1*Theta2';
h2 = sigmoid(z2);

% pval: highest value in each row
% p: position in each row
[pval, p] = max(h2, [], 2);

end
