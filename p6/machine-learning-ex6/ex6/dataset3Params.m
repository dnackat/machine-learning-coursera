function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Options for C and sigma
const_vec = [0.01 0.03 0.1 0.3 1 3 10];

% Matrix for storing prediction errors of tries
err_mat = zeros(length(const_vec), length(const_vec));

% Try all possible C and sigma combinations (8x8=64)
for i = 1:length(const_vec)
    for j = 1:length(const_vec)
        % Train model using selected C and sigma
        model = svmTrain(X, y, const_vec(i), ...
            @(x1, x2) gaussianKernel(x1, x2, const_vec(j)));
        
        % Predict labels on the cross-vaidation set
        predictions = svmPredict(model, Xval);
        
        % Compute the prediction error
        err_mat(i,j) = mean(double(predictions ~= yval));
    endfor
endfor

% Select the C and sigma that have the lowest prediction errors (on CV set)
minVal = min(min(err_mat));
[i, j] = find(err_mat == minVal);
C = const_vec(i);
sigma = const_vec(j);

% =========================================================================

end
