function [error_train, error_val] = ...
    randomTrials(X, y, Xval, yval, lambda, num_trials)

## Copyright (C) 2018 Dileep Nackathaya
## 
## Determine the training error and cross validation error for
## i examples, you should first randomly select i examples from the training set
## and i examples from the cross validation set. You will then learn the param-
## eters θ using the randomly chosen training set and evaluate the parameters
## θ on the randomly chosen training set and cross validation set. The above
## steps should then be repeated multiple times (say 50) and the averaged error
## should be used to determine the training error and cross validation error for
## i examples.

##  Author: Dileep Nackathaya <dileepn@gmail.com>
##  Created: 2018-04-18

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% Loop over example sets of different sizes 
for i = 1:m

    % Initialize variables to accumulate error_train and error_val
    err_train = 0;
    err_val = 0;
    
%   Loop over number of trials    
    for j = 1:num_trials
    
        % Randomly select i examples
        randIndex_train = randperm(size(X, 1), i);
        randIndex_val = randperm(size(Xval, 1), i);
        
        % Get slices of X and Xval using these random indices
        X_rand_train = X(randIndex_train, :);
        y_rand_train = y(randIndex_train);
        X_rand_val = Xval(randIndex_val, :);
        y_rand_val = yval(randIndex_val);
        
        % Train model to get parameters, theta, and accumulate errors
        theta = trainLinearReg(X_rand_train, y_rand_train, lambda); 
        err_train += linearRegCostFunction(X_rand_train, ...
                                    y_rand_train, theta, 0); 
        err_val += linearRegCostFunction(X_rand_val, ...
                                        y_rand_val, theta, 0);
    endfor 
    
    % Store average errors in error_train and error_val vectors
    error_train(i) = err_train/num_trials;
    error_val(i) = err_val/num_trials;
    
endfor    

endfunction
