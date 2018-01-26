function [i_train, i_test, numTrain, numTest] = splitIndices(n, ratio)
    numTrain = floor(ratio*n);
    numTest = n - numTrain;

    % Mix it up
    ri = randperm(n);
    i_train = ri(1:numTrain);
    i_test  = ri(numTrain+1:end);
end