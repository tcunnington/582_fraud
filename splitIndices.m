function [i_train, i_test] = splitIndices(n, ratio)
    numTrain = floor(ratio*n);

    % Mix it up
    ri = randperm(n);
    i_train = ri(1:numTrain);
    i_test  = ri(numTrain+1:end);
end