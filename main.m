clear,close,clc;

% Load data
if exist('data.mat', 'file') == 2
    load('data.mat')
else
    readData
end

n = length(time);
fracpos = sum(class)/n;

full = [data amount]; % time is not really important

plotFeatureSpace(full(class == 1,:), full(class == 0,:), [3 10 29]);

%% Split data into test/train
[iTrain, iTest] = splitIndices(n,0.8);
trainData = full(iTrain,:);
testData = full(iTest,:);
nTrain = length(iTrain);
nTest = length(iTest);

%% Train and test model
tree = fitctree(trainData,class(iTrain));%,'OptimizeHyperparameters','auto');
pre = predict(tree, testData);
% view(tree)

testClasses = class(iTest);

%% Evaluate performance
tp = sum(pre == 1 & pre == testClasses);
fp = sum(pre == 1 & pre ~= testClasses);
tn = sum(pre == 0 & pre == testClasses);
fn = sum(pre == 0 & pre ~= testClasses);

% True/False Pos/Neg Rate
TPR = tp./(tp + fn); % aka recall, sensitivity (out of pos. values)
FPR = fp./(fp + tn); % aka fall-out, 1-TNR (out of neg. values)
TNR = tn./(tn + fp); % aka specificity (out of neg. values)
FNR = fn./(tp + fn); % aka miss rate, 1-TPR (out of pos. values)
accuracy = (tp+tn)./nTest;
errorRate = 1-accuracy;
% Plot TPR against FPR to get receiver operating characteristic curve (ROC)
% curve...
% plot(TNR,TPR) % when these are matrices

% PR curves
precision = tp./(tp + fp); % was pos. when you guessed pos.
recall = TPR;
% plot(recall, precision); % once these are vectors over a certain
% parameter...