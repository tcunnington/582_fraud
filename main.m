clear,close,clc;

% Load data
if exist('data.mat', 'file') == 2
    load('data.mat')
else
    readData
end

n = length(time);
fracpos = sum(class)/n;

full = [data amount time];

[iTrain, iTest] = splitIndices(n,0.8);
trainData = full(iTrain,:);
testData = full(iTest,:);
nTrain = length(iTrain);
nTest = length(iTest);

tree = fitctree(trainData,class(iTrain));
pre = predict(tree, testData);
% view(tree)

testClasses = class(iTest);

tp = sum(pre == 1 & pre == testClasses);
fp = sum(pre == 1 & pre ~= testClasses);
tn = sum(pre == 0 & pre == testClasses);
fn = sum(pre == 0 & pre ~= testClasses);

% True/False Pos/Neg Rate
TPR = tp/(tp + fn); % aka recall, sensitivity
FPR = fp/(fp + tn); % aka fall-out, 1-TNR
TNR = tn/(tn + fp); % aka specificity
FNR = fn/(tp + fn); % aka miss rate, 1-TPR
% Plot TPR against FPR to get receiver operating characteristic curve (ROC)
% curve...

% PR curves
precision = tp/(tp + fp); % was pos. when you guessed pos.
recall = TPR;
% plot...