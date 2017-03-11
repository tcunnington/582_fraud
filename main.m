clear,close,clc;

% Load data
if exist('data.mat', 'file') == 2
    load('data.mat')
else
    readData
end

n = length(time);
n0 = sum(class==0);
n1 = sum(class==1);
fracpos = sum(class)/n;

full = [data amount]; % time is not really important
posData = full(class == 1,:);
negData = full(class == 0,:);

% plotFeatureSpace(posData, negData, [3 10 29]);

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

%%

nIter = 100;
TPR = zeros(nIter,1);
FPR = zeros(nIter,1);
PPV = zeros(nIter,1);

for i=1:nIter
    % Take Random sample for cross-validation
    [negTrainInd, negTestInd, numNegTrain] = splitIndices(n0,0.8);
    [posTrainInd, posTestInd, numPosTrain] = splitIndices(n1,0.8);
    
    xtrain = [negData(negTrainInd,:); posData(posTrainInd,:)];
    xtest = [negData(negTestInd,:); posData(posTestInd,:)];
    ltrain = [zeros(numNegTrain,1)
              ones(numPosTrain,1)];
    
%     iNeg = 1:n0;
%     iPos = n0+1:n0+n1;
    
%     %% Gaussian Mixture Model
%     try % Fails in <5% of test sets
%         gm = fitgmdist(xtrain,2);
%     catch ME
%         continue
%     end
%     pre = cluster(gm,xtest);
%     [~,~,err(i,:),tot(i)] = validateUnsuper( 1:n1-nt1, n1-nt1+1:n1+n2-nt1-nt2, pre);

    % Linear discriminant analysis
    pre = classify(xtest,xtrain,ltrain);
    [TP,FN,FP,TN] = validate( iNeg, iPos, pre);
    TPR(i) = TP/(TP+FN); % sensitivity, recall, hit rate, or true positive rate (TPR)
    FPR(i) = FP/(FP+TN); % fall-out or false positive rate (FPR)
    PPV(i) = TP/(TP+FP); % precision or positive predictive value (PPV)
end
disp(mean(TPR))
disp(mean(FPR))
disp(mean(PPV))

%% Perform error anlysis
function [TP,FN,FP,TN] = validate(iNeg,iPos,pre)
    TP = sum( pre(iPos) == 1 );
    TN = sum( pre(iNeg) == 0 );
    FN = sum( pre(iPos) == 0 );
    FP = sum( pre(iNeg) == 1 );
end

%% Plot error results
% figure;
% hold on;
% plot(0:1:100,ones(1,101)*mean(tot*100),'-k','LineWidth',2);
% bar(tot*100);
% hold off
% legend(['Mean: ' num2str(mean(tot*100))])
% axis([0,100,0,100])
% xlabel('Trial')
% ylabel('Percent Incorrectly Classified')
% title('Total Error')
% figure;
% subplot(1,2,1);
% bar(err(:,1)*100);
% axis([0,100,0,100])
% ylabel('Percent Incorrectly Classified')
% title('Group 1')
% subplot(1,2,2);
% bar(err(:,2)*100);
% axis([0,100,0,100])
% xlabel('Trial')
% title('Group 2')