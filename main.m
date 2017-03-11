clear,close,clc;

% Load data
if exist('data.mat', 'file') == 2
    load('data.mat')
else
    readData
end

runTree = true;

n = length(time);
n0 = sum(class==0);
n1 = sum(class==1);
fracpos = sum(class)/n;

full = [data amount]; % time is not really important
posData = full(class == 1,:);
negData = full(class == 0,:);
% plotFeatureSpace(posData, negData, [3 10 29]);

%% Train and test models
nIter = 5;

ld = struct;
ld.TPR = zeros(nIter,1);
ld.FPR = zeros(nIter,1);
ld.TNR = zeros(nIter,1);
ld.FNR = zeros(nIter,1);
ld.PPV = zeros(nIter,1);

tree = struct;
tree.TPR = zeros(nIter,1);
tree.FPR = zeros(nIter,1);
tree.TNR = zeros(nIter,1);
tree.FNR = zeros(nIter,1);
tree.PPV = zeros(nIter,1);

disp('Start trials')
tic % timer. note toc command(s) below
for i=1:nIter
    % Split data for cross-validation
    [negTrainInd, negTestInd, numNegTrain, numNegTest] = splitIndices(n0,0.8);
    [posTrainInd, posTestInd, numPosTrain, numPosTest] = splitIndices(n1,0.8);
    
    xtrain = [negData(negTrainInd,:); posData(posTrainInd,:)];
    xtest  = [negData(negTestInd,:);  posData(posTestInd,:)];
    ltrain = [zeros(numNegTrain,1);   ones(numPosTrain,1)];
    
    negIdxTest = 1:numNegTest;
    posIdxTest = numNegTest+1:numNegTest+numPosTest;
    
%     %% Gaussian Mixture Model
%     try % Fails in <5% of test sets
%         gm = fitgmdist(xtrain,2);
%     catch ME
%         continue
%     end
%     pre = cluster(gm,xtest);
%     [~,~,err(i,:),tot(i)] = validateUnsuper( 1:n1-nt1, n1-nt1+1:n1+n2-nt1-nt2, pre);

    % Linear discriminant analysis
    ldpre = classify(xtest,xtrain,ltrain);
    [TP,FN,FP,TN] = validate( negIdxTest, posIdxTest, ldpre);
    [ld.TPR(i), ld.FPR(i), ld.TNR(i), ld.FNR(i), ld.PPV(i)] = evaluate(TP,FN,FP,TN);
    
    if runTree
        tree.model = fitctree(xtrain,ltrain); %,'OptimizeHyperparameters','auto');
        treepre = predict(tree.model, xtest);

        [TP,FN,FP,TN] = validate( negIdxTest, posIdxTest, treepre); 
        [tree.TPR(i), tree.FPR(i), tree.TNR(i), tree.FNR(i), tree.PPV(i)] = evaluate(TP,FN,FP,TN);
    end
    
    disp('iteration complete')
    disp(toc)
end

ld.numTrain = numNegTrain + numPosTrain;
ld.numTest = numNegTest + numPosTest;
tree.numTrain = ld.numTrain;
tree.numTest = ld.numTest;

%% Evaluate performance...
ld.meanTPR = mean(ld.TPR); % TPR = recall
ld.meanPPV = mean(ld.PPV); % PPV = precision
ld.meanFNR = mean(ld.FNR);

tree.meanTPR = mean(tree.TPR);
tree.meanPPV = mean(tree.PPV);
tree.meanFNR = mean(tree.FNR);


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