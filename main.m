clear,close all
tic
disp('Loading data and initializing')
%% Parameters
% Ratio of training to test subsets
ratio = 0.8;
% Desired ratio of class 0 to class 1 for ADASYN
classBalance = 0.1; 
% Cost of false negative for cost models
costs = [10 100 10^5 10^10];

%% Initialize variables
% amount  | amount of transaction in cents
% time    | time of transaction in seconds
% class   | boolean fraudulent/legitimate-->1/0
% data    | features (US in svd USV')

if exist('data.mat', 'file') == 2
    load('data.mat')
else
    readData
end

data = [data amount]; % time does not seem to improve results

%% Test and train subsets
% TODO use kfold cross validation instead--it's better and actually easier
% to implement.........
[trainData, testData, trainClasses, testClasses] = ...
    splitBinaryClassData(ratio, data, class);

%% Undersampling / Oversampling / ADASYN
disp(['Running ADASYN (',num2str(toc),')'])
% Scaling is required for ADASYN's internal KNN search
Ztrain = zscore(trainData);
Ztest = zscore(testData);
[synZ, synClassOut] = ADASYN(Ztrain, trainClasses, classBalance);
synFull = [Ztrain; synZ];
synClass = [trainClasses; synClassOut];

%% Classify
disp(['Building/ running classification models (',num2str(toc),')'])
% Binary classification decision tree
disp(['Binary classification decision tree (',num2str(toc),')']);
tree = evaluate(fitctree(Ztrain,trainClasses), Ztest, testClasses);

% Support vector machine
disp(['Support vector machine (',num2str(toc),')']);
svm = evaluate(fitcsvm(Ztrain, trainClasses), Ztest, testClasses);

% Linear discriminant analysis
disp(['Linear discriminant analyses (',num2str(toc),')']);
lda = evaluate(fitcdiscr(Ztrain, trainClasses), Ztest, testClasses);

% LDA with synthetic data -- train with syn data but test with original
ldaSyn = evaluate(fitcdiscr(synFull, synClass), Ztest, testClasses);

% LDA with custom cost function
costModel = fitcdiscr(Ztrain, trainClasses,'ClassNames',[0,1]);
for i=1:length(costs) % TODO is something wrong here?? all perf. values are exactly the same as vanilla LDA......
    costModel.Cost = [0 1; costs(i) 0];
    ldaCosts(i) = evaluate(costModel, testData, testClasses);
end

disp(['Done classifying (',num2str(toc),')'])


%% Plot
% lda
plotModels = {lda};
modelNames = {'LDA'};
AUCs = lda.AUC;
AUCPRs = lda.AUCPR;
% ldaSyn
plotModels = {plotModels{:},ldaSyn};
modelNames = {modelNames{:},'LDA w/ADASYN'};
AUCs = [AUCs;ldaSyn.AUC];
AUCPRs = [AUCPRs;ldaSyn.AUCPR];
% tree
plotModels = {plotModels{:},tree};
modelNames = {modelNames{:},'Tree'};
AUCs = [AUCs;tree.AUC];
AUCPRs = [AUCPRs;tree.AUCPR];
% svm
plotModels = {plotModels{:},svm};
modelNames = {modelNames{:},'SVM'};
AUCs = [AUCs;svm.AUC];
AUCPRs = [AUCPRs;svm.AUCPR];
% ldaCosts
for i=1:length(costs)
    plotModels = {plotModels{:},ldaCosts(i)};
    modelNames = {modelNames{:},['LDA w/Costs ',num2str(i)]};
    AUCs = [AUCs;ldaCosts(i).AUC];
    AUCPRs = [AUCPRs;ldaCosts(i).AUCPR];
end

T = table(AUCs,AUCPRs,'RowNames',modelNames)

% ROC Curve
figure
hold on
for i=1:length(plotModels)
    mdl = plotModels{i};
%     errorbar(mdl.rocX,mdl.rocY(:,1),mdl.rocY(:,1)-mdl.rocY(:,2),mdl.rocY(:,3)-mdl.rocY(:,1));
    plot(plotModels{i}.rocX,plotModels{i}.rocY);
end
hold off
legend(modelNames,'Location','Best');
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curves')

% PR Curve
figure
hold on
for i=1:length(plotModels)
    mdl = plotModels{i};
%     errorbar(mdl.prX,mdl.prY(:,1),mdl.prY(:,1)-mdl.prY(:,2),mdl.prY(:,3)-mdl.prY(:,1));
    plot(plotModels{i}.prX,plotModels{i}.prY);
end
hold off
legend(modelNames,'Location','Best')
xlabel('Recall')
ylabel('Precision')
title('Precision-Recall Curves')












%% Explore feature space
% Singular values and POD modes
% S = zeros(size(data,2),1);
% U = zeros(size(data));
% for i=1:size(data,2)
%     S(i)=norm(data(:,i));
%     U(:,i)=data(:,i)/S(i);
% end
% plot(S)
% plotFeatureSpace(U(class==1,:), U(class==0,:), [1 2 3]);
% plotFeatureSpace(posData, negData, [3 10 29]);    
%% Run model nIter times
% nIter = 1;
% disp(['Starting trials (',num2str(toc),')'])
% for i=1:nIter
%     disp(['Running test ',num2str(i),' out of ',num2str(nIter)])
%     t = tic;
%     t = toc(t);
%     disp(['Test ',num2str(i),' took ',num2str(t),' seconds.'])
% end
% % NOTE: if we are going to do cross validation MATLAB suggests:
%     cvmodel = crossval(lda,'kfold',2);
%     cverror = kfoldLoss(cvmodel)