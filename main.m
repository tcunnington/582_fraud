clear,close,%clc;
tic
disp('Loading data and initializing')
%% Initialize variables
% amount  | amount of transaction in cents
% time    | time of transaction in seconds
% class   | boolean fraudulent/legitimate-->1/0
% data    | features (US in svd USV')
% n       | total number of transactions
% n0      | number of legitimate transactions
% n1      | number of fraudulent transactions
% fracpos | fraction of fraudulent transactions 

if exist('data.mat', 'file') == 2
    load('data.mat')
else
    readData
end

runTree = false;

n = length(time);
n0 = sum(class==0);
n1 = sum(class==1);
fracpos = n1/n;

full = [data amount]; % time is not really important
% posData = full(class == 1,:);
% negData = full(class == 0,:);

%% Explore feature space
% Singular values and POD modes
S = zeros(size(data,2),1);
U = zeros(size(data));
for i=1:size(data,2)
    S(i)=norm(data(:,i));
    U(:,i)=data(:,i)/S(i);
end
% plot(S)
% plotFeatureSpace(U(class==1,:), U(class==0,:), [1 2 3]);
% plotFeatureSpace(posData, negData, [3 10 29]);


%% Test and train subsets
ratio = 0.8;
[trainData, testData, trainClasses, testClasses] = splitBinaryClassData(ratio, full, class);

%% Undersampling / Oversampling / ADASYN
% TODO apply ADASYN only to the training data....
disp(['Running ADASYN (',num2str(toc),')'])
% Scaling is required for ADASYN's internal KNN search
Ztrain = zscore(trainData);
Ztest = zscore(testData);
% ratiopos = n1/n0;
classBalance = 0.1; % desired ratio of class 0 to class 1
[synZ, synClassOut] = ADASYN(Ztrain, trainClasses, classBalance);
synFull = [Ztrain; synZ];
synClass = [trainClasses; synClassOut];

%% Classify
disp(['Building/ running classification models (',num2str(toc),')'])
% Binary classification decision tree
if runTree
    treefit = fitctree(trainData,trainClasses); %,'OptimizeHyperparameters','auto');
    tree = evaluate(treefit, testData, testClasses);
end

% Linear discriminant analysis
[iTrainSyn, ~] = splitIndices(size(synFull,1),ratio);
% Baseline / default
lda = evaluate(fitcdiscr(trainData, trainClasses), testData, testClasses);

% LDA with cusotm cost function
% NOTE: if you plan to vary cost many times use a single model and update
% model.Cost then evaluate again.
costs = [10 100 10^5 10^10];
costModel = fitcdiscr(Ztrain, trainClasses,'ClassNames',[0,1]);
for i=1:length(costs) % TODO is something wrong here?? all perf. values are exactly the same as vanilla LDA......
    costModel.Cost = [0 1; costs(i) 0];
%     R = confusionmat(costModel.Y,resubPredict(costModel))
    ldaCosts(i) = evaluate(costModel, testData, testClasses);
end

% LDA with synthetic data -- train with syn data but test with original
ldaSyn = evaluate(fitcdiscr(synFull, synClass), Ztest, testClasses);

disp(['Done classifying (',num2str(toc),')'])
    
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

%% Plot
plotModels = [lda ldaCosts ldaSyn];
figure(1)
hold on
for c = arrayfun(@(s) [s.TPR,s.PPV], plotModels, 'un',0)
    scatter(c{1}(1),c{1}(2))
end
% scatter(lda.TPR,lda.PPV)
% scatter(ldaSyn.TPR,ldaSyn.PPV)
% scatter(ldaCost.TPR,ldaCost.PPV)
if runTree
    scatter(tree.TPR,tree.PPV)
end
plot([0,1],[0,1],'-r')
legend('show','location','northwest')
hold off

title('Precision-Recall')
xlabel('Recall')
ylabel('Precision')