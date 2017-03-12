clear,close,%clc;

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

runTree = true;

n = length(time);
n0 = sum(class==0);
n1 = sum(class==1);
fracpos = sum(class)/n;

full = [data amount]; % time is not really important
posData = full(class == 1,:);
negData = full(class == 0,:);

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

%% Run model nIter times
nIter = 1;
ratio = 0.8;
nTest = ceil((1-ratio)*n);
runTree = true;

lda = struct;
lda.prediction = zeros(nIter,nTest);
lda.TPR = zeros(nIter,1);
lda.FPR = zeros(nIter,1);
lda.TNR = zeros(nIter,1);
lda.FNR = zeros(nIter,1);
lda.ACC = zeros(nIter,1);
lda.PPV = zeros(nIter,1);

tree = struct;
tree.TPR = zeros(nIter,1);
tree.FPR = zeros(nIter,1);
tree.TNR = zeros(nIter,1);
tree.FNR = zeros(nIter,1);
tree.ACC = zeros(nIter,1);
tree.PPV = zeros(nIter,1);
disp('Starting trials')
for i=1:nIter
    disp(['Running test ',num2str(i),' out of ',num2str(nIter)])
    t = tic;
    % Initialize test and train subsets
    [iTrain, iTest] = splitIndices(n,ratio);
    trainData = full(iTrain,:);
    testData = full(iTest,:);
    trainClasses = class(iTrain);
    testClasses = class(iTest);
    
    % Binary classification decision tree
    if runTree
        treefit = fitctree(trainData,trainClasses); %,'OptimizeHyperparameters','auto');
        tree.prediction(i,:) = predict(treefit, testData);
%         view(tree)
        [tree.TPR(i), tree.FPR(i), tree.TNR(i), tree.FNR(i), tree.ACC(i), tree.PPV(i)] = analyzePerformance(testClasses,tree.prediction(i,:)');
    end

    % Linear discriminant analysis
    lda.prediction(i,:) = classify(testData,trainData,trainClasses);

    % Evaluate performance
    [lda.TPR(i), lda.FPR(i), lda.TNR(i), lda.FNR(i), lda.ACC(i), lda.PPV(i)] = analyzePerformance(testClasses,lda.prediction(i,:)');
    t=toc(t);
    disp(['Test ',num2str(i),' took ',num2str(t),' seconds.'])
end
figure
hold on
scatter(lda.TPR,lda.PPV)
if runTree
    scatter(tree.TPR,tree.PPV)
end
plot([0,1],[0,1],'-r')
hold off
title('Precision-Recall')
xlabel('Recall')
ylabel('Precision')