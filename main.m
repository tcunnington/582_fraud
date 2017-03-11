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
nIter = 10;
errorChar = zeros(7,nIter);
for i=1:nIter
    disp(['Running test ',num2str(i),' out of ',num2str(nIter)])
    % Initialize test and train subsets
    [iTrain, iTest] = splitIndices(n,0.8);
    trainData = full(iTrain,:);
    testData = full(iTest,:);
    trainClasses = class(iTrain);
    testClasses = class(iTest);
    
    % Binary classification decision tree
%     tree = fitctree(trainData,class(iTrain)); %,'OptimizeHyperparameters','auto');
%     prediction = predict(tree, testData);
% %     view(tree)

    % Linear discriminant analysis
    prediction = classify(testData,trainData,trainClasses);

    % Gaussian Mixture Model
%     gm = fitgmdist(trainData,2);
%     prediction = cluster(gm,testData);

    % Evaluate performance
    errorChar(:,i) = analyzePerformance(testClasses,prediction);
end
%% Interpret error characteristics
% PR Curve
figure
hold on
scatter(errorChar(1,:),errorChar(7,:))
plot([0,1],[0,1],'-r')
hold off
title('Precision-Recall')
xlabel('Recall')
ylabel('Precision')