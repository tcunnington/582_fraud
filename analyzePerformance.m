%% Compute performance characteristics for error analysis
% returns vector containing:
% Index | Name                | Description
%   1   | True positive rate  | Rate correctly classified positive, aka recall, sensitivity
%   2   | False positive rate | Rate incorrectly classified positive, aka fall-out
%   3   | True negative rate  | Rate correctly classified negative, aka specificity
%   4   | False negative rate | Rate incorrectly classified negative, aka miss rate
%   5   | Accuracy            | Rate correctly classified
%   6   | Error rate          | Rate incorrectly classified
%   7   | Precision           | Rate correct positive classifications
function errorChar = analyzePerformance(testClasses,prediction)
    TP = sum(prediction == 1 & prediction == testClasses);
    FP = sum(prediction == 1 & prediction ~= testClasses);
    TN = sum(prediction == 0 & prediction == testClasses);
    FN = sum(prediction == 0 & prediction ~= testClasses);
    TPR = TP./(TP + FN);
    FPR = FP./(FP + TN);
    TNR = TN./(TN + FP);
    FNR = FN./(TP + FN);
    accuracy = (TP+TN)./length(prediction);
    errorRate = 1-accuracy;
    precision = TP./(TP + FP);
    errorChar = [TPR;FPR;TNR;FNR;accuracy;errorRate;precision];
end