function [TPR, FPR, TNR, FNR, PPV] = evaluate(TP,FN,FP,TN)


    TPR = TP./(TP + FN); % recall, sensitivity, hit rate (out of pos. values)
    FPR = FP./(FP + TN); % fall-out, false positivve rate, 1-TNR (out of neg. values)
    TNR = TN./(TN + FP); % aka specificity (out of neg. values)
    FNR = FN./(TP + FN); % aka miss rate, 1-TPR (out of pos. values)
    PPV = TP/(TP+FP);    % precision or positive predictive value (PPV)

end