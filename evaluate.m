function s = evaluate(model, testData, testClasses)
    s = struct;
    s.model = model;
    [s.labels,s.scores,s.costs] = predict(model, testData);
    s.confusion = confusionmat(testClasses,s.labels);
    [s.rocX,s.rocY,s.rocT,s.AUC] = perfcurve(testClasses,s.scores(:,2),'1');
    [s.prX,s.prY,s.prT,s.AUCPR] = perfcurve(testClasses,s.scores(:,2),'1','XCrit','reca','YCrit','prec');
%     disp('Bootstrapping ROC')
%     [s.rocX,s.rocY,s.rocT,s.AUC] = perfcurve(testClasses,s.scores(:,2),'1','NBoot',1,'TVals',0:0.05:1);
%     disp('Bootstrapping PR')
%     [s.prX,s.prY,s.prT,s.AUCPR] = perfcurve(testClasses,s.scores(:,2),'1','XCrit','reca','YCrit','prec','NBoot',1,'TVals',0:0.05:1);
end