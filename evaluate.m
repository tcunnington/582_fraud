function s = evaluate(model, testData, testClasses)

    s = struct;
    s.model = model;
    s.prediction = predict(model, testData);
    s.confusion = confusionmat(testClasses,s.prediction);
    [s.TPR, s.FPR, s.TNR, s.FNR, s.ACC, s.PPV] = analyzePerformance(testClasses,s.prediction);

end