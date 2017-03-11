function plotFeatureSpace(posData, negData, features)
    assert(ismatrix(features));
    
    figure()
    hold on
    scatter3(posData(:,features(1)),posData(:,features(2)),posData(:,features(3)))
    scatter3(negData(:,features(1)),negData(:,features(2)),negData(:,features(3)))
    hold off
end