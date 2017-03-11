function plotFeatureSpace(posData, negData, features)
    assert(ismatrix(features));
    
    figure()
    scatter3(posData(:,features(1)),posData(:,features(2)),posData(:,features(3)))
    hold on
    scatter3(negData(:,features(1)),negData(:,features(2)),negData(:,features(3)))
    hold off
    title('Feature Space (projection onto POD modes)')
    legend('Fraudulent','Legitimate')
    xlabel(['Feature ' num2str(features(1))])
    ylabel(['Feature ' num2str(features(2))])
    zlabel(['Feature ' num2str(features(3))])
end