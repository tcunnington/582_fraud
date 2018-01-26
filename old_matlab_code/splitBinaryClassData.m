function [train, test, trainClass, testClass] = splitBinaryClassData(ratio, data, class)
    n0 = sum(class==0);
    n1 = sum(class==1);
    
    data0 = data(class == 0,:);
    data1 = data(class == 1,:);
    
    [iTrain0, iTest0] = splitIndices(n0,ratio);
    [iTrain1, iTest1] = splitIndices(n1,ratio);
    
    train = [data1(iTrain1,:); data0(iTrain0,:)];
    test = [data1(iTest1,:); data0(iTest0,:)];
    
    trainClass = [ones(length(iTrain1),1); zeros(length(iTrain0),1)];
    testClass = [ones(length(iTest1),1); zeros(length(iTest0),1)];
end