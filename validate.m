function [TP,FN,FP,TN] = validate(negIdx,posIdx,pre)
    % Perform error anlysis
    TP = sum( pre(posIdx) == 1 );
    TN = sum( pre(negIdx) == 0 );
    FN = sum( pre(posIdx) == 0 );
    FP = sum( pre(negIdx) == 1 );
end