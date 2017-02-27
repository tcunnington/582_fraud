%% Import data from text file.
filename = 'creditcard.csv';
delimiter = ',';
startRow = 2;
formatSpec1 = '%*q%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%*s%*s%[^\n\r]';
formatSpec2 = '%f%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%[^\n\r]';
formatSpec3 = '%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%f%*s%[^\n\r]';
formatSpec4 = '%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%*q%q%[^\n\r]';
%% Open the text file.
fileID = fopen(filename,'r');
dataArray1 = textscan(fileID, formatSpec1, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
% fclose(fileID);
% fileID = fopen(filename,'r');
frewind(fileID)
dataArray2 = textscan(fileID, formatSpec2, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
frewind(fileID)
dataArray3 = textscan(fileID, formatSpec3, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
frewind(fileID)
dataArray4 = textscan(fileID, formatSpec4, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
%% Allocate imported array to matrices
data = zeros(size(dataArray1{:,1},1),28);
for i=1:28
    data(:,i) = dataArray1{:,i};
end
time = dataArray2{:, 1};
amount = dataArray3{:, 1};
% Convert the contents of columns containing numeric text to numbers.
% Replace non-numeric text with NaN.
raw = repmat({''},length(dataArray4{1}),length(dataArray4)-1);
for col=1:length(dataArray4)-1
    raw(1:length(dataArray4{col}),col) = dataArray4{col};
end
numericData = NaN(size(dataArray4{1},1),size(dataArray4,2));
% Converts text in the input cell array to numbers. Replaced non-numeric
% text with NaN.
rawData = dataArray4{1};
for row=1:size(rawData, 1);
    % Create a regular expression to detect and remove non-numeric prefixes and
    % suffixes.
    regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
    try
        result = regexp(rawData{row}, regexstr, 'names');
        numbers = result.numbers;
        
        % Detected commas in non-thousand locations.
        invalidThousandsSeparator = false;
        if any(numbers==',');
            thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
            if isempty(regexp(numbers, thousandsRegExp, 'once'));
                numbers = NaN;
                invalidThousandsSeparator = true;
            end
        end
        % Convert numeric text to numbers.
        if ~invalidThousandsSeparator;
            numbers = textscan(strrep(numbers, ',', ''), '%f');
            numericData(row, 1) = numbers{1};
            raw{row, 1} = numbers{1};
        end
    catch me
    end
end
% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells
class = cell2mat(raw(:, 1));
%% Clear temporary variables
clearvars filename delimiter startRow fileID i ans;
clearvars dataArray1 dataArray2 dataArray3 dataArray4 formatSpec1 formatSpec2 formatSpec3 formatSpec4
clearvars raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me R;
save('data.mat')