clear all;
close all;
if exist('data.mat', 'file') == 2
    load('data.mat')
else
    readData
end