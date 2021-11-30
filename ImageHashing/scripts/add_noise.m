clear ; close all; clc

% dataset = '.\Copydays\raw';
dataset = '..\data\COREL\query_database';
% dst = '.\Copydays\salt_and_pepper_noise';
% dst = '.\Copydays\speckle_noise';
dst = '..\data\COREL\test_images';
% params = [0.001: 0.001: 0.01];
params = [0.05];
raw = dir(dataset);
for i = 1: length(raw)
    file_name = raw(i).name;
    file = fullfile(dataset, file_name);
    if isequal(file_name, '.') || isequal(file_name, '..')
        continue;
    end
    M = imread(file);
    for j = 1: length(params)
        % J = imnoise(M, 'salt & pepper', params(j));
        % J = imnoise(M, 'speckle', params(j));
        J = imnoise(M, 'gaussian', 0, params(j));
        imshow(J);
        split = strsplit(file_name, '.');
        % image_name = [split{1} '_' num2str(j-1) '_' num2str(params(j)) '.' split{2}];
        image_name = [split{1} '_white_' num2str(params(j)) '.' split{2}];
        path = fullfile(dst, image_name);
        imwrite(J, path);
    end
end