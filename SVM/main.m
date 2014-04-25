close all
clear all

numLabels = 10;
numTrain = 60000;
numTest = 10000;
cellSize = [4 4];

[train_imgs, train_labels] = readMNIST('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', numTrain, 0);
[test_imgs, test_labels] = readMNIST('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', numTest, 0);

[g, gn] = grp2idx(train_labels);

pairwise = nchoosek(1:length(gn),2);
svmModel = cell(size(pairwise,1),1);
predTest = zeros(numel(test_labels), numel(svmModel));

% hogFeatureSize = length(extractHOGFeatures(train_imgs(:,:,1), 'CellSize', cellSize));
% trainData = zeros(size(train_imgs,3), hogFeatureSize);
% testData = zeros(size(test_imgs,3), hogFeatureSize);
trainData = zeros(size(train_imgs,3), size(train_imgs,1) * size(train_imgs,2));
testData = zeros(size(test_imgs,3), size(test_imgs,1) * size(test_imgs,2));

for i = 1:size(trainData,1)
    trainData(i,:) = reshape(train_imgs(:,:,i), 1, size(trainData,2));
%     trainData(i, :) = extractHOGFeatures(train_imgs(:,:,i), 'CellSize', cellSize);
end

for i = 1:size(testData,1)
    testData(i,:) = reshape(test_imgs(:,:,i), 1, size(testData,2));
%     testData(i, :) = extractHOGFeatures(test_imgs(:,:,i), 'CellSize', cellSize);
end

for k = 1:numel(svmModel)
    idx = any(bsxfun(@eq, g, pairwise(k,:)), 2);
    k
    svmModel{k} = svmtrain(trainData(idx,:), g(idx), 'method', 'LS', 'kernel_function', 'polynomial', 'polyorder', 4);
    
    predTest(:,k) = svmclassify(svmModel{k}, testData) - 1;
end

pred = mode(predTest,2); % voting

cmat = confusionmat(test_labels, pred); % confusion matrix
acc = 100 * sum(diag(cmat)) ./ sum(cmat(:)); % accuracy

%Raw image => 95.9% accuracy

%HOG feature => 98.6% accuracy