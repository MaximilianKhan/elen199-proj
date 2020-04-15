%% load the pretrained model
alex = alexnet;
layers = alex.Layers;

%% Modify the network to have the new number of classifications (38)
layers(23) = fullyConnectedLayer(38);
layers(25) = classificationLayer;

%% Setup training data
trainImages = imageDatastore('train', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
trainImages.ReadFcn = @customerReadDatastoreImage;

testImages = imageDatastore('valid', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testImages.ReadFcn = @customerReadDatastoreImage;

%% Re-train the network
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.0001, 'MaxEpochs', 4, 'MiniBatchSize', 200, 'Plots', 'training-progress');
myNet = trainNetwork(trainImages, layers, opts);

%% Measure network accuracy
predictedLabels = classify(myNet, testImages);
accuracy = mean(predictedLabels == testImages.Labels)

%% Image processing function
function data = customerReadDatastoreImage(filename)
% code from default function:
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename); %added lines:
data = data(:, :, min(1:3, end));
data = imresize(data, [227 227]);
end