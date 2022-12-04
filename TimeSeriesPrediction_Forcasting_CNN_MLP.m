clc; clear; close all;
%% ---------------------------- init Variabels ----------------------------
opt.PredictionHorizone  = 200; 

opt.Delays = 1;
opt.dataPreprocessMode  = 'None'; % 'None' 'Data Standardization' 'Data Normalization'
opt.learningMethod      = 'MLP';                 % 'MLP' 'LSTM' 'CNN'   
opt.trPercentage        = 0.8;                    %  divide data into Test  and Train dataset

% ---- General Deep Learning Parameters(LSTM and CNN General Parameters)
opt.maxEpochs     = 200;                         % maximum number of training Epoch in deeplearning algorithms.
opt.miniBatchSize = 256;                          % minimum batch size in deeplearning algorithms .
opt.executionEnvironment = 'gpu';                % 'cpu' 'gpu' 'auto'
opt.LR                   = 'adam';               % 'sgdm' 'rmsprop' 'adam'
opt.trainingProgress     = 'training-progress';% 'training-progress' 'none'.



% ------------- BILSTM parameters 
opt.NumOfHiddenLayers = 1;                        %  number of (bi)LSTM layers. integer number between 1 to 4.

opt.NumOfUnitsInFirstlayer  = 120;                %  number of (bi)LSTM units in the first  layer
opt.NumOfUnitsInSecondlayer = 75;                %  number of (bi)LSTM units in the second layer
opt.NumOfUnitsInThirdlayer  = 100;                 %  number of (bi)LSTM units in the third  layer
opt.NumOfUnitsInFourthlayer = 75;                 %  number of (bi)LSTM units in the forth  layer

opt.isUseBiLSTMLayer  = true;                    % if it is true the layer turn to the Bidirectional-LSTM and if it is false it will turn the units to the simple LSTM
opt.isUseDropoutLayer = false;                    % dropout layer avoid of bieng overfit
opt.DropoutValue      = 0.5;

% ------------- CNN parameters 
opt.isUsePretrainResNet50 = true;               % if it is true network will train a pretrained Resnet50 other wise it will train by a simple CNN 

% ------------- MLP parameters
opt.NumOfFeedForwardLeyars = 3;             % integer number between 1 to 3.

opt.NumOfNeuronsInFirstlayer  =30;         %  number of neurons in the first  layer
opt.NumOfNeuronsInSecondlayer = 20;         %  number of neurons in the second layer
opt.NumOfNeuronsInThirdlayer  = 10;         %  number of neurons in the third  layer
opt.i=7;                                     % number of input.
opt.trainFcn = 'trainlm';                   % 'trainlm' 'trainscg' 'traincgf' 'trainbr'
opt.maxItrations = 1000;                     % maximum number of training itration.
opt.showWindow             = true;          % display training window.
opt.showCommandLine        = true;          % display training process on workspace.

opt.isSavePredictedData    = false;         %  save output prediction on an excel file

%% --------------- load Data
data = loadData(opt);
if ~data.isDataRead
    return;
end
%% --------------- Train Network
[opt,data] = TrainData(opt,data);

%% --------------- Evaluate Data 
[opt,data] = EvaluationData(opt,data);

%% --------------- predict Future data
[opt,data] = PredictionData(opt,data);

%% --------------- save predicted data
SavePredictedData(opt,data)

%% ---------------------------- Local Functions ---------------------------
function data = loadData(opt)
 [chosenfile,chosendirectory] = uigetfile({'*.xlsx';'*.csv'},...
                       'Select Excel time series Data sets','data.xlsx');
   filePath = [chosendirectory chosenfile];
    if filePath ~= 0 
        data.DataFileName = chosenfile;
        data.CompleteData = readtable(filePath); 
        if size(data.CompleteData,2)>1
             warning('Input data should be an excel file with only one column!'); 
                 
             data.x = [];
             data.isDataRead = false;
             
        end
        data.seriesdataHeder = data.CompleteData.Properties.VariableNames(1,:);
        data.seriesdata = table2array(data.CompleteData(:,:));
        disp('Input data successfully read.');
        data.isDataRead = true;
       
        
        figure('Name','InputData','NumberTitle','off');
        plot(data.seriesdata); grid minor;
        title({['Mean = ' num2str(mean(data.seriesdata)) ', STD = ' num2str(std(data.seriesdata)) ];});
        if strcmpi(opt.dataPreprocessMode,'None')
            data.x = data.seriesdata;
        elseif strcmpi(opt.dataPreprocessMode,'Data Normalization')
            data.x = DataNormalization(data.seriesdata);
            figure('Name','NormilizedInputData','NumberTitle','off');
            plot(data.x); grid minor;
            title({['Mean = ' num2str(mean(data.x)) ', STD = ' num2str(std(data.x)) ];});
        elseif strcmpi(opt.dataPreprocessMode,'Data Standardization')
            data.x = DataStandardization(data.seriesdata);
            figure('Name','NormilizedInputData','NumberTitle','off');
            plot(data.x); grid minor;
            title({['Mean = ' num2str(mean(data.x)) ', STD = ' num2str(std(data.x)) ];});
        end
        
    else 
        warning(['In order to train network, please load data.' ...
        'Input data should be an excel file with only one column!']);    
        disp('Operation Cancel.'); 
        data.isDataRead = false;
    end
end

function vars = DataStandardization(data)
    for i=1:size(data,2)
        x.mu(1,i)   = mean(data(:,i),'omitnan');
        x.sig(1,i)  = std (data(:,i),'omitnan');
        vars(:,i) = (data(:,i) - x.mu(1,i))./ x.sig(1,i);
    end
end
function vars = DataNormalization(data)
    for i=1:size(data,2)
        vars(:,i) = (data(:,i) -min(data(:,i)))./ (max(data(:,i))-min(data(:,i)));
    end
end
% --------------- Train Network ---------------
% ---------------------------------------------
function [opt,data] = TrainData(opt,data)
% Bi-LSTM parameters
if opt.NumOfHiddenLayers ==1
    opt.numHiddenUnits1 = opt.NumOfUnitsInFirstlayer;   
elseif opt.NumOfHiddenLayers ==2
    opt.numHiddenUnits1 = opt.NumOfUnitsInFirstlayer; 
    opt.numHiddenUnits2 = opt.NumOfUnitsInSecondlayer;    
elseif opt.NumOfHiddenLayers ==3
    opt.numHiddenUnits1 = opt.NumOfUnitsInFirstlayer; 
    opt.numHiddenUnits2 = opt.NumOfUnitsInSecondlayer;  
    opt.numHiddenUnits3 = opt.NumOfUnitsInThirdlayer; 
elseif opt.NumOfHiddenLayers ==4
    opt.numHiddenUnits1 = opt.NumOfUnitsInFirstlayer; 
    opt.numHiddenUnits2 = opt.NumOfUnitsInSecondlayer;  
    opt.numHiddenUnits3 = opt.NumOfUnitsInThirdlayer; 
    opt.numHiddenUnits4 = opt.NumOfUnitsInFourthlayer; 
end
% MLP parameters
if opt.NumOfFeedForwardLeyars ==1
    opt.ShallowhiddenLayerSize = [opt.NumOfNeuronsInFirstlayer];                                       % number of Hidden layers in MLP network.
elseif opt.NumOfFeedForwardLeyars ==2
    opt.ShallowhiddenLayerSize = [opt.NumOfNeuronsInFirstlayer opt.NumOfNeuronsInSecondlayer];          % number of Hidden layers in MLP network. 
elseif opt.NumOfFeedForwardLeyars ==3
    opt.ShallowhiddenLayerSize = [opt.NumOfNeuronsInFirstlayer opt.NumOfNeuronsInSecondlayer opt.NumOfNeuronsInThirdlayer];% number of Hidden layers in MLP network.
end



% prepare delays for time serie network
data = CreateTimeSeriesData(opt,data);

% divide data into test and train data
data = dataPartitioning(opt,data);

if strcmpi(opt.learningMethod,'LSTM')
    % LSTM data form
    data = LSTMInput(data);
    %  Define LSTM  architect 
    opt = LSTMArchitect(opt,data);
elseif strcmpi(opt.learningMethod,'CNN')
    data = CNNInput(data);
    opt  = CNNArchitect(opt); 
elseif strcmpi(opt.learningMethod,'MLP')
    % Prepare input data for MLP network.
    FeedForwardInput();
    %  Define MLP architect 
    opt  = FeedForwardArchitect(opt);    
end

% Train LSTM, MLP 
data = TrainNet(opt,data);

end
% make some delays on input filed
function data = CreateTimeSeriesData(opt,data)
    Delays = opt.Delays;

    x = data.x';
    data.M=x;
    i=opt.i
    y = x([i+1],:);
    x=x([1:i],:);
    data.X  = x;
    data.Y  = y;
end
% partitioning input data 
function data = dataPartitioning(opt,data)
data.XTr   = [];
data.YTr   = [];
data.XTs   = [];
data.YTs   = [];

numTrSample = round(opt.trPercentage*size(data.X,2));

data.XTr   = data.X(:,1:numTrSample);
data.YTr   = data.Y(:,1:numTrSample);

data.XTs   = data.X(:,numTrSample+1:end);
data.YTs   = data.Y(:,numTrSample+1:end);
disp(['Time Series data divided to ' num2str(opt.trPercentage*100) '% Train data and ' num2str((1-opt.trPercentage)*100) '% Test data']);
end
% Prepare input data for MLP network.
function FeedForwardInput()
    disp('Time Series data prepared as suitable MLP Input data.');
end
% Prepare input data for LSTM network.
function data = LSTMInput(data)

for i=1:size(data.XTr,2)
    XTr{i,1} = data.XTr(:,i);  
    YTr(i,1) = data.YTr(:,i);  
end

for i=1:size(data.XTs,2)
    XTs{i,1} =  data.XTs(:,i);  
    YTs(i,1) =  data.YTs(:,i); 
end
data.XTr   = XTr;
data.YTr   = YTr;
data.XTs   = XTs;
data.YTs   = YTs;

disp('Time Series data prepared as suitable LSTM Input data.');
end
function data = CNNInput(data)
% reshape 2D data to 4D data
data.XTr = reshape(data.XTr, [size(data.XTr,1),1,1,size(data.XTr,2)]);
data.XTs = reshape(data.XTs, [size(data.XTs,1),1,1,size(data.XTs,2)]);

for i=1:size(data.YTr,2)
    YTr(i,1) = data.YTr(:,i);  
end
for i=1:size(data.YTs,2)
    YTs(i,1) = data.YTs(:,i); 
end
data.YTr  = YTr;  
data.YTs = YTs; 

data.input = [];
data.target= [];
disp('Time Series data prepared as suitable CNN Input data.');
end

% ---- network structure ----
% bi-LSTM Deeplearning Architect
function opt = LSTMArchitect(opt,data)

miniBatchSize   = opt.miniBatchSize;
maxEpochs       = opt.maxEpochs;
trainingProgress = opt.trainingProgress;
executionEnvironment = opt.executionEnvironment;

inputSize = size(4,1);
outputMode = 'last';
numResponses   = 1;
dropoutVal = opt.DropoutValue;
if opt.isUseDropoutLayer % if dropout layer is true
    if opt.NumOfHiddenLayers ==1
        if opt.isUseBiLSTMLayer == 1
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(opt.numHiddenUnits1,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(opt.numHiddenUnits1,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif opt.NumOfHiddenLayers ==2
        if opt.isUseBiLSTMLayer
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            bilstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            bilstmLayer(opt.numHiddenUnits2,'OutputMode',outputMode)
            dropoutLayer(dropoutVal)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        else
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            lstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            lstmLayer(opt.numHiddenUnits2,'OutputMode',outputMode)
            dropoutLayer(dropoutVal)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        end
    elseif opt.NumOfHiddenLayers ==3
        if opt.isUseBiLSTMLayer
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            bilstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            bilstmLayer(opt.numHiddenUnits2,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            bilstmLayer(opt.numHiddenUnits3,'OutputMode',outputMode)
            dropoutLayer(dropoutVal)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        else
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            bilstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            bilstmLayer(opt.numHiddenUnits2,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            bilstmLayer(opt.numHiddenUnits3,'OutputMode',outputMode)
            dropoutLayer(dropoutVal)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        end
    elseif opt.NumOfHiddenLayers ==4
        if opt.isUseBiLSTMLayer
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            bilstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            bilstmLayer(opt.numHiddenUnits2,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            bilstmLayer(opt.numHiddenUnits3,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            bilstmLayer(opt.numHiddenUnits4,'OutputMode',outputMode)
            dropoutLayer(dropoutVal)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        else
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            bilstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            bilstmLayer(opt.numHiddenUnits2,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            bilstmLayer(opt.numHiddenUnits3,'OutputMode','sequence')
            dropoutLayer(dropoutVal)
            bilstmLayer(opt.numHiddenUnits4,'OutputMode',outputMode)
            dropoutLayer(dropoutVal)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        end
    end
else % if dropout layer is false
    if opt.NumOfHiddenLayers ==1
        if opt.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(opt.numHiddenUnits1,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(opt.numHiddenUnits1,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif opt.NumOfHiddenLayers ==2
        if opt.isUseBiLSTMLayer
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            bilstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            bilstmLayer(opt.numHiddenUnits2,'OutputMode',outputMode)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        else
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            lstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            lstmLayer(opt.numHiddenUnits2,'OutputMode',outputMode)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        end
    elseif opt.NumOfHiddenLayers ==3
        if opt.isUseBiLSTMLayer
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            bilstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            bilstmLayer(opt.numHiddenUnits2,'OutputMode','sequence')
            bilstmLayer(opt.numHiddenUnits3,'OutputMode',outputMode)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        else
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            bilstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            bilstmLayer(opt.numHiddenUnits2,'OutputMode','sequence')
            bilstmLayer(opt.numHiddenUnits3,'OutputMode',outputMode)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        end
    elseif opt.NumOfHiddenLayers ==4
        if opt.isUseBiLSTMLayer
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            bilstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            bilstmLayer(opt.numHiddenUnits2,'OutputMode','sequence')
            bilstmLayer(opt.numHiddenUnits3,'OutputMode','sequence')
            bilstmLayer(opt.numHiddenUnits4,'OutputMode',outputMode)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        else
            opt.layers = [ ...
            sequenceInputLayer(inputSize)
            bilstmLayer(opt.numHiddenUnits1,'OutputMode','sequence')
            bilstmLayer(opt.numHiddenUnits2,'OutputMode','sequence')
            bilstmLayer(opt.numHiddenUnits3,'OutputMode','sequence')
            bilstmLayer(opt.numHiddenUnits4,'OutputMode',outputMode)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        end
    end
end
% Training Network Options
% 'sgdm' 
% 'rmsprop'
% 'adam'

opt.opts = trainingOptions(opt.LR, ...
    'MaxEpochs',maxEpochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',1, ...
    'MiniBatchSize',miniBatchSize,...
    'ExecutionEnvironment',executionEnvironment,...
    'Plots',trainingProgress);
    disp('LSTM architect successfully created.');
end
% CNN Deeplearning Architect
function opt = CNNArchitect(opt)
miniBatchSize  = opt.miniBatchSize;
maxEpochs      = opt.maxEpochs;
trainingProgress = opt.trainingProgress;
executionEnvironment = opt.executionEnvironment;
numResponses   = 1;
inputSize = size(opt.i,2);

if opt.isUsePretrainResNet50 
    opt = pretrainedNets(opt);
else
    opt.layers = [
        imageInputLayer([opt.i 1],'name','Input')
        convolution2dLayer(2,16,'Padding',1)
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(2,32,'Padding',1)
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer(2,'Stride',1)

        convolution2dLayer(2,64,'Padding',1)
        batchNormalizationLayer
        reluLayer

        fullyConnectedLayer(numResponses, 'Name','fc')
        regressionLayer('Name','regression','Name', 'OutputRegression')];
end

opt.opts = trainingOptions(opt.LR,...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',maxEpochs,...
    'Plots',trainingProgress,...
    'MiniBatchSize',miniBatchSize,...
    'ExecutionEnvironment',executionEnvironment);
end
function opt = pretrainedNets(opt)
opt.layers = Resnet50(opt);
% Plot Layers
figure, plot(opt.layers);title('ResNet50 Architect')
end
% pretrained ResNet Function
function lgraph = Resnet50(opt)
lgraph = layerGraph();

% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear array of layers.
tempLayers = [
    imageInputLayer([opt.i 1],"Name","input_1")
    convolution2dLayer([7 7],64,"Name","conv1","Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","bn_conv1","Epsilon",0.001)
    reluLayer("Name","activation_1_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2a_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_2_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_3_relu")
    convolution2dLayer([1 1],256,"Name","res2a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res2a_branch1","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_1")
    reluLayer("Name","activation_4_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_5_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_6_relu")
    convolution2dLayer([1 1],256,"Name","res2b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_2")
    reluLayer("Name","activation_7_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_8_relu")
    convolution2dLayer([3 3],64,"Name","res2c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_9_relu")
    convolution2dLayer([1 1],256,"Name","res2c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_3")
    reluLayer("Name","activation_10_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_11_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_12_relu")
    convolution2dLayer([1 1],512,"Name","res3a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_4")
    reluLayer("Name","activation_13_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_14_relu")
    convolution2dLayer([3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_15_relu")
    convolution2dLayer([1 1],512,"Name","res3b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_5")
    reluLayer("Name","activation_16_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_17_relu")
    convolution2dLayer([3 3],128,"Name","res3c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_18_relu")
    convolution2dLayer([1 1],512,"Name","res3c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_6")
    reluLayer("Name","activation_19_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3d_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3d_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_20_relu")
    convolution2dLayer([3 3],128,"Name","res3d_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3d_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_21_relu")
    convolution2dLayer([1 1],512,"Name","res3d_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3d_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_7")
    reluLayer("Name","activation_22_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_23_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_24_relu")
    convolution2dLayer([1 1],1024,"Name","res4a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1024,"Name","res4a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_8")
    reluLayer("Name","activation_25_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_26_relu")
    convolution2dLayer([3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_27_relu")
    convolution2dLayer([1 1],1024,"Name","res4b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_9")
    reluLayer("Name","activation_28_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_29_relu")
    convolution2dLayer([3 3],256,"Name","res4c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_30_relu")
    convolution2dLayer([1 1],1024,"Name","res4c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_10")
    reluLayer("Name","activation_31_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4d_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4d_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_32_relu")
    convolution2dLayer([3 3],256,"Name","res4d_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4d_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_33_relu")
    convolution2dLayer([1 1],1024,"Name","res4d_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4d_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_11")
    reluLayer("Name","activation_34_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4e_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4e_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_35_relu")
    convolution2dLayer([3 3],256,"Name","res4e_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4e_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_36_relu")
    convolution2dLayer([1 1],1024,"Name","res4e_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4e_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_12")
    reluLayer("Name","activation_37_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4f_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4f_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_38_relu")
    convolution2dLayer([3 3],256,"Name","res4f_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn4f_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_39_relu")
    convolution2dLayer([1 1],1024,"Name","res4f_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn4f_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_13")
    reluLayer("Name","activation_40_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],2048,"Name","res5a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_41_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn5a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_42_relu")
    convolution2dLayer([1 1],2048,"Name","res5a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_14")
    reluLayer("Name","activation_43_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_44_relu")
    convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_45_relu")
    convolution2dLayer([1 1],2048,"Name","res5b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_15")
    reluLayer("Name","activation_46_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_47_relu")
    convolution2dLayer([3 3],512,"Name","res5c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn5c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_48_relu")
    convolution2dLayer([1 1],2048,"Name","res5c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn5c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_16")
    reluLayer("Name","activation_49_relu")
    globalAveragePooling2dLayer("Name","avg_pool")
    fullyConnectedLayer(1,"Name","fc1000","BiasLearnRateFactor",0)
    regressionLayer("Name","RegressionLayer_fc1000")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

% Connect Layer Branches
% Connect all the branches of the network to create the network graph.
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch2a");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch1");
lgraph = connectLayers(lgraph,"bn2a_branch2c","add_1/in1");
lgraph = connectLayers(lgraph,"bn2a_branch1","add_1/in2");
lgraph = connectLayers(lgraph,"activation_4_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"activation_4_relu","add_2/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2c","add_2/in1");
lgraph = connectLayers(lgraph,"activation_7_relu","res2c_branch2a");
lgraph = connectLayers(lgraph,"activation_7_relu","add_3/in2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","add_3/in1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"bn3a_branch1","add_4/in2");
lgraph = connectLayers(lgraph,"bn3a_branch2c","add_4/in1");
lgraph = connectLayers(lgraph,"activation_13_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"activation_13_relu","add_5/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2c","add_5/in1");
lgraph = connectLayers(lgraph,"activation_16_relu","res3c_branch2a");
lgraph = connectLayers(lgraph,"activation_16_relu","add_6/in2");
lgraph = connectLayers(lgraph,"bn3c_branch2c","add_6/in1");
lgraph = connectLayers(lgraph,"activation_19_relu","res3d_branch2a");
lgraph = connectLayers(lgraph,"activation_19_relu","add_7/in2");
lgraph = connectLayers(lgraph,"bn3d_branch2c","add_7/in1");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"bn4a_branch1","add_8/in2");
lgraph = connectLayers(lgraph,"bn4a_branch2c","add_8/in1");
lgraph = connectLayers(lgraph,"activation_25_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"activation_25_relu","add_9/in2");
lgraph = connectLayers(lgraph,"bn4b_branch2c","add_9/in1");
lgraph = connectLayers(lgraph,"activation_28_relu","res4c_branch2a");
lgraph = connectLayers(lgraph,"activation_28_relu","add_10/in2");
lgraph = connectLayers(lgraph,"bn4c_branch2c","add_10/in1");
lgraph = connectLayers(lgraph,"activation_31_relu","res4d_branch2a");
lgraph = connectLayers(lgraph,"activation_31_relu","add_11/in2");
lgraph = connectLayers(lgraph,"bn4d_branch2c","add_11/in1");
lgraph = connectLayers(lgraph,"activation_34_relu","res4e_branch2a");
lgraph = connectLayers(lgraph,"activation_34_relu","add_12/in2");
lgraph = connectLayers(lgraph,"bn4e_branch2c","add_12/in1");
lgraph = connectLayers(lgraph,"activation_37_relu","res4f_branch2a");
lgraph = connectLayers(lgraph,"activation_37_relu","add_13/in2");
lgraph = connectLayers(lgraph,"bn4f_branch2c","add_13/in1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"bn5a_branch1","add_14/in2");
lgraph = connectLayers(lgraph,"bn5a_branch2c","add_14/in1");
lgraph = connectLayers(lgraph,"activation_43_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"activation_43_relu","add_15/in2");
lgraph = connectLayers(lgraph,"bn5b_branch2c","add_15/in1");
lgraph = connectLayers(lgraph,"activation_46_relu","res5c_branch2a");
lgraph = connectLayers(lgraph,"activation_46_relu","add_16/in2");
lgraph = connectLayers(lgraph,"bn5c_branch2c","add_16/in1");
end
% MLP Shallowlearning Architect
function opt = FeedForwardArchitect(opt)
opt.Net = feedforwardnet(opt.ShallowhiddenLayerSize,opt.trainFcn);
opt.Net.divideParam.trainRatio = 80/100;
opt.Net.divideParam.valRatio  = 10/100;
opt.Net.divideParam.testRatio = 10/100;

opt.Net.trainParam.epochs          = opt.maxItrations;
opt.Net.trainParam.showWindow      = opt.showWindow;
opt.Net.trainParam.showCommandLine = opt.showCommandLine;
disp('MLP architect successfully created.');

end
% Train  Network
function data = TrainNet(opt,data)

if strcmpi(opt.learningMethod,'LSTM')
    try
        data.BiLSTM.Net = trainNetwork(data.XTr,data.YTr,opt.layers,opt.opts);
        disp('LSTM Netwwork successfully trained.');
        data.IsNetTrainSuccess =true;
    catch me
        disp('Error on Training LSTM Network');
        data.IsNetTrainSuccess = false;
        return;
    end
elseif strcmpi(opt.learningMethod,'CNN')
    try
        data.CNN.Net = trainNetwork(data.XTr,data.YTr,opt.layers,opt.opts);
        disp('CNN Netwwork successfully trained.');
        data.IsNetTrainSuccess =true;
    catch me
        disp('Error on Training LSTM Network');
        data.IsNetTrainSuccess = false;
        return;
    end
elseif strcmpi(opt.learningMethod,'MLP')
    try
        [data.FF.Net,~] = train(opt.Net,data.XTr,data.YTr);
        disp('Feed Forward Netwwork successfully trained.');
        data.IsNetTrainSuccess = true;
    catch me
        disp('Error on Training FF Network');
        data.IsNetTrainSuccess =false;
        return;
    end
end

end


% --------------- Evaluate Data ---------------
% ---------------------------------------------
function [opt,data] = EvaluationData(opt,data)
if strcmpi(opt.learningMethod,'LSTM')
    
    data.BiLSTM.TrainOutputs = deNorm(data.seriesdata,predict(data.BiLSTM.Net,data.XTr,'MiniBatchSize',opt.miniBatchSize),opt.dataPreprocessMode);
    data.BiLSTM.TrainTargets = deNorm(data.seriesdata,data.YTr,opt.dataPreprocessMode);
    data.BiLSTM.TestOutputs  = deNorm(data.seriesdata,predict(data.BiLSTM.Net,data.XTs,'MiniBatchSize',opt.miniBatchSize),opt.dataPreprocessMode);
    data.BiLSTM.TestTargets  = deNorm(data.seriesdata,data.YTs,opt.dataPreprocessMode);
    data.BiLSTM.AllDataTargets = [data.BiLSTM.TrainTargets data.BiLSTM.TestTargets];
    data.BiLSTM.AllDataOutputs = [data.BiLSTM.TrainOutputs data.BiLSTM.TestOutputs];
   
    data = PlotResults(data,'Tr',...
        data.BiLSTM.TrainOutputs, ...
        data.BiLSTM.TrainTargets);
    data = plotReg(data,'Tr',data.BiLSTM.TrainTargets,data.BiLSTM.TrainOutputs);

    data = PlotResults(data,'Ts',....
        data.BiLSTM.TestOutputs, ...
        data.BiLSTM.TestTargets);
    data = plotReg(data,'Ts',data.BiLSTM.TestTargets,data.BiLSTM.TestOutputs);

    data = PlotResults(data,'All',...
        data.BiLSTM.AllDataOutputs, ...
        data.BiLSTM.AllDataTargets);
    data = plotReg(data,'All',data.BiLSTM.AllDataTargets,data.BiLSTM.AllDataOutputs);
    
    disp('Bi-LSTM network performance evaluated.');

elseif strcmpi(opt.learningMethod,'CNN')
    
    data.CNN.TrainOutputs = deNorm(data.seriesdata,predict(data.CNN.Net,data.XTr,'MiniBatchSize',opt.miniBatchSize),opt.dataPreprocessMode);
    data.CNN.TrainTargets = deNorm(data.seriesdata,data.YTr,opt.dataPreprocessMode);
    data.CNN.TestOutputs  = deNorm(data.seriesdata,predict(data.CNN.Net,data.XTs,'MiniBatchSize',opt.miniBatchSize),opt.dataPreprocessMode);
    data.CNN.TestTargets  = deNorm(data.seriesdata,data.YTs,opt.dataPreprocessMode);
    data.CNN.AllDataTargets = [data.CNN.TrainTargets data.CNN.TestTargets];
    data.CNN.AllDataOutputs = [data.CNN.TrainOutputs data.CNN.TestOutputs];
   
    data = PlotResults(data,'Tr',...
        data.CNN.TrainOutputs, ...
        data.CNN.TrainTargets);
    data = plotReg(data,'Tr',data.CNN.TrainTargets,data.CNN.TrainOutputs);

    data = PlotResults(data,'Ts',....
        data.CNN.TestOutputs, ...
        data.CNN.TestTargets);
    data = plotReg(data,'Ts',data.CNN.TestTargets,data.CNN.TestOutputs);

    data = PlotResults(data,'All',...
        data.CNN.AllDataOutputs, ...
        data.CNN.AllDataTargets);
    data = plotReg(data,'All',data.CNN.AllDataTargets,data.CNN.AllDataOutputs);
    
    disp('CNN network performance evaluated.');
elseif strcmpi(opt.learningMethod,'MLP')
    
        data.FF.TrainOutputs = deNorm(data.seriesdata,data.FF.Net(data.XTr)',opt.dataPreprocessMode);
        data.FF.TrainTargets = deNorm(data.seriesdata,(data.YTr)',opt.dataPreprocessMode);
        data.FF.TestOutputs  = deNorm(data.seriesdata,data.FF.Net(data.XTs)',opt.dataPreprocessMode);
        data.FF.TestTargets  = deNorm(data.seriesdata,(data.YTs)',opt.dataPreprocessMode);

    data.FF.AllDataTargets = [data.FF.TrainTargets data.FF.TestTargets];
    data.FF.AllDataOutputs = [data.FF.TrainOutputs data.FF.TestOutputs];
 
    DispVal = 1;
    for i= DispVal
        data = PlotResults(data,'Tr',...
            data.FF.TrainOutputs(i,:), ...
            data.FF.TrainTargets(i,:));
        data = plotReg(data,'Tr',data.FF.TrainTargets(i,:),data.FF.TrainOutputs(i,:));
        data = PlotResults(data,'Ts',....
            data.FF.TestOutputs(i,:), ...
            data.FF.TestTargets(i,:));
        data = plotReg(data,'Ts',data.FF.TestTargets(i,:),data.FF.TestOutputs(i,:));
        data = PlotResults(data,'All',...
            data.FF.AllDataOutputs(i,:), ...
            data.FF.AllDataTargets(i,:));
        data = plotReg(data,'All',data.FF.AllDataTargets(i,:),data.FF.AllDataOutputs(i,:));
        disp('MLP network performance evaluated.');
    end

end
end
function vars = deNorm(data,stdData,deNormMode)
if iscell(stdData(1,1))
        for i=1:size(stdData,1)
            tmp(i,:) = stdData{i,1}';
        end
        stdData = tmp;
end
if strcmpi(deNormMode,'Data Normalization')
    for i=1:size(data,2)
        vars(:,i) = (stdData(:,i).*(max(data(:,i))-min(data(:,i)))) + min(data(:,i));
    end
    vars = vars';
    
elseif strcmpi(deNormMode,'Data Standardization')
    for i=1:size(data,2)
        x.mu(1,i)   = mean(data(:,i),'omitnan');
        x.sig(1,i)  = std (data(:,i),'omitnan');
        vars(:,i) = ((stdData(:,i).* x.sig(1,i))+ x.mu(1,i));
    end
    vars = vars';
    
else
    vars = stdData';
    return;
end
end
% plot the output of networks and real output on test and train data
function data = PlotResults(data,firstTitle,Outputs,Targets)  
Errors = Targets - Outputs;
    MSE   = mean(Errors.^2);
    RMSE  = sqrt(MSE);
    NRMSE = RMSE/mean(Targets);
    ErrorMean = mean(Errors);
    ErrorStd  = std(Errors);
    rankCorre = RankCorre(Targets,Outputs);

    if strcmpi(firstTitle,'tr')
         Disp1Name = 'OutputGraphEvaluation_TrainData';
         Disp2Name = 'ErrorEvaluation_TrainData';
         Disp3Name = 'ErrorHistogram_TrainData';
    elseif strcmpi(firstTitle,'ts')
        Disp1Name = 'OutputGraphEvaluation_TestData';
        Disp2Name = 'ErrorEvaluation_TestData';
        Disp3Name = 'ErrorHistogram_TestData';
    elseif strcmpi(firstTitle,'all')
        Disp1Name = 'OutputGraphEvaluation_ALLData';
        Disp2Name = 'ErrorEvaluation_ALLData';
        Disp3Name = 'ErrorHistogram_AllData';
    end
    
    figure('Name',Disp1Name,'NumberTitle','off');
    plot(1:length(Targets),Targets,...
        1:length(Outputs),Outputs);grid minor
    legend('Targets','Outputs','Location','best') ;
    title(['Rank Correlation = ' num2str(rankCorre)]);
   
    figure('Name',Disp2Name,'NumberTitle','off');
    plot(Errors);grid minor
    title({['MSE = ' num2str(MSE) ', RMSE = ' num2str(RMSE)...
        ' NRMSE = ' num2str(NRMSE)] ;});
    xlabel(['Error Per Sample']);
 
    figure('Name',Disp3Name,'NumberTitle','off');
    histogram(Errors);grid minor

    title(['Error Mean = ' num2str(ErrorMean) ', Error StD = ' num2str(ErrorStd)]);
    xlabel(['Error Histogram']);
    
    if strcmpi(firstTitle,'tr')
        data.Err.MSETr = MSE;
        data.Err.STDTr = ErrorStd;
        data.Err.NRMSETr     = NRMSE;
        data.Err.rankCorreTr = rankCorre;
    elseif strcmpi(firstTitle,'ts')
        data.Err.MSETs = MSE;
        data.Err.STDTs = ErrorStd;
        data.Err.NRMSETs     = NRMSE;
        data.Err.rankCorreTs = rankCorre;
    elseif strcmpi(firstTitle,'all')
        data.Err.MSEAll = MSE;
        data.Err.STDAll = ErrorStd;
        data.Err.NRMSEAll     = NRMSE;
        data.Err.rankCorreAll = rankCorre;
    end
end
% find rank correlation between network output and real data
function [r]=RankCorre(x,y)
x=x';
y=y';
% Find the data length
N = length(x);
% Get the ranks of x
R = crank(x)';
for i=1:size(y,2)
    % Get the ranks of y
    S = crank(y(:,i))';
    % Calculate the correlation coefficient
    r(i) = 1-6*sum((R-S).^2)/N/(N^2-1); %#ok
end
end
function r=crank(x)
u = unique(x);
[~,z1] = sort(x);
[~,z2] = sort(z1);
r = (1:length(x))';
r=r(z2);
for i=1:length(u)
    s=find(u(i)==x);
    r(s,1) = mean(r(s));
end
end
% plot the regression line of output and real value
function data = plotReg(data,Title,Targets,Outputs)

 if strcmpi(Title,'tr')
     DispName = 'RegressionGraphEvaluation_TrainData';
elseif strcmpi(Title,'ts')
    DispName = 'RegressionGraphEvaluation_TestData';
elseif strcmpi(Title,'all')
    DispName = 'RegressionGraphEvaluation_ALLData';
end
figure('Name',DispName,'NumberTitle','off');
x = Targets';
y = Outputs';
format long
b1 = x\y
yCalc1 = b1*x;
scatter(x,y,'MarkerEdgeColor',[0 0.4470 0.7410],'LineWidth',.7);
hold('on');
plot(x,yCalc1,'Color',[0.8500 0.3250 0.0980]);
xlabel('Prediction');
ylabel('Target');
grid minor
% xgrid = 'on';
% disp.YGrid = 'on';
X = [ones(length(x),1) x];
b = X\y;
yCalc2 = X*b;
plot(x,yCalc2,'-.','MarkerSize',4,"LineWidth",.1,'Color',[0.9290 0.6940 0.1250])
legend('Data','Fit','Y=T','Location','best');
%
Rsq2 = 1 -  sum((y - yCalc1).^2)/sum((y - mean(y)).^2);

if strcmpi(Title,'tr')
    data.Err.RSqur_Tr = Rsq2;
    title(['Train Data, R^2 = ' num2str(Rsq2)]);
elseif strcmpi(Title,'ts')
    data.Err.RSqur_Ts = Rsq2;
    title(['Test Data, R^2 = ' num2str(Rsq2)]);
elseif strcmpi(Title,'all')
    data.Err.RSqur_All = Rsq2;
    title(['All Data, R^2 = ' num2str(Rsq2)]);
end

end


% --------------- Prediction ---------------
% ---------------------------------------------
function [opt,data] = PredictionData(opt,data)

delays  = opt.Delays;
PD = [];
if strcmpi(opt.learningMethod,'LSTM')
    for i=1:opt.PredictionHorizone
        Data = TSDataPrepreation(opt,data);
        predictedSample = predict(data.BiLSTM.Net,Data.x,'MiniBatchSize',opt.miniBatchSize);
        PD = predictedSample(end-min(delays)+1:end)';
        AllData = [PD];
    end 
elseif strcmpi(opt.learningMethod,'CNN')
    for i=1:opt.PredictionHorizone
        Data = TSDataPrepreation(opt,data);
        predictedSample = predict(data.CNN.Net,Data.x,'MiniBatchSize',opt.miniBatchSize);
        PD = predictedSample(end-min(delays)+1:end)';
        AllData = [predictedSample];
    end 
elseif strcmpi(opt.learningMethod,'MLP')
    for i=1:opt.PredictionHorizone
        Data = TSDataPrepreation(opt,data);
        predictedSample = data.FF.Net(Data.x);
        PD= predictedSample(end-min(delays)+1:end);
        AllData = [predictedSample];
    end 
end
AllData = deNorm(data.seriesdata,AllData',opt.dataPreprocessMode);

% check if the input data is rounded number so make the prediction rounded too
if sum(data.seriesdata-round(data.seriesdata))==0
    AllData = round(AllData);
end

PlotTS(opt,data,AllData,delays);
data.Predicted = AllData;
end
function PlotTS(opt,data,AllData,delays)

 figure('Name',['PredictionOutput on horizon: ' num2str(opt.PredictionHorizone)],'NumberTitle','off');
 plot(1:length(data.x'),...
               AllData(1:length(data.x')),'blue',...
               length(AllData)-(opt.PredictionHorizone)*min(delays):length(AllData)-(opt.PredictionHorizone)*min(delays)+1,...  
               AllData(length(AllData)-(opt.PredictionHorizone)*min(delays):length(AllData)-(opt.PredictionHorizone)*min(delays)+1),'-r',...
               length(AllData)-(opt.PredictionHorizone)*min(delays)+1:length(AllData),...  
               AllData(length(AllData)-(opt.PredictionHorizone)*min(delays)+1:length(AllData)),'-or','MarkerSize',3,"LineWidth",1.1);
grid minor;
grid on;
hold  on;
title('predicted data');
legend('Real Data','Predicted Data','Location','best','FontSize',8,'TextColor','black');  
end
 
function Data = TSDataPrepreation(opt,data)
    LRmethod      = opt.learningMethod; 
    % make some delays on input filed
    delay         = opt.Delays;
    
     X=data.M([2:(opt.i+1)],:) ;
    Y =[];
    
    

if strcmpi(LRmethod,'LSTM')
    for i=1:size(X,2)
        Data.x{i,1} = X(:,i); 
        Data.y{i,1} = Y(:,i);  
    end
elseif strcmpi(LRmethod,'CNN')
    Data.x = reshape(X, [size(X,1),1,1,size(X,2)]);
    for i=1:size(Y,2)
        Data.y{i,1} = Y(:,i);  
    end
elseif strcmpi(LRmethod,'MLP')
    Data.x  = X ; 
    Data.y  = Y ;  
end

end




% ----------- Save Output Prediction ----------
% ---------------------------------------------
function SavePredictedData(opt,data)
if ~opt.isSavePredictedData
    return;
end
[file,path] = uiputfile({'*.xlsx';'*.xlsx';'*.txt';'*.*'},'Please Select Output file',['Predicted_' data.DataFileName]);
if isequal(path,0) 
    disp('User clicked Cancel.')
     return;
end
Data      = data.Predicted';
RowNames  = (data.seriesdataHeder);
t = table(Data,'VariableNames',RowNames);


try
   writetable(t,[path file]);  
catch ME
    msg = [{ME.message};{''};{'MATLAB Error type: '};{ME.identifier}];
    disp(msg)
    return;
end
% open the save  file
pause(.5);
winopen([path file]);
end

