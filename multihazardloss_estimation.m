clear
close all
clc

load mlthazard_data


%% option
response_region = 2; % 1: Iwanuma, 2: Onagawa, 3: joint

loss_type = 3; % 1:EQ, 2:Tsu, 3:EQ+Tsu

PGV_size = 5; % 0 (none), 1 (small), 3 (median), 5 (large)

S_net_size = 40; % 0,5,10,20,40

max_time = 5; % waiting time

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% prepare explanatory variables
%% prepare earthquake info
EQdata = [reshape(SCENARIO(:,1,:),[4000 1]) reshape(SCENARIO(:,2,:),[4000 1]) reshape(SCENARIO(:,3,:),[4000 1])];
%% prepare PGV
if PGV_size ~= 0
    selected_PGV_sensor_Iwanuma = PGV_sensor_Iwanuma(1:PGV_size*2);
    selected_PGV_sensor_Onagawa = PGV_sensor_Onagawa(1:PGV_size);
    PGVdata_Iwanuma = PGV_Iwanuma(selected_PGV_sensor_Iwanuma,:)';
    PGVdata_Onagawa = PGV_Onagawa(selected_PGV_sensor_Onagawa,:)';
    if response_region == 1
        PGVdata = [PGVdata_Iwanuma];
    elseif response_region == 2
        PGVdata = [PGVdata_Onagawa];
    else
        PGVdata = [PGVdata_Iwanuma PGVdata_Onagawa];
    end
else
    selected_PGV_sensor_Iwanuma = [];
    selected_PGV_sensor_Onagawa = [];
    PGVdata = [];
end
%% prepare wave amplitude
if S_net_size ~= 0
    selected_S_net = selected_station(1:S_net_size,1);
else
    selected_S_net = [];
end

WAVEdata = [];
if S_net_size ~=0
    for ii = 1:length(max_time)
        disp(["Waiting duration", num2str(max_time(ii)), ' minutes'])
        datatmp1 = [];
        for jj = 1:length(selected_S_net)
            datatmp1 = max(abs(WAVE_DATA(1:max_time(ii), selected_S_net(jj),:,:)));
            WAVEdata = [WAVEdata, reshape(datatmp1(:), [4000 1])];
            clear datatmp1
        end
    end
else
    WAVEdata = [];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% prepare resonse variable
%% joint Iwanuma and Onagawa loss
joint_EQ = Iwanuma_EQ + Onagawa_EQ;
joint_TSU = Iwanuma_TSU + Onagawa_TSU;
joint_MLT = Iwanuma_MLT + Onagawa_MLT;


if loss_type == 1
    if response_region == 1
        disp('The interested response variable is EQ loss only in Iwanuma')
        res1 = Iwanuma_EQ(:);
    elseif response_region == 2
        disp('The interested response variable is EQ loss only in Onagawa')
        res1 = Onagawa_EQ(:);
    else
        disp('The interested response variable is JOINT EQ loss')
        res1 = joint_EQ(:);
    end
elseif loss_type == 2
    if response_region == 1
        disp('The interested response variable is Tsu loss only in Iwanuma')
        res1 = Iwanuma_TSU(:);
    elseif response_region == 2
        disp('The interested response variable is Tsu loss only in Onagawa')
        res1 = Onagawa_TSU(:);
    else
        disp('The interested response variable is JOINT Tsu loss')
        res1 = joint_TSU(:);
    end
elseif loss_type == 3
    if response_region == 1
        disp('The interested response variable is EQ+Tsu loss only in Iwanuma')
        res1 = Iwanuma_MLT(:);
    elseif response_region == 2
        disp('The interested response variable is EQ+Tsu loss only in Onagawa')
        res1 = Onagawa_MLT(:);
    else
        disp('The interested response variable is JOINT EQ+Tsu loss')
        res1 = joint_MLT(:);
    end
end

res1 = max(res1, 0.0001);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% merge data
if loss_type == 1
    data_t1 = [PGVdata];
    data_t1e = [EQdata PGVdata];
elseif loss_type == 2
    data_t1 = [WAVEdata];
    data_t1e = [EQdata WAVEdata];
elseif loss_type == 3
    data_t1 = [PGVdata WAVEdata];
    data_t1e = [EQdata PGVdata WAVEdata];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fit RF
rf1e = fitensemble(data_t1e, log(res1),'Bag',500,'Tree','Type','regression');
rf1e_pred = predict(rf1e, data_t1e);
mse1e_rf = sum((log(res1) - rf1e_pred).^2)/length(rf1e_pred);
disp(['The MSE for RF model with EQ(mag,lat,long) is: ', num2str(mse1e_rf)])
figure(4)
plot(res1, exp(rf1e_pred), 'b.')
if loss_type == 1
    axis([0 800 0 800])
else
    axis([0 1800 0 1800])
end
axis square
%% CV
nFolds = 10;
foldSize = ceil(size(res1, 1) / nFolds); %400
indices = randperm(size(res1, 1));
mse_rf = zeros(nFolds, 1);
SST = zeros(nFolds, 1);
SSE = zeros(nFolds, 1);
R_squared = zeros(nFolds, 1);
% 10-fold cross-validation
for i = 1:nFolds
    fprintf('Processing fold %d...\n', i);
    
    % Get test and training indices
    testIdx = indices((i-1)*foldSize + 1 : min(i*foldSize, end));
    trainIdx = setdiff(indices, testIdx);
    
    % Split data into training and test sets
    X_train = data_t1e(trainIdx, :);
    y_train = log(res1(trainIdx, :));
    X_test = data_t1e(testIdx, :);
    y_test = log(res1(testIdx, :));
    
    % Fit model
    %model_mlr = fitlm(X_train, y_train);
    model_rf = fitensemble(X_train, y_train,'Bag',500,'Tree','Type','regression');
    
    % Predict test set
    %y_pred_mlr = predict(model_mlr, X_test);
    y_pred_rf = predict(model_rf, X_test);
    % Compute mean squared error
    %mse_mlr(i) = mean((y_test - y_pred_mlr).^2);
    mse_rf(i) = mean((y_test - y_pred_rf).^2);
    SST(i) = sum((y_test - mean(y_test)).^2);
    SSE(i) = sum((y_test - y_pred_rf).^2);
    R_squared(i) = 1 - SSE(i)/SST(i); 
end

% Calculate average mean squared error
avgMSE_rf = mean(mse_rf);
fprintf('RF average mean squared error: %.4f\n', avgMSE_rf);
avgR_squared = mean(R_squared);
fprintf('RF average R squared : %.4f\n', avgR_squared);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


