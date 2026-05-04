%% Pearson Correlation Matrix Calculation and Visualization
% Read dataset
dataTable = readtable('C:\Users\dogan\Desktop\tek isimNLOS lazerses verifuzyonu\veriler_dataset\lazer_ses_birlikte\dataset_fused.csv');

% Select only numeric columns
numericVars = vartype("numeric");
Xmat = table2array(dataTable(:, numericVars));
predictorNames = dataTable.Properties.VariableNames(numericVars);

% Remove the last column (assumed to be class labels)
Xmat = Xmat(:,1:end-1);
predictorNames = predictorNames(1:end-1);

% Compute Pearson correlation matrix
R = corr(Xmat, 'Type', 'Pearson');

%% 1. Visualization with Heatmap
figure;
h = heatmap(predictorNames, predictorNames, R, ...
            'Colormap', parula, 'ColorLimits', [-1 1]);

% Show labels with underscores as normal characters
h.XDisplayLabels = predictorNames;
h.YDisplayLabels = predictorNames;

title('Pearson Correlation Matrix (Heatmap)');

%% 2. Visualization with Imagesc
figure;
imagesc(R);
colormap(parula);
colorbar;

% Set axis ticks and labels
xticks(1:length(predictorNames));
yticks(1:length(predictorNames));
xticklabels(predictorNames);
yticklabels(predictorNames);

% Prevent underscores from being interpreted as subscripts
set(gca, 'TickLabelInterpreter', 'none', 'XTickLabelRotation', 45);

title('Pearson Correlation Matrix (Imagesc)');