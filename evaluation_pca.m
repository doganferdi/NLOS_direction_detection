%% PCA Explained Variance Summary
% Read dataset
dataTable = readtable('C:\Users\dogan\Desktop\tek isimNLOS lazerses verifuzyonu\veriler_dataset\lazer_ses_birlikte\dataset_fused.csv');

% Select only numeric columns
numericVars = vartype("numeric");
Xmat = table2array(dataTable(:, numericVars));
predictorNames = dataTable.Properties.VariableNames(numericVars);

% Remove the last column (assumed to be class labels)
Xmat = Xmat(:,1:end-1);
predictorNames = predictorNames(1:end-1);

% Standardize data (important for PCA)
Xstd = zscore(Xmat);

% Perform PCA
[coeff, score, latent, tsquared, explained] = pca(Xstd);

%% Plot explained variance
figure;
bar(explained);
xlabel('Principal Component');
ylabel('Explained Variance (%)');
title('PCA Explained Variance');

%% Plot cumulative explained variance
figure;
plot(cumsum(explained), '-o');
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('Cumulative Explained Variance by PCA');
grid on;