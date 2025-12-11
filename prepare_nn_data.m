clear all; close all; clc;

% Load dataset
load X.mat     % signals (N_samples × 2048)
load Y.mat     % labels  (N_samples × 1)

fprintf("Loaded %d samples...\n", length(Y));

% --------------------------------------
% 1. Normalize data (important for ML)
% --------------------------------------
X = X ./ max(abs(X), [], 2);

% --------------------------------------
% 2. Train/Test Split 80/20
% --------------------------------------
numSamples = size(X,1);
idx = randperm(numSamples);

train_size = floor(0.8 * numSamples);

train_idx = idx(1:train_size);
test_idx  = idx(train_size+1:end);

X_train = X(train_idx, :);
Y_train = Y(train_idx);

X_test  = X(test_idx, :);
Y_test  = Y(test_idx);

% --------------------------------------
% 3. Reshape for neural network
%    CNN expects 2048 × 1 × 1 format
% --------------------------------------

X_train = reshape(X_train.', 2048, 1, 1, []);
X_train = permute(X_train, [1 2 3 4]);   % (2048×1×1×N)

X_test = reshape(X_test.', 2048, 1, 1, []);
X_test = permute(X_test, [1 2 3 4]);

% --------------------------------------
% 4. Save NN-ready dataset
% --------------------------------------
save nn_dataset.mat X_train Y_train X_test Y_test

fprintf("Neural network dataset prepared & saved as nn_dataset.mat\n");

