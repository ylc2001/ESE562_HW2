%% Data-driven DCPF dataset generation
clear; clc; close all;

% Load the B matrix
load('B_pf.mat'); % B_pf must be inside this file

% Base values
Pgen_base = [1.63; 0.85; 0; 0; 0; 0; 0; 0];
Pload_base = [0; 0; 0; 0.9; 0; 1; 0; 1.25];

%% Function: generate dataset
function [Pgen_all, Pload_all, theta_all] = generate_dataset(N, Pgen_base, Pload_base, deviation, B_pf)
    Pgen_all = zeros(N, length(Pgen_base));
    Pload_all = zeros(N, length(Pload_base));
    theta_all = zeros(N, length(Pgen_base)); % same size as number of buses
    
    for i = 1:N
        % random deviations within ±deviation
        Pgen_var = Pgen_base + (rand(size(Pgen_base)) * 2 - 1) .* (deviation * Pgen_base);
        Pload_var = Pload_base + (rand(size(Pload_base)) * 2 - 1) .* (deviation * Pload_base);

        % DC power flow
        theta = inv(B_pf) * (Pgen_var - Pload_var);

        % save
        Pgen_all(i,:) = Pgen_var';
        Pload_all(i,:) = Pload_var';
        theta_all(i,:) = theta';
    end
end

%% Generate datasets
% Training set 1: 1000 samples, ±20%
[Pgen_train1, Pload_train1, theta_train1] = generate_dataset(1000, Pgen_base, Pload_base, 0.2, B_pf);

% Training set 2: 200 samples, ±20%
[Pgen_train2, Pload_train2, theta_train2] = generate_dataset(200, Pgen_base, Pload_base, 0.2, B_pf);

% Testing set 1: 1000 samples, ±20%
[Pgen_test1, Pload_test1, theta_test1] = generate_dataset(1000, Pgen_base, Pload_base, 0.2, B_pf);

% Testing set 2: 1000 samples, ±40%
[Pgen_test2, Pload_test2, theta_test2] = generate_dataset(1000, Pgen_base, Pload_base, 0.4, B_pf);

%% Save as .mat files
save('train_set1.mat', 'Pgen_train1', 'Pload_train1', 'theta_train1');
save('train_set2.mat', 'Pgen_train2', 'Pload_train2', 'theta_train2');
save('test_set1.mat', 'Pgen_test1', 'Pload_test1', 'theta_test1');
save('test_set2.mat', 'Pgen_test2', 'Pload_test2', 'theta_test2');

%% Also export to CSV for Python
writematrix([Pgen_train1 Pload_train1 theta_train1], 'train_set1.csv');
writematrix([Pgen_train2 Pload_train2 theta_train2], 'train_set2.csv');
writematrix([Pgen_test1 Pload_test1 theta_test1], 'test_set1.csv');
writematrix([Pgen_test2 Pload_test2 theta_test2], 'test_set2.csv');

disp('All datasets generated and saved.');