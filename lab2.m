addpath("Documents/kth/dl/labs/lab2/");
train_path = "data_batch_1.mat";
valid_path = "data_batch_2.mat";
test_path = "test_batch.mat";

%A = load(train_path);
%I = reshape(A.data', 32, 32, 3, 10000);
%I = permute(I, [2, 1, 3, 4]);
%montage(I(:, :, :, 1:500), 'Size', [5,5]);

%% Load data
[X_train, Y_train, y_train] = LoadBatch(train_path);
[X_valid, Y_valid, y_valid] = LoadBatch(valid_path);
[X_test, Y_test, y_test] = LoadBatch(test_path);

%% Preprocess data
rng(400);
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2);

X_train = Preprocess(X_train, mean_train, std_train);
X_valid = Preprocess(X_valid, mean_train, std_train);
X_test = Preprocess(X_test, mean_train, std_train);

[~, n] = size(X_train);

%% Initialize parameters of the network
m = 50;
theta = InitializeParam(X_train, Y_train, m);
W = theta(1:2);
b = theta(3:4);
d_batch = 10;
end_batch = 3;
h = 1e-5;
lambda = 0;
eps = 1e-6;
n_batch = 100;
eta = 0.001;
n_epochs = 200;

%% Evaluate the network function
% [H, P] = EvaluateClassifier(X_train, theta);

%% Compute the cost function
% [J, loss] = ComputeCost(X_train, Y_train, W, b, lambda);

%% Compute the gradients
% X_batch = X_train(1:d_batch, 1:end_batch);
% Y_batch = Y_train(:, 1:end_batch);
% theta_batch = InitializeParam(X_batch, Y_batch, m);
% W_batch = theta_batch(1:2);
% b_batch = theta_batch(3:4);

%% Gradients comparisons
% [H_batch, P_batch] = EvaluateClassifier(X_batch, theta_batch);
% [grad_W_an, grad_b_an] = ComputeGradients(X_batch, Y_batch, H_batch, P_batch, theta_batch, lambda);
% 
% [grad_b_num_fast, grad_W_num_fast] = ComputeGradsNum(X_batch, Y_batch, W_batch, b_batch, lambda, h);
% [grad_b_num_slow, grad_W_num_slow] = ComputeGradsNumSlow(X_batch, Y_batch, W_batch, b_batch, lambda, h);
% 
% [grad_W_err_fast, grad_b_err_fast] = ComputeRelativeError(grad_W_an, grad_b_an, grad_W_num_fast, grad_b_num_fast, eps);
% [grad_W_err_slow, grad_b_err_slow] = ComputeRelativeError(grad_W_an, grad_b_an, grad_W_num_slow, grad_b_num_slow, eps);
% [grad_W_err_given, grad_b_err_given] = ComputeRelativeError(grad_W_num_slow, grad_b_num_slow, grad_W_num_fast, grad_b_num_fast, eps);

% max(grad_W_err_fast{1}, [], 'all')
% max(grad_W_err_fast{2}, [], 'all')
% max(grad_b_err_fast{1}, [], 'all')
% max(grad_b_err_fast{2}, [], 'all')

% max(grad_W_err_slow{1}, [], 'all')
% max(grad_W_err_slow{2}, [], 'all')
% max(grad_b_err_slow{1}, [], 'all')
% max(grad_b_err_slow{2}, [], 'all')

% max(grad_W_err_given{1}, [], 'all')
% max(grad_W_err_given{2}, [], 'all')
% max(grad_b_err_given{1}, [], 'all')
% max(grad_b_err_given{2}, [], 'all')

%% Exercise 2
% GDparams = {n_batch, eta, n_epochs};
% [Wstar, bstar, J_train_array, loss_train_array, ...
%     J_valid_array, loss_valid_array, acc_train, acc_valid] = ...
%     MiniBatchGD(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, theta, lambda);

% figure;
% plot(1:n_epochs, J_train_array, 1:n_epochs, J_valid_array, '--');
% title('Cost J over epochs');
% xlabel('epochs');
% ylabel('cost J');
% legend('Training','Validation');
% figure;
% plot(1:n_epochs, loss_train_array, 1:n_epochs, loss_valid_array, '--');
% title('Loss over epochs');
% xlabel('epochs');
% ylabel('loss');
% legend('Training','Validation');
% figure;
% plot(1:n_epochs, acc_train, 1:n_epochs, acc_valid, '--');
% title('Accuracy over epochs');
% xlabel('epochs');
% ylabel('accuracy');
% legend('Training','Validation');

Exercise 3 - One Cycle
% lambda = 0.01;
% eta_min = 1e-5;
% eta_max = 1e-1;
% n_s = 500;
% nb_cycles = 1;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};
% 
% [Wstar, bstar, J_train_array, loss_train_array, ...
%    J_valid_array, loss_valid_array, acc_train, acc_valid, etas] = ...
%    MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, theta, lambda, etaparams);

% len_array = size(J_train_array, 2);
% figure;
% update_steps = (1:len_array)*10;
% plot(update_steps, J_train_array, update_steps, J_valid_array, '--');nb_updates = size(J_train_array, 2);
% yl = ylim;
% ylim([0, yl(2)]);
% title('Cost J over updates');
% xlabel('update steps');
% ylabel('cost J');
% legend('Training','Validation');
% figure;
% plot(update_steps, loss_train_array, update_steps, loss_valid_array, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Loss over updates');
% xlabel('update steps');
% ylabel('loss');
% legend('Training','Validation');
% figure;
% plot(update_steps, acc_train, update_steps, acc_valid, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Accuracy over updates');
% xlabel('update steps');
% ylabel('accuracy');
% legend('Training','Validation');

% nb_updates = size(etas, 2);
% figure;
% plot(1:nb_updates, etas);
% title('Eta values over updates');
% xlabel('update steps');
% ylabel('eta');

% ComputeAccuracy(X_test, y_test, {Wstar{1}, Wstar{2}, bstar{1}, bstar{2}})

% wanted_nbpoints_per_cycle = 10;
% actual_nbpoints_per_cycle = n/n_batch;
% nb_points_total = nb_updates;
% subJ_train = SubSample(J_train_array, wanted_nbpoints_per_cycle, actual_nbpoints_per_cycle, nb_points_total);
% subJ_valid = SubSample(J_valid_array, wanted_nbpoints_per_cycle, actual_nbpoints_per_cycle, nb_points_total);
% 
% figure;
% nb_points_sub = size(subJ_train, 2);
% plot(1:nb_points_sub, subJ_train, 1:nb_points_sub, subJ_valid, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Subsampled Cost J over updates');
% xlabel('update steps');
% ylabel('cost J');
% legend('Training','Validation');
% subloss_train = SubSample(loss_train_array, wanted_nbpoints_per_cycle, actual_nbpoints_per_cycle, nb_points_total);
% subloss_valid = SubSample(loss_valid_array, wanted_nbpoints_per_cycle, actual_nbpoints_per_cycle, nb_points_total);
% 
% figure;
% nb_points_sub = size(subloss_train, 2);
% plot(1:nb_points_sub, subloss_train, 1:nb_points_sub, subloss_valid, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Subsampled loss over updates');
% xlabel('update steps');
% ylabel('loss');
% legend('Training','Validation');
% subacc_train = SubSample(acc_train, wanted_nbpoints_per_cycle, actual_nbpoints_per_cycle, nb_points_total);
% subacc_valid = SubSample(acc_valid, wanted_nbpoints_per_cycle, actual_nbpoints_per_cycle, nb_points_total);
% 
% figure;
% nb_points_sub = size(subacc_train, 2);
% plot(1:nb_points_sub, subacc_train, 1:nb_points_sub, subacc_valid, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Subsampled loss over updates');
% xlabel('update steps');
% ylabel('loss');
% legend('Training','Validation');

%% Exercise 4 - Several Cycles
% lambda = 0.01;
% eta_min = 1e-5;
% eta_max = 1e-1;
% n_s = 800;
% nb_cycles = 3;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};
% 
% [Wstar, bstar, J_train_array, loss_train_array, ...
%    J_valid_array, loss_valid_array, acc_train, acc_valid, etas] = ...
%    MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, theta, lambda, etaparams);

% len_array = size(J_train_array, 2);
% figure;
% update_steps = (1:len_array)*10;
% plot(update_steps, J_train_array, update_steps, J_valid_array, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Cost J over updates');
% xlabel('update steps');
% ylabel('cost J');
% legend('Training','Validation');
% figure;
% plot(update_steps, loss_train_array, update_steps, loss_valid_array, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Loss over updates');
% xlabel('update steps');
% ylabel('loss');
% legend('Training','Validation');
% figure;
% plot(update_steps, acc_train, update_steps, acc_valid, '--');
% yl = ylim;
% ylim([0, yl(2)]);
% title('Accuracy over updates');
% xlabel('update steps');
% ylabel('accuracy');
% legend('Training','Validation');

% figure;
% nb_updates = size(etas, 2);
% plot(1:nb_updates, etas);
% title('Eta values over updates');
% xlabel('update steps');
% ylabel('eta');

% ComputeAccuracy(X_test, y_test, {Wstar{1}, Wstar{2}, bstar{1}, bstar{2}})

%% Exercise 4 : Tuning lambda
%% Data loading,splitting and preprocessing
path_batch1 = "data_batch_1.mat";
path_batch2 = "data_batch_2.mat";
path_batch3 = "data_batch_3.mat";
path_batch4 = "data_batch_4.mat";
path_batch5 = "data_batch_5.mat";

[X1, Y1, y1] = LoadBatch(path_batch1);
[X2, Y2, y2] = LoadBatch(path_batch2);
[X3, Y3, y3] = LoadBatch(path_batch3);
[X4, Y4, y4] = LoadBatch(path_batch4);
[X5, Y5, y5] = LoadBatch(path_batch5);

X = [X1 X2 X3 X4 X5];
Y = [Y1 Y2 Y3 Y4 Y5];
y = [y1; y2; y3; y4; y5];

validset_size = 5000;
[X_train, Y_train, y_train, X_valid, Y_valid, y_valid] = SplitData(X, Y, y, validset_size);


rng(400);
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2);

X_train = Preprocess(X_train, mean_train, std_train);
X_valid = Preprocess(X_valid, mean_train, std_train);

%% First search of lambda
[~, n] = size(X_train);
m = 50;
theta = InitializeParam(X_train, Y_train, m);
W = theta(1:2);
b = theta(3:4);
n_batch = 100;

nb_cycles = 2;
n_s = 2 * floor(n/n_batch);
eta_min = 1e-5;
eta_max = 1e-1;
etaparams = {nb_cycles, n_s, eta_min, eta_max};

nb_lambdas = 8;
lambdas = linspace(1e-5, 1e-1, nb_lambdas);

% save_path1 = "~/Documents/kth/dl/labs/lab2/Saved_Files/lambdas_results1.txt";
% 
% writecell({'lambda', 'acc_valid'}, save_path1, 'Delimiter', 'tab');
% 
% for lbda_idx=1:nb_lambdas
% 
%     lambda = lambdas(lbda_idx);
% 
%     [Wstar, bstar, J_train_array, loss_train_array, ...
%        J_valid_array, loss_valid_array, acc_train, acc_valid, etas] = ...
%        MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, theta, lambda, etaparams);
% 
%     writematrix([lambda acc_valid(end)],save_path1,'WriteMode','append', 'Delimiter', 'tab');
% 
% end

% lambdas_results1 = importdata(save_path1);
% lambda_accuracies = lambdas_results1.data;

% %% Second search of lambda
% theta = InitializeParam(X_train, Y_train, m);
% 
% nb_cycles = 2;
% n_s = 2 * floor(n/n_batch);
% eta_min = 1e-5;
% eta_max = 1e-1;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};
% 
% nb_lambdas = 4;
% lambdas = linspace(1e-7, 1e-5, nb_lambdas);
% 
% save_path2 = "~/Documents/kth/dl/labs/lab2/Saved_Files/lambdas_results2.txt";
% 
% writecell({'lambda', 'acc_valid'}, save_path2, 'Delimiter', 'tab');
% 
% for lbda_idx=1:nb_lambdas
% 
%     lambda = lambdas(lbda_idx);
% 
%     [Wstar, bstar, J_train_array, loss_train_array, ...
%        J_valid_array, loss_valid_array, acc_train, acc_valid, etas] = ...
%        MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, theta, lambda, etaparams);
% 
%     writematrix([lambda acc_valid(end)],save_path2,'WriteMode','append', 'Delimiter', 'tab');
% 
% end

% lambdas_results2 = importdata(save_path2);
% lambda_accuracies = lambdas_results2.data;

% %% Third search of lambda
% theta = InitializeParam(X_train, Y_train, m);
% 
% nb_cycles = 2;
% n_s = 2 * floor(n/n_batch);
% eta_min = 1e-5;
% eta_max = 1e-1;
% etaparams = {nb_cycles, n_s, eta_min, eta_max};
% 
% path2 = "~/Documents/kth/dl/labs/lab2/Saved_Files/lambdas_results2.txt";
% lbdas2 = importdata(path2);
% data2 = lbdas2.data;    
% [~, argmax] = max(data2(:, 2));
% lbda_min = data2(argmax - 1, 1);
% lbda_max = data2(argmax + 1, 1);
% 
% nb_lambdas = 8;
% lambdas = linspace(lbda_min, lbda_max, nb_lambdas);
% 
% save_path3 = "~/Documents/kth/dl/labs/lab2/Saved_Files/lambdas_results3.txt";
% 
% writecell({'lambda', 'acc_valid'}, save_path3, 'Delimiter', 'tab');
% 
% for lbda_idx=1:nb_lambdas
% 
%     lambda = lambdas(lbda_idx);
% 
%     [Wstar, bstar, J_train_array, loss_train_array, ...
%        J_valid_array, loss_valid_array, acc_train, acc_valid, etas] = ...
%        MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, theta, lambda, etaparams);
% 
%     writematrix([lambda acc_valid(end)],save_path3,'WriteMode','append', 'Delimiter', 'tab');
% 
% end

% lambdas_results3 = importdata(save_path3);
% lambda_accuracies = lambdas_results3.data;

%% Best found lambda settings - 3 cycles

lbdas3 = importdata(save_path3);
data2 = lbdas3.data;    
[~, argmax] = max(data2(:, 2));
lambda_finalist = data2(argmax, 1);

validset_size = 1000;
[X_train, Y_train, y_train, X_valid, Y_valid, y_valid] = SplitData(X, Y, y, validset_size);


rng(400);
mean_train = mean(X_train, 2);
std_train = std(X_train, 0, 2);

X_train = Preprocess(X_train, mean_train, std_train);
X_valid = Preprocess(X_valid, mean_train, std_train);
X_test = Preprocess(X_test, mean_train, std_train);

%% Initialization of the parameters

[~, n] = size(X_train);
m = 50;
theta = InitializeParam(X_train, Y_train, m);
W = theta(1:2);
b = theta(3:4);
n_batch = 100;

nb_cycles = 3;
n_s = 2 * floor(n/n_batch);
eta_min = 1e-5;
eta_max = 1e-1;
etaparams = {nb_cycles, n_s, eta_min, eta_max};

[Wstar, bstar, J_train_array, loss_train_array, ...
       J_valid_array, loss_valid_array, acc_train, acc_valid, etas] = ...
       MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, theta, lambda, etaparams);

len_array = size(J_train_array, 2);
figure;
update_steps = (1:len_array)*50;
plot(update_steps, J_train_array, update_steps, J_valid_array, '--');
nb_updates = size(J_train_array, 2);
yl = ylim;
ylim([0, yl(2)]);
title('Cost J over updates');
xlabel('update steps');
ylabel('cost J');
legend('Training','Validation');
figure;
plot(update_steps, loss_train_array, update_steps, loss_valid_array, '--');
yl = ylim;
ylim([0, yl(2)]);
title('Loss over updates');
xlabel('update steps');
ylabel('loss');
legend('Training','Validation');
figure;
plot(update_steps, acc_train, update_steps, acc_valid, '--');
yl = ylim;
ylim([0, yl(2)]);
title('Accuracy over updates');
xlabel('update steps');
ylabel('accuracy');
legend('Training','Validation');

nb_updates = size(etas, 2);
figure;
plot(1:nb_updates, etas);
title('Eta values over updates');
xlabel('update steps');
ylabel('eta');

test_path = "test_batch.mat";
[X_test, Y_test, y_test] = LoadBatch(test_path);
X_test = Preprocess(X_test, mean_train, std_train);

ComputeAccuracy(X_test, y_test, {Wstar{1}, Wstar{2}, bstar{1}, bstar{2}})

function [X, Y, y] = LoadBatch(filename)
    dict = load(filename);
    X = double(dict.data');
    y = dict.labels + 1;
    Y = (y == 1:10)';
end


function preprocessed_X = Preprocess(X, mean, std)
    X = X - repmat(mean, [1, size(X, 2)]);
    preprocessed_X = X ./ repmat(std, [1, size(X, 2)]);
end


function theta = InitializeParam(X_train, Y_train, m)
    [d, ~] = size(X_train);
    [K, ~] = size(Y_train);
    
    W1 = randn(m, d)/sqrt(d);
    W2 = randn(K, m)/sqrt(m);
    
    b1 = zeros(m, 1);
    b2 = zeros(K, 1);
    
    theta = {W1, W2, b1, b2};
end


function P = softmax(s)
    K = size(s, 1);
    P = exp(s)./repmat(ones(1, K)*exp(s), [K, 1]);
end


function [H, P] = EvaluateClassifier(X, theta)
    W = theta(1:2);
    b = theta(3:4);
    
    s1 = W{1} * X + repmat(b{1}, [1, size(X, 2)]);
    H = max(0, s1);
    s = W{2} * H + repmat(b{2}, [1, size(X, 2)]);
    P = softmax(s);
end


function [J, loss] = ComputeCost(X, Y, W, b, lambda)
    theta = {W{1}, W{2}, b{1}, b{2}};
    [~, P] = EvaluateClassifier(X, theta);
    n = size(Y, 2);
    lcross = zeros(1, n);
    for i = 1:n
        lcross(i) = Y(:, i)' * log(P(:, i));
    end
    loss = - sum(lcross)/n;
    J = loss + lambda * (sum(W{1} .* W{1}, 'all') + sum(W{2} .* W{2}, 'all'));
end


function [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, H_batch, P_batch, theta, lambda)
    
    n_batch = size(X_batch, 2);
    
    G_batch = - (Y_batch - P_batch);
    
    grad_W2 = (G_batch * H_batch')/n_batch + 2 * lambda * theta{2};
    grad_b2 = (G_batch * ones(n_batch, 1))/n_batch;
    
    %Propagate the grad back through the second layer
    G_batch = theta{2}' * G_batch;
    G_batch( H_batch <= 0 ) = 0;
    
    grad_W1 = (G_batch * X_batch')/n_batch + 2 * lambda * theta{1};
    grad_b1 = (G_batch * ones(n_batch, 1))/n_batch;
    
    grad_W = {grad_W1, grad_W2};
    grad_b = {grad_b1, grad_b2};
end


function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);
    
    [c, ~] = ComputeCost(X, Y, W, b, lambda);
    
    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));
        
        for i=1:length(b{j})
            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            [c2, ~] = ComputeCost(X, Y, W, b_try, lambda);
            grad_b{j}(i) = (c2-c) / h;
        end
    end
    
    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));
        
        for i=1:numel(W{j})   
            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            [c2, ~] = ComputeCost(X, Y, W_try, b, lambda);
            
            grad_W{j}(i) = (c2-c) / h;
        end
    end
end


function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);
    
    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));
        
        for i=1:length(b{j})
            
            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b_try, lambda);
            
            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);
            
            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));
        
        for i=1:numel(W{j})
            
            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W_try, b, lambda);
        
            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W_try, b, lambda);
        
            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end


function [grad_W_err, grad_b_err] = ComputeRelativeError(grad_W_an, grad_b_an, grad_W_num, grad_b_num, eps)
    %% Instructions formula
    %grad_W1_err = abs(grad_W_an{1} - grad_W_num{1})./max(eps, abs(grad_W_an{1}) + grad_W_num{1});
    %grad_W2_err = abs(grad_W_an{2} - grad_W_num{2})./max(eps, abs(grad_W_an{2}) + grad_W_num{2});
    %grad_b1_err = abs(grad_b_an{1} - grad_b_num{1})./max(eps, abs(grad_b_an{1}) + grad_b_num{1});
    %grad_b2_err = abs(grad_b_an{2} - grad_b_num{2})./max(eps, abs(grad_b_an{2}) + grad_b_num{2});
    
    %% Wikipedia formula
    %grad_W1_err = abs(grad_W_an{1} - grad_W_num{1})./abs(grad_W_num{1});
    %grad_W2_err = abs(grad_W_an{2} - grad_W_num{2})./abs(grad_W_num{2});
    %grad_b1_err = abs(grad_b_an{1} - grad_b_num{1})./abs(grad_b_num{1});
    %grad_b2_err = abs(grad_b_an{2} - grad_b_num{2})./abs(grad_b_num{2});
    
    %% Standford's course formula
    grad_W1_err = abs(grad_W_an{1} - grad_W_num{1})./max(abs(grad_W_num{1}), abs(grad_W_an{1}));
    grad_W2_err = abs(grad_W_an{2} - grad_W_num{2})./max(abs(grad_W_num{2}), abs(grad_W_an{2}));
    grad_b1_err = abs(grad_b_an{1} - grad_b_num{1})./max(abs(grad_b_num{1}), abs(grad_b_an{1}));
    grad_b2_err = abs(grad_b_an{2} - grad_b_num{2})./max(abs(grad_b_num{2}), abs(grad_b_an{2}));
    
    grad_W_err = {grad_W1_err, grad_W2_err};
    grad_b_err = {grad_b1_err, grad_b2_err};
end


% Compute argmax of each column
function argmax = Argmax(matrix)
    [~, argmax] = max(matrix);
end


function accuracy = ComputeAccuracy(X, y, theta)
    n = size(y, 1);

    acc = zeros(1, n);

    [~, P] = EvaluateClassifier(X, theta);
    prediction = Argmax(P);

    for i = 1:n
        if prediction(i) == y(i)
            acc(i) = 1;
        else
            acc(i) = 0;
        end
    end

    accuracy = sum(acc)/n;
end


function permutation = Shuffle(vector)
    permutation = vector(randperm(length(vector)));
end


function [Wstar, bstar, J_train_array, loss_train_array, ...
    J_valid_array, loss_valid_array, acc_train, acc_valid] = ...
    MiniBatchGD(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, theta, lambda)

    n = size(X_train, 2);
    
    n_batch = GDparams{1};
    eta = GDparams{2};

    n_epochs = GDparams{3};
    
        J_train_array = zeros(1, n_epochs);
        loss_train_array = zeros(1, n_epochs);
        J_valid_array = zeros(1, n_epochs);
        loss_valid_array = zeros(1, n_epochs);
        acc_train = zeros(1, n_epochs);
        acc_valid = zeros(1, n_epochs);
    
        for epoch = 1:n_epochs
            for j=1:n/n_batch
                
                idx_permutation = Shuffle(1:n);
        
                j_start = (j-1)*n_batch + 1;
                j_end = j*n_batch;
                inds = j_start:j_end;
        
                X_batch = X_train(:, idx_permutation(inds));
                Y_batch = Y_train(:, idx_permutation(inds));
        
                [H_batch, P_batch] = EvaluateClassifier(X_batch, theta);
        
                [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, H_batch, P_batch, theta, lambda);
        
                theta{1} = theta{1} - eta * grad_W{1};
                theta{2} = theta{2} - eta * grad_W{2};
                theta{3} = theta{3} - eta * grad_b{1};
                theta{4} = theta{4} - eta * grad_b{2};
        
            end
        
            [J_train, loss_train] = ComputeCost(X_train, Y_train, theta(1:2), theta(3:4), lambda);
            [J_valid, loss_valid] = ComputeCost(X_valid, Y_valid, theta(1:2), theta(3:4), lambda);
        
            J_train_array(epoch) = J_train;
            loss_train_array(epoch) = loss_train;
            J_valid_array(epoch) = J_valid;
            loss_valid_array(epoch) = loss_valid;
    
            acc_train(epoch) = ComputeAccuracy(X_train, y_train, theta);
            acc_valid(epoch) = ComputeAccuracy(X_valid, y_valid, theta);
        
        end

    Wstar = theta(1:2);
    bstar = theta(3:4);

end


function [Wstar, bstar, J_train_array, loss_train_array, ...
    J_valid_array, loss_valid_array, acc_train, acc_valid, etas] = ...
    MiniBatchGDCyclical(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, n_batch, theta, lambda, etaparams)

    n = size(X_train, 2);
    
    nb_cycles = etaparams{1};
    n_s = etaparams{2};
    eta_min = etaparams{3};
    eta_max = etaparams{4};

    J_train_array = [];
    loss_train_array = [];
    J_valid_array = [];
    loss_valid_array = [];
    acc_train = [];
    acc_valid = [];

    
    t = 0;
    epoch = 1;
    l = 0;

    etas = [];
        
    while l < nb_cycles
    
        for j=1:n/n_batch % = 1000 = 1 cycle

            
            idx_permutation = Shuffle(1:n);
    
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            inds = j_start:j_end;
    
            X_batch = X_train(:, idx_permutation(inds));
            Y_batch = Y_train(:, idx_permutation(inds));
    
            [H_batch, P_batch] = EvaluateClassifier(X_batch, theta);
    
            [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, H_batch, P_batch, theta, lambda);
            
            if mod(floor(t/n_s), 2) == 0 %case of equation 14 : positive slope
                
                eta = eta_min + (t - 2*l*n_s)/n_s*(eta_max - eta_min);

            else %case of equation 15 : negative slope

                eta = eta_max - (t - (2*l+1)*n_s)/n_s*(eta_max - eta_min);

            end

            theta{1} = theta{1} - eta * grad_W{1};
            theta{2} = theta{2} - eta * grad_W{2};
            theta{3} = theta{3} - eta * grad_b{1};
            theta{4} = theta{4} - eta * grad_b{2};

            etas(end+1) = eta;
            t = t + 1;
            l = floor(t/(2*n_s));


            [J_train, loss_train] = ComputeCost(X_train, Y_train, theta(1:2), theta(3:4), lambda);
            [J_valid, loss_valid] = ComputeCost(X_valid, Y_valid, theta(1:2), theta(3:4), lambda);
            
            interval_size = n/n_batch/10; % 10 = wanted_nb_points_per_cycle

            if mod(t, interval_size) == 0

                J_train_array(end+1) = J_train;
                loss_train_array(end+1) = loss_train;
                J_valid_array(end+1) = J_valid;
                loss_valid_array(end+1) = loss_valid;
        
                acc_train(end+1) = ComputeAccuracy(X_train, y_train, theta);
                acc_valid(end+1) = ComputeAccuracy(X_valid, y_valid, theta);

            end

            if mod(t, 100) == 0    
                t
            end
    
        end
    
%         [J_train, loss_train] = ComputeCost(X_train, Y_train, theta(1:2), theta(3:4), lambda);
%         [J_valid, loss_valid] = ComputeCost(X_valid, Y_valid, theta(1:2), theta(3:4), lambda);
%     
%         J_train_array(epoch) = J_train;
%         loss_train_array(epoch) = loss_train;
%         J_valid_array(epoch) = J_valid;
%         loss_valid_array(epoch) = loss_valid;
% 
%         acc_train(epoch) = ComputeAccuracy(X_train, y_train, theta);
%         acc_valid(epoch) = ComputeAccuracy(X_valid, y_valid, theta);

        epoch = epoch + 1;
                    
    end
        
    Wstar = theta(1:2);
    bstar = theta(3:4);

end


function subsampled = SubSample(array, wanted_nbpoints_per_cycle, ...
    actual_nbpoints_per_cycle, nb_points_tot)
    
    interval_size = actual_nbpoints_per_cycle/wanted_nbpoints_per_cycle;
    sub_idx = interval_size * (1:(nb_points_tot/interval_size));
    subsampled = array(sub2ind(size(array), ones(1, size(sub_idx, 2)), sub_idx));

end


function [X_train, Y_train, y_train, X_valid, Y_valid, y_valid] = ...
    SplitData(X, Y, y, validset_size)

    X_train = X(:, 1:end-validset_size);
    Y_train = Y(:, 1:end-validset_size);
    y_train = y(1:end-validset_size);
    
    X_valid = X(:, end-validset_size+1:end);
    Y_valid = Y(:, end-validset_size+1:end);
    y_valid = y(end-validset_size+1:end);

end




