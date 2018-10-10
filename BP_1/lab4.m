%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course:  Understanding Deep Neural Networks
% Teacher: Zhang Yi
% Student: ³ÂÏé
% ID:2016141482081
%
% Lab 4 - Handwritten Digit Recognition
%
% Task: recognize handwritten digits in images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear workspace and close plot windows
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code BELOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prepare the data set
load mnist_small_matlab.mat
train_size = 10000;
x_train{1} = reshape(trainData(1:14, 1:14, :), [], train_size);
x_train{2} = reshape(trainData(15:28, 1:14, :), [], train_size);
x_train{3} = reshape(trainData(15:28, 15:28, :), [], train_size);
x_train{4} = reshape(trainData(1:14, 15:28, :), [], train_size);
x_train{5} = zeros(0, train_size);
x_train{6} = zeros(0, train_size);
x_train{7} = zeros(0, train_size);
x_train{8} = zeros(0, train_size);

test_size = 2000;
x_test{1} = reshape(testData(1: 14, 1: 14, :), [], test_size);
x_test{2} = reshape(testData(15: 28, 1: 14, :), [], test_size);
x_test{3} = reshape(testData(15: 28, 15: 28, :), [], test_size);
x_test{4} = reshape(testData(1: 14, 15: 28, :), [], test_size);
x_test{5} = zeros (0, test_size);
x_test{6} = zeros (0, test_size);
x_test{7} = zeros (0, test_size);
x_test{8} = zeros (0, test_size);

% choose parameters
alpha = 1;
max_iter = 300;
mini_batch = 100;
J = [];
Acc = [];

% define network architecture
layer_size = [196 32
              196 32
              196 32
              196 32
                0 64
                0 64
                0 64
                0 10];
L = 8;



% initialize weights
for l = 1:L-1
    %w{l} = randn(layer_size(l+1,2), sum(layer_size(l,:)));
    w{l} = (rand(layer_size(l+1,2), sum(layer_size(l,:))) * 2 -1) * sqrt(6/(layer_size(l+1,2)+sum(layer_size(l,:))));
end



% train

for iter = 1:max_iter
   ind = randperm(train_size);
   for k = 1:ceil(train_size/mini_batch)
       a{1} = zeros(layer_size(1,2),mini_batch);
       for l=1:L
           x{l} = x_train{l}(:,ind((k-1)*mini_batch+1:min(k*mini_batch, train_size)));
       end
       y = double(trainLabels(:,ind((k-1)*mini_batch+1:min(k*mini_batch, train_size))));
       
       for l=1:L-1
           [a{l+1}, z{l+1}] =fc(w{l}, a{l}, x{l});
       end
       
       delta{L} = (a{L}-y).* a{L} .*(1-a{L});
       
       for l=L-1:-1:2
           delta{l} = bc(w{l}, z{l}, delta{l+1});
       end
       
       for l = 1:L-1
           gw= delta{l+1} * [x{l};a{l}]' / mini_batch;
           w{l} = w{l} -alpha * gw;
       end
       
       J = [J 1/2/mini_batch*sum((a{L}(:)-y(:)).^2)];
       
       [~,ind_train] = max(y);
       [~,ind_pred] = max(a{L});
       Acc= [Acc sum(ind_train == ind_pred) / mini_batch];
       if mod(k,20)==0
           fprintf('epoch %d: Accuracy on training dataset is %f%%\n',k, (sum(ind_train == ind_pred) / mini_batch)*100);
       end  
   end
end


figure
plot(J);
%saveas(gcf, 'J.jpg');
figure
plot(Acc);
%saveas(gcf, 'Acc.jpg');
% save model
save model.mat w layer_size



% test
a{1} = zeros(layer_size(1,2),train_size);
for l=1:L-1
       a{l+1} = fc(w{l}, a{l}, x_train{l});
end
[~,ind_test] = max(trainLabels);
[~,ind_pred] = max(a{L});
train_acc= sum(ind_test== ind_pred)/train_size;
fprintf('Accuracy on training dataset is %f%%\n', train_acc*100);

a{1} = zeros(layer_size(1,2),test_size);
for l=1:L-1
       a{l+1} = fc(w{l}, a{l}, x_test{l});
end
[~,ind_test] = max(testLabels);
[~,ind_pred] = max(a{L});
test_acc= sum(ind_test== ind_pred)/test_size;
fprintf('Accuracy on testing dataset is %f%%\n', test_acc*100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your code ABOVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%