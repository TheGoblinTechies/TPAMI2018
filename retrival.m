function retrival(Num_layers, Num_pretrain, Num_train, Num_AE)
trn_label = load('/home/ubuntu/SZL/TPAMI2018/transfer/train-label.txt');
tst_label = load('/home/ubuntu/SZL/TPAMI2018/transfer/test-label.txt');
load_path = sprintf('/home/ubuntu/SZL/TPAMI2018/keyFile/feat16_trn.mat');
binary_train = load(load_path);
load_path = sprintf('/home/ubuntu/SZL/TPAMI2018/keyFile/feat16_tst.mat');
binary_test = load(load_path);
%binary_train = load('/home/zwz/zwz/cvpr16-deepbit/autoencoder/mnistAutoender/64bit/trainbifeat/trn_bifeat_l2p6t6.mat');
% binary_test = load('/home/zwz/zwz/cvpr16-deepbit/autoencoder/mnistAutoender/64bit/testbifeat/tst_bifeat_l2p6t6.mat');
binary_train = binary_train.prob_train;
binary_test = binary_test.prob_test;
top_k = 5000;
[map, ~] = precision( trn_label, binary_train', tst_label, binary_test', top_k, 1);
fprintf('MAP = %f\n',map);
ends