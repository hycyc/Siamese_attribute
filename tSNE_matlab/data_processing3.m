siamese_pred = load('/home/xiaoyao/data/123/model_siameseprediction.mat');
siamese_pred = siamese_pred.data;
lstm_pred = load('/home/xiaoyao/data/123/model_LSTMbaselineprediction.mat');
lstm_pred = lstm_pred.data;
cnn_pred = load('/home/xiaoyao/data/123/model_CNNprediction.mat');
cnn_pred = cnn_pred.data;
contextsum_pred = load('/home/xiaoyao/data/123/model_ContextSumprediction.mat');
contextsum_pred = squeeze(contextsum_pred.data);
contextweighted_pred = load('/home/xiaoyao/data/123/model_ContextWeightedprediction.mat');
contextweighted_pred = squeeze(contextweighted_pred.data);
y_label = load('/home/xiaoyao/data/123/model_siamesegroundtruth.mat');
y_label = y_label.data;
fprintf('%d\n', 0);
contextsumX = tsne(contextsum_pred);
save contextsum.mat;
fprintf('%d\n',4);
