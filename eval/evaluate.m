clear; 
addpath('./tools');

threshold = 0.5;   % threshold for generate positive predictions

inputpath = '../logs/rsn101_tinyimagenet';

files = dir(fullfile(inputpath,'results', '*test*.mat'));
files = {files.name};

fprintf('epoch\tmAP\tF1-C\tP-C\tR-C\tF1-O\tP-O\tR-O\n');
for i=1:numel(files)
    input = files{i};
    
    matcontent = load(fullfile(inputpath,'results', input));
    probs = double(matcontent.predProb);
    predLabels = double(matcontent.predLabel);
    labels = double(matcontent.gtLabel);
    % evaluate
    mAP_voc = AP_VOC(labels, probs);
    [P_C, R_C, F1_C] = precision_recall_f1(labels, predLabels);
    [P_O, R_O, F1_O] = precision_recall_f1(labels(:), reshape(predLabels,[],1));

    metrics = 100*[mean(mAP_voc), mean(F1_C), mean(P_C), mean(R_C), F1_O, P_O, R_O];
    fprintf('%d\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n',i,metrics(1),metrics(2),metrics(3),metrics(4),metrics(5),metrics(6),metrics(7));
end