clear 
S = dir('mobile_preds_kurt');

srocc = zeros(length(S)-2,1);
lcc = zeros(length(S)-2,1);
for k=3:length(S)
    filename = S(k).name;
    fullFileName = fullfile(S(k).folder, filename);
    load(fullFileName)
    srocc(k-2) = corr(pred.',y.','Type','Spearman');
    [ beta, ehat, J ] = train_nonlinear_map(pred,y);
    preds_rescale = beta(2)+(beta(1)-beta(2))./(1+exp(-((pred-beta(3))/abs(beta(4)))));
    lcc(k-2) = corr(preds_rescale.',y.');
end
disp(median(srocc))
disp(median(lcc))