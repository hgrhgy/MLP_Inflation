

% train rmse and spread
case_no=5;
gs_casename='basic_newloss_imr_rl_gs';
gs_rslt = load(sprintf('result/%s/%s_result_%d.mat', gs_casename,gs_casename, 10000)).rslt;
casename = 'basic_newloss_imr_rl';

% subplot(2,3,1);
% bar(gs_rslt.rank);
% std(gs_rslt.rank)
% ylim([0 80000]);
% title(sprintf("A09"), FontSize=24)
% xlabel('rank', 'fontsize',24);
% ylabel('P(x)', 'fontsize', 24);

% rslt = load(sprintf('result/%s/%s_result_%d.mat', casename, casename, 10000)).rslt;
% subplot(2,3,2);
% b2 = bar(rslt.rank);
% std(rslt.rank)
% ylim([0 80000]);
% title(sprintf("epoch %d", 1), FontSize=24)
% xlabel('rank', 'fontsize',24);
% ylabel('P(x)', 'fontsize', 24);

for i=1:20
    loop = i * 10000;
    rslt = load(sprintf('result/%s/%s_result_%d.mat', casename, casename, loop)).rslt;
    subplot(4,5,i);
    bar(rslt.rank);
    std(rslt.rank)
    ylim([0 80000]);
    xlabel(sprintf("Epoch %d", i), 'fontsize', 20);
%     ylabel('P(x)', 'fontsize', 16);
end


