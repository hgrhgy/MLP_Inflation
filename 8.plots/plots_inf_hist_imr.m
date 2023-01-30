gs_casename='basic_newloss_imr_rl_gs';
gs_rslt = load(sprintf('result/%s/%s_result_%d.mat', gs_casename,gs_casename, 10000)).rslt;


perfectcase = 'basic_newloss_imr_rl';
perfectloop = 70000;
perfect_rslt = load(sprintf('result/%s/%s_result_%d.mat', perfectcase, perfectcase, perfectloop)).rslt;

h1 = histfit(reshape(perfect_rslt.inflation,1,[]), 1000,'normal');
% mean(var(perfect_rslt.inflation))

hold on
h2 = histfit(reshape(gs_rslt.inflation,1,[]), 1000, 'normal');


% var(gs_rslt.inflation,[1,2])
