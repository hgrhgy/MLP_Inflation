
refcase = 'basic_newloss_rl_gs';
refloop = 10000;
ref_rslt = load(sprintf('result/%s/%s_result_%d.mat', refcase, refcase, refloop)).rslt;

perfectcase = 'basic_newloss_rl';
perfectloop = 110000;
perfect_rslt = load(sprintf('result/%s/%s_result_%d.mat', perfectcase, perfectcase, perfectloop)).rslt;

inf_m22 = mean(perfect_rslt.inflation,1);
inf_a09 = mean(ref_rslt.inflation,1);



x=1:40;

s1 = scatter(x, inf_m22, '*', MarkerEdgeColor='red');
hold on
s2 = scatter(x, inf_a09, '*', MarkerEdgeColor='blue');



set(gca, 'XTickLabel', get(gca,"XTick"))

xlabel('time', 'fontsize', 24);
ylabel('inflation', 'fontsize', 24);
ylim([1,  1.15])

legend([s1, s2], {'M22', 'A09'}, 'fontsize', 24)



