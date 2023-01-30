
refcase = 'basic_newloss_imr_rl_gs';
refloop = 10000;
ref_rslt = load(sprintf('result/%s/%s_result_%d.mat', refcase, refcase, refloop)).rslt;

perfectcase = 'basic_newloss_imr_rl';
perfectloop = 110000;
perfect_rslt = load(sprintf('result/%s/%s_result_%d.mat', perfectcase, perfectcase, perfectloop)).rslt;


perfect_inflation = perfect_rslt.inflation(10001:11000);
ref_inflation  = ref_rslt.inflation(10001:11000);


x=10001:11000;

s1 = scatter(x, perfect_inflation,'filled', 'red');
hold on
s2 = scatter(x, ref_inflation, 'filled','blue');



xlabel('time', 'fontsize', 24);
ylabel('inflation', 'fontsize', 24);
set(gca, 'XTickLabel', get(gca,"XTick"))
ylim([1,6])

legend([s1, s2], {'M22', 'A09'}, 'fontsize', 24)
