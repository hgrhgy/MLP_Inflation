
refcase = 'GS';
refloop = 10000;
ref_rslt = load(sprintf('result/%s/%s_result_%d.mat', refcase, refcase, refloop)).rslt;

perfectcase = 'basic_newloss_rl_lb0';
perfectloop = 90000;
perfect_rslt = load(sprintf('result/%s/%s_result_%d.mat', perfectcase, perfectcase, perfectloop)).rslt;


perfect_inflation = perfect_rslt.inflation(10001:11000);
ref_inflation  = ref_rslt.inflation(10001:11000);


x=1:1000;

s1 = scatter(x, perfect_inflation,'filled', 'red');
hold on
s2 = scatter(x, ref_inflation, 'filled','blue');


xlabel('time', 'fontsize', 14);
ylabel('inflation', 'fontsize', 14);

legend([s1, s2], {'MLP', 'A09'}, 'fontsize', 14)
