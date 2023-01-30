
refcase = 'basic_newloss_imr_rl_gs';
refloop = 10000;
ref_rslt = load(sprintf('result/%s/%s_result_%d.mat', refcase, refcase, refloop)).rslt;

perfectcase = 'basic_newloss_imr_rl';
perfectloop = 90000;
perfect_rslt = load(sprintf('result/%s/%s_result_%d.mat', perfectcase, perfectcase, perfectloop)).rslt;

m22_rmse = sqrt(mean(power(mean(perfect_rslt.post_state(10001:11000,:,:),3) - perfect_rslt.true_state(10001:11000,:), 2),2)) ;
m22_spread  = mean(std(perfect_rslt.post_state(10001:11000, :, :),0,3),2);
a09_rmse = sqrt(mean(power(mean(ref_rslt.post_state(10001:11000,:,:),3) - ref_rslt.true_state(10001:11000,:), 2),2)) ;
a09_spread  = mean(std(ref_rslt.post_state(10001:11000, :, :),0,3),2);

x=10001:11000;

s1 = scatter(x, m22_rmse, '*', MarkerEdgeColor='red');
hold on
s2 = scatter(x, m22_spread, 'o', MarkerEdgeColor='red');
hold on
s3 = scatter(x, a09_rmse,  '*', MarkerEdgeColor='blue');
hold on
s4 = scatter(x, a09_spread, 'o', MarkerEdgeColor='blue');



set(gca, 'XTickLabel', get(gca,"XTick"))

xlabel('time', 'fontsize', 24);
ylabel('RMSE and Spread', 'fontsize', 24);

legend([s1, s2, s3, s4], {'RMSE(M22)', 'Spread(M22)','RMSE(A09)', 'Spread(A09)'}, 'fontsize', 24)
