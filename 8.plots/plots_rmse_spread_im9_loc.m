
refcase = 'basic_newloss_im9_rl_gs';
refloop = 10000;
ref_rslt = load(sprintf('result/%s/%s_result_%d.mat', refcase, refcase, refloop)).rslt;

perfectcase = 'basic_newloss_im9_rl';
perfectloop = 90000;
perfect_rslt = load(sprintf('result/%s/%s_result_%d.mat', perfectcase, perfectcase, perfectloop)).rslt;

rmse_mlp = sqrt(mean(power((mean(perfect_rslt.post_state(10001:11000, :, :),3) - perfect_rslt.true_state(10001:11000, :)),2),1));
rmse_ref = sqrt(mean(power((mean(ref_rslt.post_state(10001:11000, :, :),3) - ref_rslt.true_state(10001:11000, :)),2),1));

spread_mlp = mean(std(perfect_rslt.post_state(10001:11000, :, :),0,3),1);
spread_ref = mean(std(ref_rslt.post_state(10001:11000, :, :),0,3),1);


x=linspace(0,1,40);
hold on
[ax, h1, h2] = plotyy(x,rmse_mlp,x,spread_mlp,@plot);


h3 = line(ax(1), x, rmse_ref, 'Color', 'blue', 'linestyle', '-', 'linewidth', 3);
h4 = line(ax(2), x, spread_ref, 'linestyle', '-.', 'Color', 'blue', 'linewidth', 3);

set(get(ax(1), 'ylabel'), 'string', 'RMSE', 'fontsize', 24);
set(get(ax(2), 'ylabel'), 'string', 'Spread', 'fontsize', 24);
xlabel('Location', 'fontsize', 24);
set(h1, 'Linestyle', '-', 'Marker', '*', 'LineWidth', 3, 'Color', 'red');
set(h2, 'Linestyle', '-.', 'Marker', '*', 'LineWidth', 3, 'Color', 'red');

set(ax(1), 'ytick', linspace(0.0, 1.8, 9), 'fontsize',24);
set(ax(2), 'ytick', linspace(0.0, 1.8, 9), 'fontsize', 24);
ylim(ax(1), [0.0,1.8]);
ylim(ax(2), [0.0,1.8]);
set(gcf, 'color', 'white');
legend([h1,h2,h3,h4], {'RMSE(M22)', 'Spread(M22)', 'RMSE(A09)', 'Spread(A09)'}, 'fontsize', 24)

mean(rmse_mlp/spread_mlp)
mean(rmse_ref/spread_ref)



