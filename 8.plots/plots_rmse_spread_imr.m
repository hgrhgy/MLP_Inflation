

% train rmse and spread
case_no=5;
gs_casename='basic_newloss_imr_rl_gs';
gs_rslt = load(sprintf('result/%s/%s_result_%d.mat', gs_casename,gs_casename, 10000)).rslt;
gs_rmse = gs_rslt.mse;
gs_spread = gs_rslt.spread;

total_loop = 20;
x=1:total_loop;
y1=[];
y2=[];
casename='basic_newloss_imr_rl';
for i=1:total_loop
    loop = i * 10000;
    rslt = load(sprintf('result/%s/%s_result_%d.mat', casename,casename, loop)).rslt;
    fprintf('%d & %.2f & %.2f & %.2f  \\\\ \n', i, rslt.mse, rslt.spread, rslt.mse/rslt.spread);
    y1(i)=rslt.mse;
    y2(i)= rslt.spread;
end
[ax, h1, h2] = plotyy(x,y1,x,y2,@plot);
h3 =line(ax(1), [0,total_loop],[ gs_rmse,gs_rmse], 'LineStyle', '-', 'LineWidth', 3, 'Color', 'blue');
h4 = line(ax(2), [0,total_loop],[ gs_spread,gs_spread], 'LineStyle', '-.','LineWidth', 3, 'Color', 'blue');
set(get(ax(1), 'ylabel'), 'string', 'RMSE', 'fontsize', 24);
set(get(ax(2), 'ylabel'), 'string', 'Spread', 'fontsize', 24);
xlabel('epoch', 'fontsize', 24);
set(h1, 'Linestyle', '-', 'Marker', '*', 'LineWidth', 3, 'Color', 'red');
set(h2, 'Linestyle', '-.', 'Marker', '*', 'LineWidth', 3, 'Color', 'red');

set(ax(1), 'ytick', linspace(0.5, 1.3, 9), 'fontsize',24);
set(ax(2), 'ytick', linspace(0.5, 1.3, 9), 'fontsize', 24);
ylim(ax(1), [0.5,1.3]);
ylim(ax(2), [0.5,1.3]);
set(gcf, 'color', 'white');
set(gca, 'linewidth', 0.5);
legend([h1,h2,h3,h4], {'RMSE(M22)', 'Spread(M22)', 'RMSE(A09)', 'Spread(A09)'}, 'fontsize', 24)

