
bins = 200;
set(gcf, "Position", [0,0,1200,900])
rl_gs_casename='basic_newloss_rl_gs';
rl_gs_rslt = load(sprintf('result/%s/%s_result_%d.mat', rl_gs_casename,rl_gs_casename, 10000)).rslt;
rl_casenme = 'basic_newloss_rl';
rl_loop = 70000;
rl_rslt = load(sprintf('result/%s/%s_result_%d.mat', rl_casenme, rl_casenme, rl_loop)).rslt;
gs_rl_hist = histfit(reshape(rl_gs_rslt.inflation,1,[]), bins, 'normal');
gs_rl_hist(1).FaceColor='blue';
gs_rl_hist(2).Color = '#aaaaaa';
hold on
rl_hist = histfit(reshape(rl_rslt.inflation,1,[]), bins, 'normal' );
rl_hist(1).FaceColor='red';
rl_hist(2).Color = 'black';
xlim([0.8,1.3])
ylim([0, 3 * 1E5])
legend([rl_hist(1),rl_hist(2),gs_rl_hist(1),gs_rl_hist(2)], {'HIST(M22)', 'PDF(M22)', 'HIST(A09)', 'PDF(A09)'}, 'fontsize', 20)
ax = gca;
ax.FontSize=20;
ylabel('freq','FontSize',20)
xlabel('inflation','FontSize',20)
rl_dist = fitdist(rl_rslt.inflation(:),'Normal');
gs_rl_dist = fitdist(rl_gs_rslt.inflation(:), 'Normal');
%p = cdf('normal', 1, gs_rl_dist.mu, gs_rl_dist.sigma);
% p = cdf('normal', 1, rl_dist.mu, rl_dist.sigma);
% p

text(0.8 + (1.3-0.8) * 0.05, 3 * 1E5 *0.80,sprintf('\\mu_{M22} = %.2f\n\\sigma^2_{M22} = %.2f\n\\mu_{A09} = %.2f\n\\sigma^2_{A09} = %.2f', rl_dist.mu, rl_dist.sigma, gs_rl_dist.mu, gs_rl_dist.sigma),'FontSize',20);

saveas(gcf, sprintf("F:\\BaiduNetdiskWorkspace\\论文写作\\MWR\\mlp_inflation\\MWR_V6.1\\fig\\rl_hist_fit.png"), 'png');
clf
set(gcf, "Position", [0,0,1200,900])
imr_gs_casename='basic_newloss_imr_rl_gs';
imr_gs_rslt = load(sprintf('result/%s/%s_result_%d.mat', imr_gs_casename,imr_gs_casename, 10000)).rslt;
imr_casenme = 'basic_newloss_imr_rl';
imr_loop = 100000;
imr_rslt = load(sprintf('result/%s/%s_result_%d.mat', imr_casenme, imr_casenme, imr_loop)).rslt;
gs_imr_hist =histfit(reshape(imr_gs_rslt.inflation,1,[]), bins, 'normal');
gs_imr_hist(1).FaceColor='blue';
gs_imr_hist(2).Color = '#aaaaaa';
hold on
imr_hist = histfit(reshape(imr_rslt.inflation,1,[]), bins, 'normal');
imr_hist(1).FaceColor='red';
imr_hist(2).Color = 'black';
xlim([0.6,6])
ylim([0.0, 1.5 * 1E5])
legend([imr_hist(1),imr_hist(2),gs_imr_hist(1),gs_imr_hist(2)], {'HIST(M22)', 'PDF(M22)', 'HIST(A09)', 'PDF(A09)'}, 'fontsize', 20)
% title('IM\_Exp\_gs','FontSize',16)
ax = gca;
ax.FontSize=20;
ylabel('freq','FontSize',20)
xlabel('inflation','FontSize',20)
imr_dist = fitdist(imr_rslt.inflation(:),'Normal');
imr_gs_dist = fitdist(imr_gs_rslt.inflation(:), 'Normal');
%p = cdf('normal', 1, imr_gs_dist.mu, imr_gs_dist.sigma);
p = cdf('normal', 1, imr_dist.mu, imr_dist.sigma);

text(0.6 + (6-0.6) * 0.05, 1.5 * 1E5 *0.80,sprintf('\\mu_{M22}: %.2f\n\\sigma^2_{M22}: %.2f\n\\mu_{A09}: %.2f\n\\sigma^2_{A09}: %.2f', imr_dist.mu, imr_dist.sigma,imr_gs_dist.mu, imr_gs_dist.sigma),'FontSize',20);


saveas(gcf, sprintf("F:\\BaiduNetdiskWorkspace\\论文写作\\MWR\\mlp_inflation\\MWR_V6.1\\fig\\imr_hist_fit.png"), 'png');
clf

set(gcf, "Position", [0,0,1200,900])
im9_gs_casename='basic_newloss_im9_rl_gs';
im9_gs_rslt = load(sprintf('result/%s/%s_result_%d.mat', im9_gs_casename,im9_gs_casename, 10000)).rslt;
im9_casenme = 'basic_newloss_im9_rl';
im9_loop = 90000;
im9_rslt = load(sprintf('result/%s/%s_result_%d.mat', im9_casenme, im9_casenme, im9_loop)).rslt;
im9_gs_hist = histfit(reshape(im9_gs_rslt.inflation,1,[]), bins, 'normal');
im9_gs_hist(1).FaceColor='blue';
im9_gs_hist(2).Color = '#aaaaaa';
hold on
im9_hist = histfit(reshape(im9_rslt.inflation,1,[]), bins, 'normal');
im9_hist(1).FaceColor='red';
im9_hist(2).Color = 'black';
xlim([0.6,2])
ylim([0, 2* 1E5])
legend([im9_hist(1),im9_hist(2),im9_gs_hist(1),im9_gs_hist(2)], {'HIST(M22)', 'PDF(M22)', 'HIST(A09)', 'PDF(A09)'}, 'fontsize', 20)
ax = gca;
ax.FontSize=20;
ylabel('freq','FontSize',20)
xlabel('inflation', 'FontSize',20)
im9_dist = fitdist(im9_rslt.inflation(:),'Normal');

im9_gs_dist = fitdist(im9_gs_rslt.inflation(:), 'Normal');
%p = cdf('normal', 1, im9_gs_dist.mu, im9_gs_dist.sigma);
% p = cdf('normal', 1, im9_dist.mu, im9_dist.sigma);
text(0.6 + (2-0.6) * 0.05, 2 * 1E5 *0.80,sprintf('\\mu_{M22}: %.2f\n\\sigma^2_{M22}: %.2f\n\\mu_{A09}: %.2f\n\\sigma^2_{A09}: %.2f', im9_dist.mu, im9_dist.sigma, im9_gs_dist.mu, im9_gs_dist.sigma),'FontSize',20);

saveas(gcf, sprintf("F:\\BaiduNetdiskWorkspace\\论文写作\\MWR\\mlp_inflation\\MWR_V6.1\\fig\\im9_hist_fit.png"), 'png');
clf
