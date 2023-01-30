

% train rmse and spread
case_no=5;
gs_casename='basic_newloss_rl_gs';
gs_rslt = load(sprintf('result/%s/%s_result_%d.mat', gs_casename,gs_casename, 10000)).rslt;

total_loop =10;
x=1:total_loop;
y1=[];
y2=[];

b = bar(gs_rslt.rank);
std(gs_rslt.rank)
ylim([0 20000]);

xlabel('rank', 'fontsize',24);
ylabel('P(x)', 'fontsize', 24);
set(gcf, "Position", [0,0,1200,900])
ax = gca;
ax.FontSize=24;
saveas(gcf, 'F:\BaiduNetdiskWorkspace\论文写作\MWR\mlp_inflation\MWR_V6.1\fig\rl_a09_rank.png', 'png');

casename = 'basic_newloss_rl';


clf
for i=1:20
   
    loop = i * 10000;
    rslt = load(sprintf('result/%s/%s_result_%d.mat', casename, casename, loop)).rslt;
    
    b = bar(rslt.rank);
    std(rslt.rank)
    ylim([0 20000]);
    
    xlabel('rank', 'fontsize',24);
    ylabel('P(x)', 'fontsize', 24);
    set(gcf, "Position", [0,0,1200,900])
    ax = gca;
    ax.FontSize=24;
    saveas(gcf, sprintf("F:\\BaiduNetdiskWorkspace\\论文写作\\MWR\\mlp_inflation\\MWR_V6.1\\fig\\rl_epoch%02d_rank.png", i), 'png');
    clf
end


