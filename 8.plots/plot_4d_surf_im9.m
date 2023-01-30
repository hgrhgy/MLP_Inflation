


casename = 'basic_newloss_im9_rl';
loop = 80000;
inweights = load(sprintf('%s/%s_inweights_%d.mat', casename,casename, loop)).inweights_save;
outweights = load(sprintf('%s/%s_outweights_%d.mat', casename, casename, loop)).outweights_save;

im9_rslt = load(sprintf('result/%s/%s_result_%d.mat', casename, casename, loop)).rslt;

smax = max(im9_rslt.true_state, [],'all');
smin = min(im9_rslt.true_state,[],'all');
xvar = var(im9_rslt.true_state,[],'all');

xb = linspace( round(smin - 1,1), round(smax+1,1), 10);
yo = linspace( round(smin - 1,1), round(smax+1,1), 10);
xv = linspace(0, round(xvar,1), 10);
inf = [];
xx = [];
yy = [];
yy = [];
for i = 1:10
    for j = 1:10
        for k = 1:10
        [output_activations,hidden_activation,hidden_activation_raw,inputs_with_bias] = ...
                    FORWARDPASS(inweights,outweights, [xb(i), xv(j), yo(k), 1] , 'relu');
        xx(i,j,k) = xb(i);
        yy(i,j,k) = xv(j);
        zz(i,j,k) = yo(k);
        inf(i,j,k) = output_activations;
        end
    end
end
subplot(1,3,1)
line(reshape(mean(xx,[2,3]),1,[]), reshape(mean(inf,[2,3]),1,[]));
subplot(1,3,2)
line(reshape(mean(yy,[1,3]),1,[]), reshape(mean(inf,[1,3]),1,[]));
subplot(1,3,3)
line(reshape(mean(zz,[1,2]),1,[]), reshape(mean(inf,[1,2]),1,[]));
% 
% scatter3(reshape(xx,1,[]),reshape(yy,1,[]),reshape(zz,1,[]), 50, reshape(inf,1,[]), "filled")
% 
% xlabel('xb');
% ylabel('xv');
% zlabel('yo');
% colorbar

