global TRUE_FORCING;
global FORCING;
global MODEL_SIZE;
global DELTA_T;
global casename;






handles.scale_factor = 2.0;

handles.training_cnt = 0;


% handles.inflation_type='Adaptive Inflation';
handles.inflation_type='MLP';
% handles.inflation_type='Fixed Inflation';
handles.filter_type='EAKF';




handles.observation = [];

handles.error_mlp = [];
handles.spread_mlp = [];
handles.error_gs = [];
handles.spread_gs = [];
handles.loss = [];

handles.train_mc = 0;
epochs = 20;
training_loop = 10000;
loop =training_loop;
conitnue_loop = 0;
casename = "basic_newloss_rl_e80";


for ep = 1: epochs - conitnue_loop/loop
    if ep == 1 && conitnue_loop==0
        init_flag = true;
       tmp_mc = 0;
    else
        init_flag = false;
        tmp_mc = handles.train_mc;
    end
   
    handles = initialize_data(init_flag, (ep -1) * training_loop + conitnue_loop );
    handles.inflation_type='MLP';
    handles.train_mc = tmp_mc;
    handles.epoch = ep;

    for i=1:loop
       handles=step_ahead(handles);
    end

    if ~exist(casename)
        mkdir(casename)
    end

    inweights_save = handles.inweights;
    save(sprintf('%s/%s_inweights_%d.mat',casename,casename, i * ep + conitnue_loop ),'inweights_save')

    outweights_save = handles.outweights;
    save(sprintf('%s/%s_outweights_%d.mat', casename,casename, i * ep + conitnue_loop),'outweights_save')

    fprintf('epoch %d, mse_loss:%f\n', round(ep + conitnue_loop/loop), mean(handles.mse_loss));
end



function h = step_ahead(handles)
        % next time state
            
        % 1. forward
        [new_truth, new_time] = lorenz_96_adv_1step(handles.true_state(handles.time, :), handles.time, handles.true_forcing);
        handles.true_state(new_time, :) = new_truth;
        
        for imem = 1:handles.ens_size
            [new_ens, new_time] = lorenz_96_adv_1step(handles.posterior_state(handles.time, :, imem), handles.time, handles.forcing);
            handles.prior_state(new_time, :, imem) = new_ens;
        end

        for imz = 1: handles.model_size
            handles.prior_state_mean(new_time, imz)=mean(handles.prior_state(new_time, imz, :));
            handles.prior_state_var(new_time, imz)=var(handles.prior_state(new_time, imz,:));
        end
        handles.time = new_time;

        % save the prior_inflation for mlp training
        handles.train_inf = handles.prior_inf;
        handles.train_prior = squeeze(handles.prior_state(new_time, :, :));

        % 2. do inflation
        handles.prior_inf = 1.0 + handles.inflation_Damp * ( handles.prior_inf - 1.0 );
        
    
        for i=1:handles.model_size
            ens_mean = handles.prior_state_mean(new_time, i);
            handles.prior_state_after_inf(new_time, i, :) = ens_mean + ...
                sqrt(handles.prior_inf(1, i)) * (handles.prior_state(new_time, i, :) - ens_mean);
            handles.prior_state_after_inf_mean(new_time, i)=mean(handles.prior_state_after_inf(new_time, i, :));
            handles.prior_state_after_inf_var(new_time, i)=var(handles.prior_state_after_inf(new_time, i, :));
        end

        % 3. generate observation
        obs_sd = 1;
        obs_error_var = obs_sd^2;
        obs_error = obs_sd * randn(1,handles.model_size);
        obs = handles.true_state(new_time, :) + obs_error;

        temp_ens = squeeze(handles.prior_state_after_inf(new_time, :, :));
        prior_inf = handles.prior_inf;
        for i = 1: handles.model_size
            
           
            obs_prior = temp_ens(i,:);

            switch handles.filter_type_string
                case 'EAKF'
                    obs_increments = obs_increment_eakf(obs_prior, obs(i), obs_error_var);
                case 'EnKF'
                    obs_increments = obs_increment_enkf(obs_prior, obs(i), obs_error_var);
                case 'RHF'
                    obs_increments = obs_increment_rhf (obs_prior, obs(i), obs_error_var);
                case 'No Assimilation'
                    %No Incrementation
                    obs_increments = 0;
            end
            
            for j = 1: handles.model_size
                [state_incs, r_xy]  = get_state_increments(temp_ens(j, :), ...
                    obs_prior, obs_increments);
                
                % localization
                dist = abs(i - j) / handles.model_size;
                if(dist > 0.5), dist = 1 - dist; end
                    
                % Compute the localization factor
                cov_factor = comp_cov_factor(dist, handles.localization);
                temp_ens(j, :) = temp_ens(j, :) + state_incs * cov_factor;
                
                % get the correlation factor 
                gamma = cov_factor * abs(r_xy);
                
                % Bayesian update of the inflation
%                 upd_inf = update_inflate( mean(obs_prior), var(obs_prior), obs(i), obs_error_var, ...
%                     prior_inf(j), handles.prior_inf(j), handles.inflation_Std, handles.inflation_Min, handles.inflation_Max, ...
%                     gamma,  handles.inflation_Std_Min, handles.ens_size, 'Gaussian');
                if handles.inflation_type == "MLP"
                    x_input = [mean(handles.train_prior(j,:)), ...
                                var(handles.train_prior(j,:)), obs(j), obs_sd];             
                                
                  [handles.output_activations,handles.hidden_activation,handles.hidden_activation_raw,handles.inputs_with_bias] = ...
                    FORWARDPASS(handles.inweights,handles.outweights,x_input ,handles.outputrule);


                    upd_inf = handles.output_activations;
                    if upd_inf <=0
                        upd_inf = 1;
                    end
                    if upd_inf < handles.inflation_Min
                    upd_inf= handles.inflation_Min;
                    end
                    
                    if upd_inf > handles.inflation_Max
                    upd_inf=handles.inflation_Max;
                    end
                    handles.diff(new_time, j) = handles.output_activations - upd_inf;

                end
                
                handles.inflation_time(new_time, j) = upd_inf;
                handles.prior_inf(j) =upd_inf;
            end

        end


        handles.posterior_state(new_time, :, :) =  temp_ens;

        handles.posterior_rms(new_time) = rms_error(handles.true_state(new_time, :), handles.posterior_state(new_time, :, :));
        handles.posterior_spread(new_time) = ens_spread(handles.posterior_state(new_time, :, :));
        
        handles.prior_rms(new_time) = rms_error(handles.true_state(new_time, :), handles.prior_state_after_inf(new_time, :, :));
        handles.prior_spread(new_time) = ens_spread(handles.prior_state_after_inf(new_time, :, :));
        for i = 1:handles.model_size
            ens_rank = get_ens_rank(squeeze(handles.posterior_state(new_time, i, :)), ...
                squeeze(handles.true_state(new_time, i)));
            handles.post_rank(ens_rank) = handles.post_rank(ens_rank) + 1;
        end

        rate = get_rate(handles.epoch);
        infs = guess(handles.train_inf, handles.guess_times, rate);
        rms_arr = ones(1, handles.guess_times);
        spread_arr=ones(1, handles.guess_times);
        rank_sd_arr = ones(1, handles.guess_times);
        for jj = 1: handles.guess_times

            inf_t = 1.0 + handles.inflation_Damp * (infs(jj,:) - 1.0);

            ens_prior_t = zeros(handles.model_size, handles.ens_size);

            for jjj =1: handles.model_size
                ens_mean_t = mean(handles.train_prior(jjj, :));
                ens_prior_t(jjj,:) = ens_mean_t + ...
                    sqrt(inf_t(jjj))*(handles.train_prior(jjj,:) - ens_mean_t);
            end

            temp_ens_t = ens_prior_t;
            

            for i = 1: handles.model_size
                obs_prior = temp_ens_t(i,:);

                switch handles.filter_type_string
                    case 'EAKF'
                        obs_increments = obs_increment_eakf(obs_prior, obs(i), obs_error_var);
                    case 'EnKF'
                        obs_increments = obs_increment_enkf(obs_prior, obs(i), obs_error_var);
                    case 'RHF'
                        obs_increments = obs_increment_rhf (obs_prior, obs(i), obs_error_var);
                    case 'No Assimilation'
                        %No Incrementation
                        obs_increments = 0;
                end
                
                 for j = 1: handles.model_size
                    [state_incs, r_xy]  = get_state_increments(temp_ens_t(j, :), ...
                        obs_prior, obs_increments);
                    
                    % localization
                    dist = abs(i - j) / handles.model_size;
                    if(dist > 0.5), dist = 1 - dist; end
                        
                    % Compute the localization factor
                    cov_factor = comp_cov_factor(dist, handles.localization);
                    temp_ens_t(j, :) = temp_ens_t(j, :) + state_incs * cov_factor;
                    

                end

            end
            
            rms_arr(jj) = rms_error(handles.true_state(new_time, :), temp_ens_t(:,:));
            spread_arr(jj) = ens_spread( temp_ens_t(:,:));
            temp_rank = handles.post_rank;
            for jjj = 1:handles.model_size
                tmp_ens_rank = get_ens_rank(temp_ens_t(jjj, :), ...
                    handles.true_state(new_time, jjj));
                temp_rank(tmp_ens_rank) =temp_rank(tmp_ens_rank) + 1;
            end
            rank_sd_arr(jj) = std(temp_rank);
        
        end
        J = rms_arr.*(rank_sd_arr/mean(rank_sd_arr).*exp(rms_arr./spread_arr - 1.0));
        
        [min_rms, min_rms_idx] = min(J);
        
        if (rms_arr(min_rms_idx) < handles.posterior_rms(new_time))
            % x = [stat_mean, stat_var, observation, obs_var, pre_infl]
            x_batch = ones(handles.model_size, 4);
            y_batch = ones(handles.model_size, 1);
            for bb=1:handles.model_size
                x_batch(bb, :) = [mean(handles.train_prior(bb,:)), ...
                    var(handles.train_prior(bb,:)), obs(bb), obs_sd];             
                y_batch(bb, :) = infs(min_rms_idx, bb);
            
              [handles.output_activations,handles.hidden_activation,handles.hidden_activation_raw,handles.inputs_with_bias] = ...
                FORWARDPASS(handles.inweights,handles.outweights,x_batch(bb, :) ,handles.outputrule);
              [handles.outweights, handles.inweights] = BACKPROP(handles.outweights,handles.inweights,...
                handles.output_activations, y_batch(bb, :) ,handles.hidden_activation,...  
                handles.hidden_activation_raw,handles.inputs_with_bias,handles.learning_rate,handles.outputrule);
            end
            
            handles.train_mc = handles.train_mc+1;
            mse_loss = 0;
            for bb=1:handles.model_size
              
                
                [handles.output_activations,handles.hidden_activation,handles.hidden_activation_raw,handles.inputs_with_bias] = ...
                FORWARDPASS(handles.inweights,handles.outweights,x_batch(bb, :) ,handles.outputrule);
                
                mse_loss = + (handles.output_activations - y_batch(bb, :))^2;
            end
            mse_loss = mse_loss/handles.model_size;

            handles.mse_loss(new_time) = mse_loss;
            % disp(fprintf('training step: %d, training loss: %f',handles.train_mc, mse_loss ))
        end
        
        
        
        h =  handles;


end
%% ----------------------------------------------------------------------

    function ens_mean_rms = rms_error(truth, ens)
        % Calculates the rms_error
        
        ens_mean = mean(squeeze(ens),2)';
        ens_mean_rms = sqrt(sum((truth - ens_mean).^2) / size(truth, 2));
    end

%% ----------------------------------------------------------------------

    function spread = ens_spread(ens)
        % Calculates the ens_spread
        % Remove the mean of each of the 40 model variables (40 locations).
        % resulting matrix is 40x20 ... each row/location is centered (zero mean).
        
        [~, model_size, ens_size] = size(ens);
        datmat = detrend(squeeze(ens)','constant'); % remove the mean of each location.
        denom  = (model_size - 1)*ens_size;
        spread = sqrt(sum(datmat(:).^2) / denom);
    end

    function h = initialize_data(init_flag,loop)
        global TRUE_FORCING;
        global FORCING;
        global MODEL_SIZE;
        global DELTA_T;
        global casename;
        % Reset all the figures and the data structures
        % Keep the current filter type, ensemble size and obs characteristics
        % Reset the time to 1 and be ready to advance
        
        % set random number seed to same value to generate known sequences
        % rng('default') is the Mersenne Twister with seed 0
        rng(0,'twister')
        
        % Set up global storage with initial values
        L96          = lorenz_96_static_init_model;
        handles.true_forcing = 8;
        handles.forcing      = 8;
        handles.model_size   = 40;
        handles.delta_t      = L96.delta_t;

        handles.filter_type_string = 'EAKF';
        handles.localization = 0.3;
        handles.inflation_Damp = 0.9;
        handles.inflation_Std =0.6;
        handles.inflation_Min     = 1;
        handles.inflation_Max     = 5;
        handles.inflation_Std_Min = 0.6;

        clear handles.true_state
        
        handles.ens_size                    = 80;
        handles.true_state(1, 1:handles.model_size) = handles.true_forcing;
        handles.true_state(1, 1)            = 1.001 * handles.true_forcing;
        handles.time                        = 1;
        handles.prior                       = 0;
        handles.prior_rms                   = 0;
        handles.prior_spread                = 0;
        handles.prior_inf(1, 1:handles.model_size)  = 1;
        handles.posterior                   = 0;
        handles.posterior_rms               = 0;
        handles.posterior_spread            = 0;
        handles.prior_rms               = 0;
        handles.prior_spread            = 0;

        
        handles.posterior_state = zeros(1, handles.model_size, handles.ens_size);
        for imem = 1:handles.ens_size
            handles.posterior_state(1, 1:handles.model_size, imem) = ...
                handles.true_state(1, :) + 0.001 * randn(1, handles.model_size);
        end

        handles.prior_state = handles.posterior_state;

        
        % An array to keep track of rank histograms
        handles.prior_rank = zeros(1, handles.ens_size + 1);
        handles.post_rank  = zeros(1, handles.ens_size + 1);
    

        for i = 1:handles.model_size
            handles.prior_state_mean(1,i) =mean(handles.prior_state(1,i,:));
            handles.prior_state_var(1,i) = var(handles.prior_state(1,i,:));
            handles.prior_state_after_inf_mean(1,i) =mean(handles.prior_state(1,i,:));
            handles.prior_state_after_inf_mean(1,i) = var(handles.prior_state(1,i,:));
        end
        handles.mlp_input_size = 4;
        handles.num_hidden_units = 400;
        handles.num_targets = 1;
        handles.weight_range = 1;
        handles.weight_center = 0;
        handles.outputrule = 'relu';
        handles.mse_loss = [];
        handles.learning_rate = get_lr(loop/10000) ;
        
        handles.guess_times=10;
        if (init_flag)

            [handles.inweights,handles.outweights] = getweights( handles.mlp_input_size, handles.num_hidden_units, ...
	        handles.num_targets, handles.weight_range, handles.weight_center);
             handles.train_gs = 0;
            handles.train_mc = 0;
        else
            handles.inweights = load(sprintf('%s/%s_inweights_%d.mat',casename,casename, loop)).inweights_save;
            handles.outweights = load(sprintf('%s/%s_outweights_%d.mat',casename,casename, loop)).outweights_save;
        end

        DELTA_T=handles.delta_t;
        TRUE_FORCING=handles.true_forcing;
        FORCING=handles.forcing;
        MODEL_SIZE=handles.model_size;
        h = handles;
end

function inf=guess(inf_gs, times, rate)
    inf = ones(times, length(inf_gs));
    for i =1 : times
        inf(i,:) = inf_gs + randn(1, length(inf_gs)) * rate;
    end
    inf(inf<0)=0;
end

function rate=get_rate(epoch)
    if (epoch < 2)
        rate = 0.5;
    elseif (epoch < 5)
        rate = 0.3;
    else
        rate = 0.1;
    end
end

function lr=get_lr(epoch)
    if (epoch < 3)
        lr = 1E-6;
    elseif (epoch < 5)
        lr = 5E-6;
    elseif (epoch < 7)
        lr = 1E-5;
    elseif (epoch < 9)
        lr = 5E-5;
    else
        lr = 1E-4;
    end
end
