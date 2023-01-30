global casename;
global resname;



handles.scale_factor = 2.0;

handles.training_cnt = 0;








handles.observation = [];

handles.error_mlp = [];
handles.spread_mlp = [];
handles.error_gs = [];
handles.spread_gs = [];


basic_loop = 10000;
train_loop = 10000;
handles.inflation_time = [];
casename = "basic_newloss_rl_pdf";
resname = ["basic_newloss_rl_pdf", "basic_newloss_imr_pdf", "basic_newloss_im9_pdf"];


inf_cfg = [ [1.07, 0.02],
    [5.08, 0.07],
    [1.36, 0.02]];

for file_cnt = 1:3
    weitghts_file_cnt =file_cnt * train_loop;
    loop = round(basic_loop * 1.1);
    init_flag=false;
    forcing = 8;
    if file_cnt == 3
        forcing = 9;
    end
    handles = initialize_data(init_flag, inf_cfg(file_cnt,:), forcing);
    
    handles.inflation_type='Adaptive Inflation';
%     handles.inflation_type='MLP';
    % handles.inflation_type='Fixed Inflation';
    handles.filter_type='EAKF';
    handles.loss = [];
    handles.diff = [];
    handles.cname = resname(file_cnt);
    
    
    for i=1:loop
       handles=step_ahead(handles);
    end
    
    rslt = struct("inflation_type", handles.inflation_type, ...
        "filter_type", handles.filter_type, ...
        "file_cnt", file_cnt * train_loop, ...
        "mse",mean(handles.posterior_rms(basic_loop:loop)), ...
        "spread", mean(handles.posterior_spread(basic_loop:loop)), ...
        "inflation", handles.inflation_time,...
        "rank",handles.post_rank,...
        "true_state", handles.true_state, ...
        "post_state", handles.posterior_state, ...
        "prior_state", handles.prior_state);
    

    if ~exist(sprintf('result/%s',casename))
        mkdir(sprintf('result/%s',casename))
    end
    save(sprintf('result/%s/%s_result.mat',casename,resname(file_cnt) ),"rslt");

    fprintf('mse: %f, spread: %f\n', mean(handles.posterior_rms(basic_loop:loop)), mean(handles.posterior_spread(basic_loop:loop)));
    
end






function h = step_ahead(handles)
        % next time state
            
        % 1. forward
        [new_truth, new_time] = lorenz_96_adv_1step(handles.true_state(handles.time, :), handles.time, handles.true_forcing);
        handles.true_state(new_time, :) = new_truth;
        
        for imem = 1:handles.ens_size
            [new_ens, new_time] = lorenz_96_adv_1step(handles.posterior_state(handles.time, :, imem), handles.time, handles.forcing);
            if (handles.cname == "basic_newloss_imr_pdf")
                new_ens = new_ens + (randn(1, handles.model_size) * handles.model_var + handles.model_error);
            end
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
%                     gamma, handles.inflation_Std_Min, handles.ens_size, 'Gaussian');

                upd_inf=handles.cfg(1);

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
        
             
        % Compute the posterior rank histograms
        for i = 1:handles.ens_size
            ens_rank = get_ens_rank(squeeze(handles.posterior_state(new_time, i, :)), ...
                squeeze(handles.true_state(new_time, i)));
            handles.post_rank(ens_rank) = handles.post_rank(ens_rank) + 1;
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

    function h = initialize_data(init_flag, cfg, forcing)
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
        handles.forcing      = forcing;
        handles.model_size   = 40;
        handles.delta_t      = L96.delta_t;

        handles.filter_type_string = 'EAKF';
        handles.localization = 0.3;
        handles.inflation_Damp = 0.9;
        handles.inflation_Std = cfg(2);
        handles.inflation_Min     = 1;
        handles.inflation_Max     = 5;
        handles.inflation_Std_Min = cfg(2);
        handles.model_error  = 1;
        handles.model_var    = 1;
        handles.cfg = cfg;
        clear handles.true_state
        
        handles.ens_size                    = 20;
        handles.true_state(1, 1:handles.model_size) = handles.true_forcing;
        handles.true_state(1, 1)            = 1.001 * handles.true_forcing;
        handles.time                        = 1;
        handles.prior                       = 0;
        handles.prior_rms                   = 0;
        handles.prior_spread                = 0;
        handles.prior_inf(1, 1:handles.model_size)  = 1 * cfg(1);
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
        handles.num_hidden_units = handles.ens_size * handles.ens_size;
        handles.num_targets = 1;
        handles.weight_range = 1;
        handles.weight_center = 0;
        handles.outputrule = 'relu';
        handles.mse_loss = [];
        handles.learning_rate = 1E-2;
        
        handles.guess_times=10;
     
        DELTA_T=handles.delta_t;
        TRUE_FORCING=handles.true_forcing;
        FORCING=handles.forcing;
        MODEL_SIZE=handles.model_size;
        h = handles;
end

function inf=guess(inf_gs, times)
    inf = ones(times, length(inf_gs));
    for i =1 : times
    inf(i,:) = inf_gs + randn(1, length(inf_gs))*0.1;
    end
    
end

