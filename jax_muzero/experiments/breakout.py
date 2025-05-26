import os


# from ray import tune  # Removed ray.tune import


from jax_muzero.algorithms.muzero import Experiment


if __name__ == '__main__':
    config = {
        'env_id': 'Breakout',
        'env_kwargs': {},
        'seed': 42,
        'num_envs': 94,
        'unroll_steps': 5,
        'td_steps': 5,
        'max_search_depth': None,

        'num_bins': 601,
        'channels': 64,
        'use_resnet_v2': True,
        'output_init_scale': 0.,
        'discount_factor': 0.997 ** 4,
        'mcts_c1': 1.25,
        'mcts_c2': 19625,
        'alpha': 0.3,
        'exploration_prob': 0.25,
        'temperature_scheduling': 'staircase',
        'q_normalize_epsilon': 0.01,
        'child_select_epsilon': 1E-6,
        'num_simulations': 50,

        'replay_min_size': 2_000,
        'replay_max_size': 100_000,
        'batch_size': 256,

        'value_coef': 0.25,
        'policy_coef': 1.,
        'max_grad_norm': 5.,
        'learning_rate': 7E-4,
        'warmup_steps': 1_000,
        'learning_rate_decay': 0.1,
        'weight_decay': 1E-4,
        'target_update_interval': 200,

        'evaluate_episodes': 32,
        'log_interval': 4_000,
        'total_frames': 100_000,
    }
    log_filename = os.path.basename(__file__).split('.')[0]
    # analysis = tune.run(
    #     Experiment,
    #     name=log_filename,
    #     config=config,
    #     stop={
    #         'num_updates': 120_000,
    #     },
    #     resources_per_trial={
    #         'gpu': 1,
    #     },
    # )
    experiment = Experiment(config)
    # A simple loop, you might want to add more sophisticated logging/stopping logic
    # Matching the loop condition from the modified muzero.py for consistency
    num_iterations = config.get('num_updates', 120_000) // config['log_interval'] 
    if 'total_frames' in config and config['log_interval'] > 0 : # use total_frames if num_updates not in config
        num_iterations = config['total_frames'] // config['log_interval']
    
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}, ",end='')
        log = experiment.step()
        print(f"Log: {log}")
        if log.get('num_updates', 0) >= config.get('num_updates', 120_000) and 'num_updates' in config:
             print(f"Stopping early based on num_updates: {log.get('num_updates')}")
             break

    experiment.cleanup()
