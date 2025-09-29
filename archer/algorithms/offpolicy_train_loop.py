from archer.environment import batch_interact_environment
from archer.data import DummyDataset,  ReplayBuffer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from archer.algorithms.archer import ArcherTrainer
from archer.algorithms.online_filteredbc import BCTrainer
import wandb
import threading
import os
import torch
import time
def offpolicy_train_loop(env,\
                eval_env,\
                agent,\
                tokenizer,\
                accelerator,\
                warmup_iter: int = 20,
                rollout_size: int = 50,\
                eval_size: int = 1,
                batch_size: int = 2,
                capacity: int = 500000,
                iterations: int = 10,\
                epochs:int = 3, \
                grad_accum_steps: int = 1,\
                env_idx:int = None,\
                do_sample: bool = False,\
                temperature: float = 2.0,\
                critic_lr: float= 1e-3,\
                lm_lr: float = 1e-5,\
                gamma: float = 0.9,
                tau: float = 0.1,
                use_wandb: bool = False,
                env_load_path: str = '',
                actor_epochs: int = 3,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                save_freq: int = 25,
                eval_freq: int = 25,
                agent_type: str = "archer",
                decode_f: callable = lambda x: x,
                **kwargs):
    
    # Set the micro-batch size for DeepSpeed before preparing anything.
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = batch_size

    # Create optimizers BEFORE preparing anything
    lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lm_lr)

    # ONLY prepare the main model with DeepSpeed (the largest one that benefits from it)
    agent.model, lm_optimizer = accelerator.prepare(agent.model, lm_optimizer)
    
    # Use optimized DDP for critics (even if they're large)
    print(f"Critic model size: {sum(p.numel() for p in agent.critic.parameters()):,} parameters")
    print("Using optimized DDP for critics")
    
    if torch.cuda.is_available():
        agent.critic = agent.critic.cuda()
        agent.target_critic = agent.target_critic.cuda()
        
        if accelerator.num_processes > 1:
            # Optimized DDP settings for large models
            agent.critic = torch.nn.parallel.DistributedDataParallel(
                agent.critic,
                device_ids=[accelerator.state.local_process_index],
                output_device=accelerator.state.local_process_index,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=critic_lr)

        else:
            critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=critic_lr)

    # No separate critic accelerator
    agent.critic_accelerator = None
    
    # Update the agent's accelerator reference
    agent.accelerator = accelerator

    if agent_type.lower() == "chai" or agent_type.lower() == "archer"\
        or agent_type.lower() == "archer_llm":
        trainer = ArcherTrainer(agent=agent,\
                            accelerator=accelerator,\
                                tokenizer=tokenizer,\
                                critic_optimizer=critic_optimizer,\
                                lm_optimizer=lm_optimizer,\
                                gamma = gamma,\
                                tau = tau,\
                                epochs = epochs,\
                                actor_epochs = actor_epochs,
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm)
    elif agent_type.lower() == "online_filteredbc":
        trainer = BCTrainer(agent=agent,\
                                tokenizer=tokenizer,\
                                accelerator=accelerator,
                                lm_lr = lm_lr,\
                                epochs = actor_epochs,\
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm)
    replay_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
    all_trajectories = []
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(save_path, 'trainer.pt')):
            # print("Not using existing checkpoint")
            print("Loading from checkpoint")
            trainer.load(os.path.join(save_path, 'trainer.pt'))
            all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
            replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        else:
            print("Creating new checkpoint directory")
            os.makedirs(save_path, exist_ok=True)

    #main training loop
    print(">>>start iterations")
    for i in tqdm(range(iterations)):
        # Distribute rollout generation across all processes
        num_local_rollouts = rollout_size // accelerator.num_processes
        
        # Each process generates its share of trajectories
        local_trajectories = batch_interact_environment(agent=agent,
                                                  tokenizer=tokenizer,
                                                  env=env,
                                                  num_trajectories=num_local_rollouts,
                                                  env_idx=env_idx,
                                                  use_tqdm=False,
                                                  decode_f=decode_f)
        
        # Note: This can still be a bottleneck if trajectories are large.
        # A fully distributed replay buffer is the ideal solution.
        gathered_trajectories = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(gathered_trajectories, local_trajectories)
        
        # All processes now have the trajectories from all other processes.
        # Flatten the list of lists on all processes.
        # All processes now have the trajectories from all other processes.
        # Flatten the list of lists on all processes.
        trajectories = [item for sublist in gathered_trajectories for item in sublist]
        data = sum(trajectories, [])

        # All processes need to update their replay buffer and all_trajectories
        for t in data:
            replay_buffer.insert(**t)
        all_trajectories += trajectories

        if accelerator.is_main_process:
            info = {"rollout.mean": np.mean([d[0]["trajectory_reward"] for d in trajectories]),\
                    "rollout.max": np.max([d[0]["trajectory_reward"] for d in trajectories]),\
                    "rollout.min": np.min([d[0]["trajectory_reward"] for d in trajectories])}
            
            # Log rollout metrics immediately for debugging
            if use_wandb:
                rollout_metrics = {
                    "iteration": i,
                    "rollout.mean": info["rollout.mean"],
                    "rollout.max": info["rollout.max"], 
                    "rollout.min": info["rollout.min"],
                    "rollout.num_trajectories": len(trajectories)
                }
                wandb.log(rollout_metrics, step=i)
                print(f"Iteration {i}: Rollout mean reward = {info['rollout.mean']:.4f}")
            
            if (i+1) % eval_freq == 0:
                old_sample = agent.do_sample
                agent.do_sample = False
                eval_trajectories =  batch_interact_environment(agent = agent,\
                                                    tokenizer= tokenizer,\
                                                    env = eval_env,\
                                                    num_trajectories=  max(eval_size, eval_env.bsize),\
                                                    env_idx = env_idx,
                                                    use_tqdm=False,
                                                    decode_f = decode_f)
                agent.do_sample = old_sample
                eval_metrics = {"eval_rollout.mean": np.mean([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                        "eval_rollout.max": np.max([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                        "eval_rollout.min": np.min([d[0]["trajectory_reward"] for d in eval_trajectories]),}
                info.update(eval_metrics)
                
                # Log eval metrics immediately
                if use_wandb:
                    wandb.log(eval_metrics, step=i)
                    print(f"Iteration {i}: Eval mean reward = {eval_metrics['eval_rollout.mean']:.4f}")
            
            buffer_metrics = {"rollout.reward.mean": np.mean([d["reward"] for d in data]),\
                    "rollout.reward.max": np.max([d["reward"] for d in data]),\
                    "rollout.reward.min": np.min([d["reward"] for d in data]),
                    "buffer.size": len(replay_buffer)}
            info.update(buffer_metrics)
            
            # Log buffer metrics immediately
            if use_wandb:
                wandb.log(buffer_metrics, step=i)
        else:
            # Non-main processes need empty info dict for later updates
            info = {}

        print("Training")
        if 'filtered' in agent_type.lower():
            filtered_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
            episode_rewards = [d[0]["trajectory_reward"] for d in all_trajectories]
            cutoff = np.quantile(episode_rewards, 1 - 0.1)
            print("Episode Reward Cutoff: ", cutoff)
            filtered_trajectories = list(filter(lambda x: x[0]["trajectory_reward"] >= cutoff, all_trajectories))
            filtered_data = sum(filtered_trajectories, [])
            for d in filtered_data:
                filtered_buffer.insert(**d)
            training_info = trainer.update(filtered_buffer, no_update_actor = (i < warmup_iter))
        else:
            training_info = trainer.update(replay_buffer, no_update_actor = (i < warmup_iter))
        
        # Gather training info from all processes
        gathered_info = [None] * accelerator.num_processes
        if accelerator.num_processes > 1:
            torch.distributed.all_gather_object(gathered_info, training_info)
        else:
            gathered_info = [training_info]

        if accelerator.is_main_process:
            # Aggregate info from all processes (simple mean for numeric values)
            aggregated_info = {}
            if gathered_info and gathered_info[0]:
                for key in gathered_info[0]:
                    if isinstance(gathered_info[0][key], (int, float, torch.Tensor)):
                        values = [d.get(key, 0) for d in gathered_info if d]
                        # Ensure values are numeric before averaging
                        if all(isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1) for v in values):
                            aggregated_info[key] = np.mean([v.item() if isinstance(v, torch.Tensor) else v for v in values])
            info.update(aggregated_info)
        
        # Log training metrics immediately
        if use_wandb and accelerator.is_main_process:
            training_metrics = {
                k: (float(v) if isinstance(v, torch.Tensor) else v)
                for k, v in training_info.items()
                if isinstance(v, (int, float, np.number, torch.Tensor))
            }
            wandb.log(training_metrics, step=i)
            print(f"Iteration {i}: Training losses logged - {len(training_metrics)} metrics")
        
        # TEMPORARILY DISABLE EXPENSIVE V-FUNCTION EVALUATION
        # This is likely the main cause of your 300-hour training time
        # Re-enable once you've confirmed the basic training speed is acceptable
        # if (i+1) % eval_freq == 0 and len(all_trajectories) > 10:
        #     print(">>>Evaluating V-functions on trajectories")
        #     try:
        #         trajectory_metrics = trainer.evaluate_v_functions_on_trajectories(
        #             all_trajectories, 
        #             n_trajectories=min(50, len(all_trajectories))
        #         )
        #         info.update(trajectory_metrics)
        #         # ... rest of evaluation code
        #     except Exception as e:
        #         print(f"Trajectory V-function evaluation failed: {e}")
        #         info.update({'trajectory_eval.status': 'failed', 'trajectory_eval.error': str(e)})
        
        # Final consolidated log (keeping the original)
        if use_wandb and accelerator.is_main_process:
            final_metrics = {
                k: (float(v) if isinstance(v, torch.Tensor) else v)
                for k, v in info.items()
                if isinstance(v, (int, float, np.number, torch.Tensor))
            }
            wandb.log(final_metrics, step=i)
            print(f"Iteration {i}: Training losses logged - {len(training_metrics)} metrics")
        
        # Add comprehensive V-function evaluation using trajectories
        if (i+1) % eval_freq == 0 and len(all_trajectories) > 10:
            print(">>>Evaluating V-functions on trajectories")
            try:
                trajectory_metrics = trainer.evaluate_v_functions_on_trajectories(
                    all_trajectories, 
                    n_trajectories=min(50, len(all_trajectories))
                )
                info.update(trajectory_metrics)
                
                # Log trajectory evaluation metrics immediately
                if use_wandb and accelerator.is_main_process:
                    traj_eval_metrics = {k: v for k, v in trajectory_metrics.items() if isinstance(v, (int, float, np.number))}
                    wandb.log(traj_eval_metrics, step=i)
                
                # Print key metrics for monitoring
                if 'trajectory_eval.v_min.pearson_corr' in trajectory_metrics:
                    print(f"V-function trajectory evaluation - "
                          f"V_min correlation: {trajectory_metrics['trajectory_eval.v_min.pearson_corr']:.4f}, "
                          f"MSE: {trajectory_metrics['trajectory_eval.v_min.mse']:.4f}, "
                          f"Explained variance: {trajectory_metrics['trajectory_eval.v_min.explained_variance']:.4f}")
                          
            except Exception as e:
                print(f"Trajectory V-function evaluation failed: {e}")
                info.update({'trajectory_eval.status': 'failed', 'trajectory_eval.error': str(e)})
        
        # Final consolidated log (keeping the original)
        if use_wandb and accelerator.is_main_process:
            final_metrics = {k: v for k, v in info.items() if isinstance(v, (int, float, np.number))}
            wandb.log(final_metrics, step=i)
            
        if (i+1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
    # return model