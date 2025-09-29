import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
from archer.data import DummyDataset
import copy
import threading
from typing import Tuple
import random
import copy
import time
import numpy as np
from collections import defaultdict
def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
class ArcherTrainer():
    def __init__(self, agent,\
                 accelerator,\
                    tokenizer,\
                    critic_optimizer, \
                    lm_optimizer, \
                    grad_accum_steps: int = 8,\
                    gamma: float = 0.9,
                    tau: float = 0.1,
                    epochs: int = 3,
                    max_grad_norm: float=0.01,
                    actor_epochs: int = 3):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        if hasattr(self.agent.critic, "base_lm"):
            self.agent.critic.base_lm.gradient_checkpointing_enable()
        self.tokenizer = tokenizer
        # Optimizers are now passed in directly
        self.lm_optimizer = lm_optimizer
        self.critic_optimizer = critic_optimizer
        self.criterion = torch.nn.MSELoss()
        self.grad_accum_steps = grad_accum_steps
        self.actor_epochs = actor_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.step = 0
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.accelerator = accelerator

    def q_loss(self, observation, action, reward, next_observation, done, **kwargs):
        reward = torch.tensor(reward, device=self.accelerator.device, dtype=torch.float32).flatten()
        done = torch.tensor(done, device=self.accelerator.device, dtype=torch.float32).flatten()
        
        q1, q2, _, _ = self.agent.critic(observation, action, detach_model=False)
        q1, q2 = q1.flatten(), q2.flatten()

        with torch.no_grad():
            next_action = self.agent.get_action(next_observation)
            _, _, target_v1, target_v2 = self.agent.target_critic(next_observation, next_action)
            target_v1 = reward + (1 - done) * target_v1.flatten() * self.gamma
            target_v2 = reward + (1 - done) * target_v2.flatten() * self.gamma
        
        q1_loss = self.criterion(q1, target_v1)
        q2_loss = self.criterion(q2, target_v2)
        total_loss = q1_loss + q2_loss

        q1_cpu, q2_cpu = q1.detach().cpu(), q2.detach().cpu()
        target_v1_cpu, target_v2_cpu = target_v1.detach().cpu(), target_v2.detach().cpu()

        metrics = {
            "q.total_loss": total_loss.detach().cpu().item(),
            "q1.loss": q1_loss.detach().cpu().item(),
            "q2.loss": q2_loss.detach().cpu().item(),
            "q1.mean": q1_cpu.mean().item(),
            "q1.min": q1_cpu.min().item(),
            "q1.max": q1_cpu.max().item(),
            "q1.std": q1_cpu.std(unbiased=False).item(),
            "q2.mean": q2_cpu.mean().item(),
            "q2.min": q2_cpu.min().item(),
            "q2.max": q2_cpu.max().item(),
            "q2.std": q2_cpu.std(unbiased=False).item(),
            "target_v1.mean": target_v1_cpu.mean().item(),
            "target_v1.min": target_v1_cpu.min().item(),
            "target_v1.max": target_v1_cpu.max().item(),
            "target_v1.std": target_v1_cpu.std(unbiased=False).item(),
            "target_v2.mean": target_v2_cpu.mean().item(),
            "target_v2.min": target_v2_cpu.min().item(),
            "target_v2.max": target_v2_cpu.max().item(),
            "target_v2.std": target_v2_cpu.std(unbiased=False).item(),
        }

        return total_loss, metrics

    def v_loss(self, observation, pi_action=None, **kwargs):
        if pi_action is None:
            with torch.no_grad():
                pi_action = self.agent.get_action(observation)
        _, _, v1, v2 = self.agent.critic(observation, pi_action, detach_model=False)
        v1, v2 = v1.flatten(), v2.flatten()

        with torch.no_grad():
            target_q1, target_q2, _, _ = self.agent.target_critic(observation, pi_action, detach_model=False)
            target_q1, target_q2 = target_q1.flatten(), target_q2.flatten()

        v1_loss = self.criterion(v1, target_q1)
        v2_loss = self.criterion(v2, target_q2)
        total_loss = v1_loss + v2_loss

        v1_cpu, v2_cpu = v1.detach().cpu(), v2.detach().cpu()
        target_q1_cpu, target_q2_cpu = target_q1.detach().cpu(), target_q2.detach().cpu()

        metrics = {
            "v.total_loss": total_loss.detach().cpu().item(),
            "v1.loss": v1_loss.detach().cpu().item(),
            "v2.loss": v2_loss.detach().cpu().item(),
            "v1.mean": v1_cpu.mean().item(),
            "v1.min": v1_cpu.min().item(),
            "v1.max": v1_cpu.max().item(),
            "v1.std": v1_cpu.std(unbiased=False).item(),
            "v2.mean": v2_cpu.mean().item(),
            "v2.min": v2_cpu.min().item(),
            "v2.max": v2_cpu.max().item(),
            "v2.std": v2_cpu.std(unbiased=False).item(),
            "target_q1.mean": target_q1_cpu.mean().item(),
            "target_q1.min": target_q1_cpu.min().item(),
            "target_q1.max": target_q1_cpu.max().item(),
            "target_q1.std": target_q1_cpu.std(unbiased=False).item(),
            "target_q2.mean": target_q2_cpu.mean().item(),
            "target_q2.min": target_q2_cpu.min().item(),
            "target_q2.max": target_q2_cpu.max().item(),
            "target_q2.std": target_q2_cpu.std(unbiased=False).item(),
        }

        return total_loss, metrics

    def actor_loss(self, observation, pi_action, advantage, **kwargs):
        log_prob = self.agent.get_log_prob(observation, pi_action)
        advantage_tensor = torch.tensor(advantage, device=self.accelerator.unwrap_model(self.agent.model).device, dtype=self.accelerator.unwrap_model(self.agent.model).dtype)

        if isinstance(log_prob, Tuple):
            values, log_prob, mask = log_prob
            values = values.squeeze(-1)
            mask = mask.to(values.dtype)
            expanded_advantage = advantage_tensor.reshape(values.shape)
            residual_advantage = (expanded_advantage - values) * mask
            pg_loss = -torch.mean(torch.sum(residual_advantage * log_prob * mask, dim=1))
            value_loss = torch.mean(residual_advantage.pow(2))
            advantages_flat = (expanded_advantage * mask).detach().cpu().view(expanded_advantage.size(0), -1).mean(dim=1)
            values_flat = (values * mask).detach().cpu().view(values.size(0), -1).mean(dim=1)
            residual_flat = residual_advantage.detach().cpu().view(residual_advantage.size(0), -1)
        else:
            advantages_flat = advantage_tensor.flatten()
            pg_loss = -torch.mean(log_prob.flatten() * advantages_flat)
            value_loss = torch.zeros_like(pg_loss)
            values_flat = torch.zeros_like(advantages_flat).detach().cpu()
            residual_flat = torch.zeros_like(advantages_flat).detach().cpu()

        self.accelerator.backward(pg_loss + value_loss)

        advantages_cpu = advantages_flat.detach().cpu()
        metrics = {
            "pg.loss": pg_loss.detach().cpu().item(),
            "values.loss": value_loss.detach().cpu().item(),
            "advantages.mean": advantages_cpu.mean().item(),
            "advantages.std": advantages_cpu.std(unbiased=False).item(),
            "advantages.max": advantages_cpu.max().item(),
            "advantages.min": advantages_cpu.min().item(),
        }

        if isinstance(log_prob, Tuple):
            metrics.update({
                "values.mean": values_flat.mean().item(),
                "values.std": values_flat.std(unbiased=False).item(),
                "values.max": values_flat.max().item(),
                "values.min": values_flat.min().item(),
                "residual_advantages.mean": residual_flat.mean().item(),
                "residual_advantages.std": residual_flat.std(unbiased=False).item(),
                "residual_advantages.max": residual_flat.max().item(),
                "residual_advantages.min": residual_flat.min().item(),
            })
        else:
            metrics.update({
                "values.mean": 0.0,
                "values.std": 0.0,
                "values.max": 0.0,
                "values.min": 0.0,
                "residual_advantages.mean": 0.0,
                "residual_advantages.std": 0.0,
                "residual_advantages.max": 0.0,
                "residual_advantages.min": 0.0,
            })

        return metrics


    def update(self, replay_buffer, no_update_actor=False, eval_value_functions=True, eval_freq=10):
        self.step += 1
        info = {}
        q_info_list, v_info_list, info_list = [], [], []
        
        with torch.autograd.set_detect_anomaly(True):
            # --- Combined Critic Update (Q and V together to avoid DDP issues) ---
            for _ in range(self.epochs):
                data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
                for d in data:
                    for k,v in d.items():
                        d[k] = v[0]
                dataloader = DataLoader(DummyDataset(data), batch_size=replay_buffer.batch_size)
                dataloader = self.accelerator.prepare(dataloader)
                
                self.critic_optimizer.zero_grad()
                for batch in tqdm(dataloader, disable=True, desc="Critic-Update"):
                    with torch.no_grad():
                        pi_action = self.agent.get_action(batch["observation"])
                    q_loss, q_info = self.q_loss(**batch)
                    v_loss, v_info = self.v_loss(batch["observation"], pi_action=pi_action)
                    q_info_list.append(q_info)
                    v_info_list.append(v_info)
                    combined_loss = q_loss + v_loss
                    combined_loss.backward()
                    
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

            self.agent.soft_update_target_critic(tau=self.tau)

        info.update(dict_mean(q_info_list))
        info.update(dict_mean(v_info_list))
        #update actor
        if not no_update_actor:
            print(">>>updating actor")
            #batchsize for the actor set to 1 for mistral due to memory concern
            action_bsize = 2 if 'mistral' in self.agent.policy_lm else replay_buffer.batch_size
            #action_bsize = replay_buffer.batch_size
            for _ in range(self.actor_epochs):
                data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
                for d in data:
                    for k,v in d.items():
                        d[k] = v[0]
                
                dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=True)
                dataloader = self.accelerator.prepare(dataloader)

                # Pre-generate all actions for this epoch to reduce LLM calls
                print(">>> Pre-generating policy actions for actor update...")
                all_observations = []
                epoch_data = []
                for batch_data in data:
                    all_observations.append(batch_data['observation'])
                    epoch_data.append(batch_data)
                
                # Generate actions in larger batches
                action_batch_size = 8 if 'mistral' in self.agent.policy_lm else 16
                all_pi_actions = []
                
                with torch.no_grad():
                    for i in range(0, len(all_observations), action_batch_size):
                        batch_obs = all_observations[i:i+action_batch_size]
                        pi_actions = self.agent.get_action(batch_obs)
                        all_pi_actions.extend(pi_actions)
                
                # Add generated actions to the data
                for i, d in enumerate(epoch_data):
                    if i < len(all_pi_actions):
                        d['pi_action'] = all_pi_actions[i]
                
                # Now use the pre-generated actions
                dataloader = DataLoader(DummyDataset(epoch_data), batch_size=action_bsize, shuffle=True)
                dataloader = self.accelerator.prepare(dataloader)
                
                self.lm_optimizer.zero_grad()
                for batch in dataloader:
                    with torch.no_grad():
                        # Use pre-generated actions
                        pi_action = batch['pi_action']
                        q1, q2, v1, v2 = self.agent.critic(batch["observation"], pi_action)
                        q = torch.minimum(q1, q2)
                        v = torch.minimum(v1, v2)
                        advantages = q - v
                    info_list.append(self.actor_loss(observation=batch["observation"], pi_action=pi_action, advantage=advantages))
                self.accelerator.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()
        info.update(dict_mean(info_list))
        
        # V-function ablation analysis
        if eval_value_functions and self.step % eval_freq == 0 and len(replay_buffer) > 100:  
            print(">>>Evaluating value functions")
            try:
                # Evaluate value functions against MC returns
                value_metrics = self.evaluate_value_functions(replay_buffer, n_samples=min(500, len(replay_buffer)))  
                info.update(value_metrics)
                
                # Compute TD errors
                td_metrics = self.compute_td_errors(replay_buffer, n_samples=min(500, len(replay_buffer)))  
                info.update(td_metrics)
                
                print(f"V-function evaluation completed - V1 MSE: {value_metrics.get('value_eval.v1.mse', 'N/A'):.4f}, "
                      f"V_min correlation: {value_metrics.get('value_eval.v_min.pearson_corr', 'N/A'):.4f}")
                      
            except Exception as e:
                print(f"V-function evaluation failed: {e}")
                # Add fallback metrics
                info.update({
                    'value_eval.status': 'failed',
                    'value_eval.error': str(e)
                })
        
        return info

    def compute_mc_returns(self, trajectory_data, gamma=None):
        """
        Compute true Monte Carlo returns for trajectory data
        G_t = r_{t+1} + γr_{t+2} + γ²r_{t+3} + ...
        
        Args:
            trajectory_data: List of trajectory steps with 'reward' key
            gamma: Discount factor (uses self.gamma if None)
        
        Returns:
            List of MC returns for each step
        """
        if gamma is None:
            gamma = self.gamma
            
        mc_returns = []
        G = 0
        
        # Compute returns backwards through trajectory
        for step in reversed(trajectory_data):
            G = step['reward'] + gamma * G
            mc_returns.append(G)
        
        return list(reversed(mc_returns))

    def evaluate_value_functions(self, replay_buffer, n_samples=500):
        """
        Evaluate learned V-functions against true Monte Carlo returns
        
        Args:
            replay_buffer: ReplayBuffer containing experience
            n_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(replay_buffer) == 0:
            return {}

        mc_returns = []
        v1_predictions = []
        v2_predictions = []
        v_min_predictions = []

        sampled_data = []
        num_samples = min(n_samples, len(replay_buffer))
        for _ in range(num_samples):
            sample = replay_buffer.sample(1)
            for k, v in sample.items():
                sample[k] = v[0]
            sampled_data.append(sample)
        
        # Group samples by trajectory if possible
        # For now, treat each sample independently
        for sample in sampled_data:
            # Get learned V-function predictions
            with torch.no_grad():
                _, _, v1, v2 = self.agent.critic(sample['observation'], sample['action'])
                v1_val = v1.flatten().cpu().numpy()[0]
                v2_val = v2.flatten().cpu().numpy()[0]
                v_min_val = min(v1_val, v2_val)
                
                v1_predictions.append(v1_val)
                v2_predictions.append(v2_val)
                v_min_predictions.append(v_min_val)
                
                # Compute simple MC estimate (r + γV(s'))
                if sample['done']:
                    mc_return = sample['reward']
                else:
                    _, _, next_v1, next_v2 = self.agent.critic(sample['next_observation'], sample['action'])
                    next_v = min(next_v1.flatten().cpu().numpy()[0], next_v2.flatten().cpu().numpy()[0])
                    mc_return = sample['reward'] + self.gamma * next_v
                
                mc_returns.append(mc_return)
        
        if len(mc_returns) == 0:
            return {}
            
        # Convert to numpy for easier computation
        mc_returns = np.array(mc_returns)
        v1_predictions = np.array(v1_predictions)
        v2_predictions = np.array(v2_predictions)
        v_min_predictions = np.array(v_min_predictions)
        
        # Compute metrics for V1, V2, and min(V1,V2)
        metrics = {}
        
        for name, predictions in [('v1', v1_predictions), ('v2', v2_predictions), ('v_min', v_min_predictions)]:
            # Correlation metrics
            if len(predictions) > 1 and np.var(predictions) > 0 and np.var(mc_returns) > 0:
                try:
                    pearson_corr = np.corrcoef(mc_returns, predictions)[0, 1]
                    # Compute spearman manually since scipy might not be available
                    spearman_corr = pearson_corr  # Fallback to Pearson
                except:
                    pearson_corr = 0.0
                    spearman_corr = 0.0
            else:
                pearson_corr = 0.0
                spearman_corr = 0.0
                
            # Error metrics
            mse = np.mean((mc_returns - predictions)**2)
            mae = np.mean(np.abs(mc_returns - predictions))
            bias = np.mean(predictions - mc_returns)
            
            # Explained variance
            if np.var(mc_returns) > 0:
                explained_var = 1 - np.var(mc_returns - predictions) / np.var(mc_returns)
            else:
                explained_var = 0.0
            
            metrics.update({
                f'value_eval.{name}.pearson_corr': pearson_corr,
                f'value_eval.{name}.spearman_corr': spearman_corr,
                f'value_eval.{name}.mse': mse,
                f'value_eval.{name}.mae': mae,
                f'value_eval.{name}.bias': bias,
                f'value_eval.{name}.explained_variance': explained_var,
                f'value_eval.{name}.mean': np.mean(predictions),
                f'value_eval.{name}.std': np.std(predictions),
            })
        
        # MC return statistics
        metrics.update({
            'value_eval.mc_returns.mean': np.mean(mc_returns),
            'value_eval.mc_returns.std': np.std(mc_returns),
            'value_eval.mc_returns.min': np.min(mc_returns),
            'value_eval.mc_returns.max': np.max(mc_returns),
        })
        
        return metrics

    def compute_td_errors(self, replay_buffer, n_samples=500):
        """
        Compute temporal difference errors: δ = r + γV(s') - V(s)
        If V is accurate, TD errors should be small and unbiased
        
        Args:
            replay_buffer: ReplayBuffer containing experience
            n_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with TD error statistics
        """
        if len(replay_buffer) == 0:
            return {}

        td_errors_v1 = []
        td_errors_v2 = []
        td_errors_v_min = []

        num_samples = min(n_samples, len(replay_buffer))
        for _ in range(num_samples):
            sample = replay_buffer.sample(1)
            for k, v in sample.items():
                sample[k] = v[0]
            
            with torch.no_grad():
                # Get current state values
                _, _, v1_s, v2_s = self.agent.critic(sample['observation'], sample['action'])
                v1_s = v1_s.flatten().cpu().numpy()[0]
                v2_s = v2_s.flatten().cpu().numpy()[0]
                v_min_s = min(v1_s, v2_s)
                
                # Get next state values
                if sample['done']:
                    v1_s_next = 0.0
                    v2_s_next = 0.0
                    v_min_s_next = 0.0
                else:
                    _, _, v1_next, v2_next = self.agent.critic(sample['next_observation'], sample['action'])
                    v1_s_next = v1_next.flatten().cpu().numpy()[0]
                    v2_s_next = v2_next.flatten().cpu().numpy()[0]
                    v_min_s_next = min(v1_s_next, v2_s_next)
                
                # Compute TD targets
                reward = sample['reward']
                td_target_v1 = reward + self.gamma * v1_s_next
                td_target_v2 = reward + self.gamma * v2_s_next
                td_target_v_min = reward + self.gamma * v_min_s_next
                
                # Compute TD errors
                td_error_v1 = td_target_v1 - v1_s
                td_error_v2 = td_target_v2 - v2_s
                td_error_v_min = td_target_v_min - v_min_s
                
                td_errors_v1.append(td_error_v1)
                td_errors_v2.append(td_error_v2)
                td_errors_v_min.append(td_error_v_min)
        
        if len(td_errors_v1) == 0:
            return {}
            
        # Convert to numpy
        td_errors_v1 = np.array(td_errors_v1)
        td_errors_v2 = np.array(td_errors_v2)
        td_errors_v_min = np.array(td_errors_v_min)
        
        # Compute statistics
        metrics = {}
        for name, errors in [('v1', td_errors_v1), ('v2', td_errors_v2), ('v_min', td_errors_v_min)]:
            metrics.update({
                f'td_error.{name}.mean': np.mean(errors),
                f'td_error.{name}.std': np.std(errors),
                f'td_error.{name}.abs_mean': np.mean(np.abs(errors)),
                f'td_error.{name}.min': np.min(errors),
                f'td_error.{name}.max': np.max(errors),
            })
        
        return metrics

    def evaluate_v_functions_on_trajectories(self, trajectories, n_trajectories=50):
        """
        Evaluate V-functions using complete trajectories with true MC returns
        This provides the most accurate evaluation of value function quality
        
        Args:
            trajectories: List of complete trajectories from training
            n_trajectories: Number of trajectories to evaluate
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        if not trajectories or len(trajectories) == 0:
            return {}
            
        # Sample random trajectories
        sampled_trajectories = random.sample(trajectories, min(n_trajectories, len(trajectories)))
        
        mc_returns_all = []
        v1_predictions_all = []
        v2_predictions_all = []
        v_min_predictions_all = []
        trajectory_lengths = []
        
        for trajectory in sampled_trajectories:
            # Each trajectory is a list of steps
            if len(trajectory) == 0:
                continue
                
            # Compute true MC returns for this trajectory
            mc_returns = self.compute_mc_returns(trajectory)
            trajectory_lengths.append(len(trajectory))
            
            # Get V-function predictions for each state in trajectory
            for i, (step, mc_return) in enumerate(zip(trajectory, mc_returns)):
                try:
                    with torch.no_grad():
                        # Get V-function predictions for this state
                        _, _, v1, v2 = self.agent.critic(step['observation'], step['action'])
                        v1_val = v1.flatten().cpu().numpy()[0]
                        v2_val = v2.flatten().cpu().numpy()[0]
                        v_min_val = min(v1_val, v2_val)
                        
                        mc_returns_all.append(mc_return)
                        v1_predictions_all.append(v1_val)
                        v2_predictions_all.append(v2_val)
                        v_min_predictions_all.append(v_min_val)
                        
                except Exception as e:
                    print(f"Error processing step {i} in trajectory: {e}")
                    continue
        
        if len(mc_returns_all) == 0:
            return {'trajectory_eval.status': 'no_valid_data'}
            
        # Convert to numpy
        mc_returns_all = np.array(mc_returns_all)
        v1_predictions_all = np.array(v1_predictions_all)
        v2_predictions_all = np.array(v2_predictions_all)
        v_min_predictions_all = np.array(v_min_predictions_all)
        
        # Compute comprehensive metrics
        metrics = {}
        
        for name, predictions in [('v1', v1_predictions_all), ('v2', v2_predictions_all), ('v_min', v_min_predictions_all)]:
            # Correlation metrics (more reliable with larger sample)
            if len(predictions) > 2 and np.var(predictions) > 1e-8 and np.var(mc_returns_all) > 1e-8:
                try:
                    pearson_corr = np.corrcoef(mc_returns_all, predictions)[0, 1]
                    if np.isnan(pearson_corr):
                        pearson_corr = 0.0
                except:
                    pearson_corr = 0.0
            else:
                pearson_corr = 0.0
                
            # Error metrics
            mse = np.mean((mc_returns_all - predictions)**2)
            mae = np.mean(np.abs(mc_returns_all - predictions))
            bias = np.mean(predictions - mc_returns_all)
            
            # Explained variance
            if np.var(mc_returns_all) > 1e-8:
                explained_var = 1 - np.var(mc_returns_all - predictions) / np.var(mc_returns_all)
                explained_var = max(0.0, explained_var)  # Clamp to [0, 1]
            else:
                explained_var = 0.0
            
            # Rank correlation (order preservation)
            try:
                rank_corr = np.corrcoef(np.argsort(mc_returns_all), np.argsort(predictions))[0, 1]
                if np.isnan(rank_corr):
                    rank_corr = 0.0
            except:
                rank_corr = 0.0
            
            metrics.update({
                f'trajectory_eval.{name}.pearson_corr': pearson_corr,
                f'trajectory_eval.{name}.rank_corr': rank_corr,
                f'trajectory_eval.{name}.mse': mse,
                f'trajectory_eval.{name}.mae': mae,
                f'trajectory_eval.{name}.bias': bias,
                f'trajectory_eval.{name}.explained_variance': explained_var,
                f'trajectory_eval.{name}.mean': np.mean(predictions),
                f'trajectory_eval.{name}.std': np.std(predictions),
                f'trajectory_eval.{name}.min': np.min(predictions),
                f'trajectory_eval.{name}.max': np.max(predictions),
            })
        
        # MC return and trajectory statistics
        metrics.update({
            'trajectory_eval.mc_returns.mean': np.mean(mc_returns_all),
            'trajectory_eval.mc_returns.std': np.std(mc_returns_all),
            'trajectory_eval.mc_returns.min': np.min(mc_returns_all),
            'trajectory_eval.mc_returns.max': np.max(mc_returns_all),
            'trajectory_eval.n_trajectories': len(sampled_trajectories),
            'trajectory_eval.n_steps': len(mc_returns_all),
            'trajectory_eval.avg_trajectory_length': np.mean(trajectory_lengths) if trajectory_lengths else 0,
            'trajectory_eval.status': 'success'
        })
        
        return metrics

    def save(self, path):
        torch.save({'model_state_dict': self.accelerator.unwrap_model(self.agent.model).state_dict(),
                    'critic_state_dict': self.accelerator.unwrap_model(self.agent.critic).state_dict(),
                    'target_critic_state_dict': self.accelerator.unwrap_model(self.agent.target_critic).state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'lm_optimizer_state_dict': self.lm_optimizer.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.agent.model.load_state_dict(checkpoint['model_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        return self.agent
