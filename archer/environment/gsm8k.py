import torch
import re
import json
from typing import Optional, List, Dict, Tuple
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

def make_step_rewards(logits, token_masks):
    """Extract step-level rewards from PRM logits."""
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

class GSM8KEnv:
    """GSM8K math reasoning environment with episode-based PRM evaluation."""
    
    def __init__(self, max_steps: int = 10, step_penalty: float = -0.05, device="auto", 
                 dataset_path: str = "dataset/gsm8k_solutions_gpt4o.jsonl"):
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.device = device
        
        # Load GSM8K problems from dataset file (like TwentyQuestions word list)
        self.problems = self._load_problems(dataset_path)
        
        # Load Qwen2.5-Math-PRM-7B
        model_name = "Qwen/Qwen2.5-Math-PRM-7B"
        self.prm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.prm_model = AutoModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        
        # Step separator token
        self.step_sep_id = self.prm_tokenizer.encode("<extra_0>")[0]
        
        self.reset()
    
    def _load_problems(self, dataset_path: str) -> List[Dict]:
        """Load problems from JSONL file (like TwentyQuestions loads word list)."""
        problems = []
        try:
            with open(dataset_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    problems.append({
                        "question": data["question"],
                        "answer": data["answer"],
                        "final_solution": data.get("final_solution", "")
                    })
        except FileNotFoundError:
            print(f"Warning: Dataset file {dataset_path} not found. Using empty problem list.")
            problems = []
        return problems
    
    def reset(self, problem_idx: Optional[int] = None):
        """Reset environment with a problem from dataset (like TwentyQuestions reset)."""
        if self.problems and problem_idx is not None:
            # Use specific problem by index
            problem_data = self.problems[problem_idx]
        elif self.problems:
            # Random problem from dataset (like TwentyQuestions random word)
            import random
            problem_data = random.choice(self.problems)
        else:
            # Fallback if no dataset
            problem_data = {"question": "No problems loaded", "answer": "0"}
        
        self.problem = problem_data["question"]
        self.ground_truth_answer = problem_data["answer"]
        self.reasoning_steps = []
        self.step_count = 0
        self.done = False
        
        # System prompt for math reasoning
        self.system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        
        # Initial observation is the problem with system prompt
        return f"Problem: {self.problem}\nSolution:"
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Take a reasoning step. Reward computation deferred to episode evaluation."""
        if self.done:
            return self.get_observation(), 0.0, True, {}
        
        self.reasoning_steps.append(action.strip())
        self.step_count += 1
        
        # Check if this step contains a final answer
        final_answer = self.extract_final_answer(action)
        
        if final_answer is not None:
            self.done = True
            return self.get_observation(), 0.0, True, {"final_answer": final_answer}
        
        elif self.step_count >= self.max_steps:
            self.done = True
            return self.get_observation(), 0.0, True, {"timeout": True}
        
        else:
            # Continue episode - no reward computed yet
            return self.get_observation(), 0.0, False, {}
    
    def get_observation(self) -> str:
        """Get current state as text observation."""
        obs = f"Problem: {self.problem}\nSolution:\n"
        for i, step in enumerate(self.reasoning_steps, 1):
            obs += f"{step}\n"
        return obs
    
    def extract_final_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from \\boxed{} format or other patterns."""
        # First try to extract from \\boxed{}
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        match = re.search(boxed_pattern, text)
        if match:
            try:
                # Clean the answer string and convert to float
                answer_str = match.group(1).strip()
                # Remove dollar signs and other formatting
                answer_str = re.sub(r'[^\d.-]', '', answer_str)
                return float(answer_str)
            except ValueError:
                pass
        
        # Fallback patterns
        patterns = [
            r"(?:answer is|therefore,?|final answer:?)\s*\$?(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:is the answer|is the final answer)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def evaluate_episode(self, episode_steps: List[Tuple[str, str]]) -> List[float]:
        """Evaluate complete episode and return step-wise rewards."""
        if not episode_steps:
            return []
        
        # Extract just the reasoning steps
        reasoning_steps = [action for _, action in episode_steps]
        
        # Format conversation for PRM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.problem},
            {"role": "assistant", "content": "<extra_0>".join(reasoning_steps) + "<extra_0>"},
        ]
        
        conversation_str = self.prm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        input_ids = self.prm_tokenizer.encode(
            conversation_str,
            return_tensors="pt",
        ).to(self.prm_model.device)
        
        with torch.no_grad():
            outputs = self.prm_model(input_ids=input_ids)
            
            # Get step-level rewards
            token_masks = (input_ids == self.step_sep_id)
            step_rewards = make_step_rewards(outputs[0], token_masks)
            
            if step_rewards and step_rewards[0]:
                prm_scores = step_rewards[0]
                
                # Convert PRM scores to rewards
                rewards = []
                for i, score in enumerate(prm_scores):
                    # Scale from [0,1] to [-1,1] and add step penalty
                    step_reward = (score - 0.5) * 2.0 + self.step_penalty
                    
                    # Check if this is the final step with an answer
                    if i == len(prm_scores) - 1:
                        final_answer = self.extract_final_answer(reasoning_steps[i])
                        if final_answer is not None:
                            terminal_reward = self.get_terminal_reward(final_answer)
                            step_reward += terminal_reward
                    
                    rewards.append(step_reward)
                
                return rewards
            else:
                # Fallback: give small negative rewards
                return [self.step_penalty] * len(reasoning_steps)
    
    def get_terminal_reward(self, predicted_answer: float) -> float:
        """Calculate terminal reward based on answer correctness."""
        try:
            ground_truth = float(self.ground_truth_answer)
            if abs(predicted_answer - ground_truth) < 1e-6:
                return 2.0  # Strong positive reward for correct answer
            else:
                return -1.0  # Negative reward for incorrect answer
        except:
            return -1.0  # Invalid answer format

class BatchedGSM8KEnv:
    """Batched version for parallel processing with episode evaluation."""
    
    def __init__(self, batch_size: int, max_steps: int = 10, step_penalty: float = -0.05, 
                 device="auto", dataset_path: str = "dataset/gsm8k_solutions_gpt4o.jsonl"):
        self.envs = [GSM8KEnv(max_steps, step_penalty, device, dataset_path) for _ in range(batch_size)]
        self.batch_size = batch_size
        self.bsize = batch_size  # For compatibility with training loop
    
    def reset(self, problem_indices: Optional[List[int]] = None) -> List[str]:
        """Reset all environments with problems from dataset."""
        if problem_indices is None:
            return [env.reset() for env in self.envs]  # Random problems
        else:
            return [env.reset(problem_idx=idx) for env, idx in zip(self.envs, problem_indices)]
    
    def step(self, actions: List[str]) -> Tuple[List[str], List[float], List[bool], List[Dict]]:
        """Take steps in all environments."""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        
        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)
    
    def interact_single_trajectory(self, agent, tokenizer, env_idx=0, decode_f=lambda x: x):
        """Collect a single complete trajectory with episode-based evaluation."""
        env = self.envs[env_idx]
        episode_steps = []
        trajectory = []
        
        # Reset environment
        obs = env.reset()
        done = False
        
        # Collect complete episode
        while not done:
            # Get action from agent
            action = agent.get_action([obs])[0]
            action = decode_f(action)
            
            # Store step
            episode_steps.append((obs, action))
            
            # Take step in environment (no reward computed yet)
            next_obs, _, done, info = env.step(action)
            obs = next_obs
        
        # Evaluate complete episode with PRM
        step_rewards = env.evaluate_episode(episode_steps)
        
        # Convert to trajectory format expected by training loop
        trajectory_reward = sum(step_rewards)
        
        for i, ((obs, action), reward) in enumerate(zip(episode_steps, step_rewards)):
            next_obs = episode_steps[i+1][0] if i+1 < len(episode_steps) else obs
            is_done = (i == len(episode_steps) - 1)
            
            step_data = {
                "observation": obs,
                "action": action,
                "reward": reward,
                "next_observation": next_obs,
                "done": is_done,
                "mc_return": sum(step_rewards[i:])  # Monte Carlo return from this step
            }
            trajectory.append(step_data)
        
        # Add trajectory-level info
        trajectory_info = {
            "trajectory_reward": trajectory_reward,
            "episode_length": len(episode_steps),
            "final_info": info
        }
        
        return [trajectory_info] + trajectory