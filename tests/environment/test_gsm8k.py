import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from archer.environment.gsm8k import GSM8KEnv, BatchedGSM8KEnv, make_step_rewards

# Mock GSM8K data points - updated to match dataset format
MOCK_GSM8K_PROBLEMS = [
    {
        "question": "Sue lives in a fun neighborhood. One weekend, the neighbors decided to play a prank on Sue. On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard. On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard. Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?",
        "answer": "24",
        "final_solution": "First, let's count the initial flamingos: 18 pink flamingos on Friday..."
    },
    {
        "question": "A baker made 10 dozen cookies. She sold 8 dozen cookies and gave away 1 dozen cookies. How many cookies does she have left?",
        "answer": "12",
        "final_solution": "The baker made 10 dozen = 10 × 12 = 120 cookies..."
    },
    {
        "question": "John has 5 apples. He gives 2 apples to his friend. How many apples does John have left?",
        "answer": "3",
        "final_solution": "John starts with 5 apples. He gives away 2 apples..."
    }
]

# Mock reasoning steps
MOCK_REASONING_STEPS = {
    0: [
        "First, let's count the initial flamingos: 18 pink flamingos on Friday.",
        "On Saturday, they took 1/3 of 18 = 6 flamingos, painted them white, and put them back.",
        "So we have 18 - 6 = 12 pink flamingos and 6 white flamingos.",
        "On Sunday, they added 18 more pink flamingos: 12 + 18 = 30 pink flamingos.",
        "The difference is 30 - 6 = 24 more pink flamingos. \\boxed{24}"
    ],
    1: [
        "The baker made 10 dozen = 10 × 12 = 120 cookies.",
        "She sold 8 dozen = 8 × 12 = 96 cookies.",
        "She gave away 1 dozen = 1 × 12 = 12 cookies.",
        "Total cookies used: 96 + 12 = 108 cookies.",
        "Cookies left: 120 - 108 = 12 cookies. \\boxed{12}"
    ],
    2: [
        "John starts with 5 apples.",
        "He gives away 2 apples to his friend.",
        "Apples left: 5 - 2 = 3 apples. \\boxed{3}"
    ]
}

class TestMakeStepRewards:
    """Test the make_step_rewards function."""
    
    def test_make_step_rewards_basic(self):
        """Test basic functionality of make_step_rewards."""
        # Mock logits (batch_size=1, seq_len=4, num_labels=2)
        logits = torch.tensor([[[1.0, 2.0], [0.5, 1.5], [2.0, 1.0], [1.5, 0.5]]])
        
        # Mock token masks (positions where step separator appears)
        token_masks = torch.tensor([[True, False, True, True]])
        
        rewards = make_step_rewards(logits, token_masks)
        
        assert len(rewards) == 1  # One batch
        assert len(rewards[0]) == 3  # Three step separators
        assert all(0.0 <= r <= 1.0 for r in rewards[0])  # Probabilities should be in [0,1]
    
    def test_make_step_rewards_empty(self):
        """Test make_step_rewards with no step separators."""
        logits = torch.tensor([[[1.0, 2.0], [0.5, 1.5]]])
        token_masks = torch.tensor([[False, False]])
        
        rewards = make_step_rewards(logits, token_masks)
        
        assert len(rewards) == 1
        assert len(rewards[0]) == 0


class TestGSM8KEnv:
    """Test the GSM8KEnv class."""
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""
        with patch('archer.environment.gsm8k.AutoTokenizer') as mock_tokenizer, \
             patch('archer.environment.gsm8k.AutoModel') as mock_model, \
             patch('archer.environment.gsm8k.GSM8KEnv._load_problems') as mock_load_problems:
            
            # Mock the dataset loading
            mock_load_problems.return_value = MOCK_GSM8K_PROBLEMS
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.encode.return_value = [12345]
            mock_tokenizer_instance.apply_chat_template.return_value = "mock conversation"
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock model with proper device attribute
            mock_model_instance = Mock()
            mock_model_instance.device = torch.device("cpu")  # Real device object
            mock_model_instance.eval.return_value = mock_model_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            
            env = GSM8KEnv(max_steps=5, step_penalty=-0.1, device="cpu")
            
            # Override the tokenizer encode method to return proper tensor
            def mock_encode(*args, **kwargs):
                if kwargs.get('return_tensors') == 'pt':
                    return torch.tensor([[1, 2, 12345, 3, 12345, 4]])
                return [1, 2, 12345, 3, 12345, 4]
            
            env.prm_tokenizer.encode = mock_encode
            
            return env
    
    def test_reset_with_index(self, mock_env):
        """Test environment reset with specific problem index."""
        obs = mock_env.reset(problem_idx=0)
        
        assert mock_env.problem == MOCK_GSM8K_PROBLEMS[0]["question"]
        assert mock_env.ground_truth_answer == MOCK_GSM8K_PROBLEMS[0]["answer"]
        assert mock_env.reasoning_steps == []
        assert mock_env.step_count == 0
        assert not mock_env.done
        assert f"Problem: {MOCK_GSM8K_PROBLEMS[0]['question']}" in obs
        assert "Solution:" in obs
    
    def test_reset_random(self, mock_env):
        """Test environment reset without index (random problem)."""
        obs = mock_env.reset()
        
        # Should select one of the mock problems
        assert mock_env.problem in [p["question"] for p in MOCK_GSM8K_PROBLEMS]
        assert mock_env.ground_truth_answer in [p["answer"] for p in MOCK_GSM8K_PROBLEMS]
        assert mock_env.reasoning_steps == []
        assert mock_env.step_count == 0
        assert not mock_env.done
        assert "Problem:" in obs
        assert "Solution:" in obs
    
    def test_step_intermediate(self, mock_env):
        """Test taking intermediate steps."""
        mock_env.reset(problem_idx=0)
        
        action = "Let's start by counting the initial flamingos."
        obs, reward, done, info = mock_env.step(action)
        
        assert action.strip() in mock_env.reasoning_steps
        assert mock_env.step_count == 1
        assert reward == 0.0  # No immediate reward
        assert not done
        assert action in obs
    
    def test_step_final_answer(self, mock_env):
        """Test step with final answer."""
        mock_env.reset(problem_idx=0)
        
        action = "Therefore, the answer is \\boxed{24}."  # Updated answer
        obs, reward, done, info = mock_env.step(action)
        
        assert done
        assert "final_answer" in info
        assert info["final_answer"] == 24.0  # Updated expected answer
    
    def test_step_max_steps(self, mock_env):
        """Test reaching max steps."""
        mock_env.reset(problem_idx=0)
        
        # Take max_steps
        for i in range(mock_env.max_steps):
            obs, reward, done, info = mock_env.step(f"Step {i+1}")
        
        assert done
        assert "timeout" in info
    
    def test_extract_final_answer_boxed(self, mock_env):
        """Test extracting answer from \\boxed{} format."""
        answer = mock_env.extract_final_answer("The answer is \\boxed{42}.")
        assert answer == 42.0
        
        answer = mock_env.extract_final_answer("\\boxed{3.14}")
        assert answer == 3.14
        
        answer = mock_env.extract_final_answer("No answer here.")
        assert answer is None
    
    def test_extract_final_answer_fallback(self, mock_env):
        """Test extracting answer from fallback patterns."""
        answer = mock_env.extract_final_answer("The answer is 25.")
        assert answer == 25.0
        
        answer = mock_env.extract_final_answer("Therefore, 15 is the final answer.")
        assert answer == 15.0
    
    def test_get_terminal_reward(self, mock_env):
        """Test terminal reward calculation."""
        mock_env.ground_truth_answer = "24"  # Updated to match corrected answer
        
        # Correct answer
        reward = mock_env.get_terminal_reward(24.0)  # Updated
        assert reward == 2.0
        
        # Incorrect answer
        reward = mock_env.get_terminal_reward(25.0)
        assert reward == -1.0
        
        # Invalid ground truth
        mock_env.ground_truth_answer = "invalid"
        reward = mock_env.get_terminal_reward(24.0)  # Updated
        assert reward == -1.0
    
    @patch('archer.environment.gsm8k.make_step_rewards')
    def test_evaluate_episode(self, mock_make_step_rewards, mock_env):
        """Test episode evaluation."""
        mock_env.reset(problem_idx=1)  # Use baker problem
        
        # Mock PRM outputs - need to return a tuple/object with proper structure
        mock_outputs = [torch.randn(1, 10, 2)]  # Mock logits
        mock_env.prm_model.return_value = mock_outputs
        mock_make_step_rewards.return_value = [[0.8, 0.9, 0.7, 1.0]]  # Mock step rewards
        
        episode_steps = [
            ("obs1", "The baker made 10 dozen = 120 cookies."),
            ("obs2", "She sold 8 dozen = 96 cookies."),
            ("obs3", "She gave away 1 dozen = 12 cookies."),
            ("obs4", "Cookies left: 120 - 108 = 12. \\boxed{12}")
        ]
        
        rewards = mock_env.evaluate_episode(episode_steps)
        
        assert len(rewards) == 4
        # Check that rewards are scaled and have step penalty
        for i, reward in enumerate(rewards[:-1]):
            if i == 0:
                expected = (0.8 - 0.5) * 2.0 + mock_env.step_penalty  # For first reward
                assert abs(reward - expected) < 0.01
        
        # Last step should have terminal reward (correct answer: 12)
        last_reward = rewards[-1]
        expected_last = (1.0 - 0.5) * 2.0 + mock_env.step_penalty + 2.0  # Correct answer
        assert abs(last_reward - expected_last) < 0.01


class TestBatchedGSM8KEnv:
    """Test the BatchedGSM8KEnv class."""
    
    @pytest.fixture
    def mock_batched_env(self):
        """Create a mock batched environment."""
        with patch('archer.environment.gsm8k.GSM8KEnv') as mock_env_class:
            # Create separate Mock objects for each environment
            mock_env_class.side_effect = [Mock() for _ in range(2)]
            batched_env = BatchedGSM8KEnv(batch_size=2, max_steps=3)
            return batched_env
    
    def test_init(self, mock_batched_env):
        """Test batched environment initialization."""
        assert mock_batched_env.batch_size == 2
        assert mock_batched_env.bsize == 2
        assert len(mock_batched_env.envs) == 2
    
    def test_reset_with_indices(self, mock_batched_env):
        """Test batched reset with specific problem indices."""
        problem_indices = [0, 1]
        
        # Mock individual env reset
        for env in mock_batched_env.envs:
            env.reset.return_value = "mock_obs"
        
        observations = mock_batched_env.reset(problem_indices)
        
        assert len(observations) == 2
        assert all(obs == "mock_obs" for obs in observations)
        
        # Check that reset was called with correct indices using keyword arguments
        mock_batched_env.envs[0].reset.assert_called_with(problem_idx=0)
        mock_batched_env.envs[1].reset.assert_called_with(problem_idx=1)
    
    def test_reset_random(self, mock_batched_env):
        """Test batched reset without indices (random problems)."""
        # Mock individual env reset
        for env in mock_batched_env.envs:
            env.reset.return_value = "mock_obs"
        
        observations = mock_batched_env.reset()
        
        assert len(observations) == 2
        assert all(obs == "mock_obs" for obs in observations)
        
        # Check that reset was called without arguments
        for env in mock_batched_env.envs:
            env.reset.assert_called_with()
    
    def test_step(self, mock_batched_env):
        """Test batched step."""
        actions = ["action1", "action2"]
        
        # Mock individual env step
        for env in mock_batched_env.envs:
            env.step.return_value = ("obs", 0.0, False, {})
        
        observations, rewards, dones, infos = mock_batched_env.step(actions)
        
        assert len(observations) == 2
        assert len(rewards) == 2
        assert len(dones) == 2
        assert len(infos) == 2
    
    def test_interact_single_trajectory(self, mock_batched_env):
        """Test single trajectory interaction."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.get_action.return_value = ["mock_action"]
        
        # Mock environment
        mock_env = mock_batched_env.envs[0]
        mock_env.reset.return_value = "initial_obs"
        mock_env.step.side_effect = [
            ("obs1", 0.0, False, {}),
            ("obs2", 0.0, True, {"final_answer": 42})
        ]
        mock_env.evaluate_episode.return_value = [0.5, 1.0]
        
        trajectory = mock_batched_env.interact_single_trajectory(
            agent=mock_agent,
            tokenizer=Mock(),
            env_idx=0
        )
        
        # Should have trajectory info + step data
        assert len(trajectory) == 3  # 1 info + 2 steps
        assert "trajectory_reward" in trajectory[0]
        assert trajectory[0]["trajectory_reward"] == 1.5  # Sum of step rewards
        
        # Check step data format
        step_data = trajectory[1]
        assert "observation" in step_data
        assert "action" in step_data
        assert "reward" in step_data
        assert "next_observation" in step_data
        assert "done" in step_data
        assert "mc_return" in step_data


class TestIntegration:
    """Integration tests for the complete environment."""
    
    @pytest.fixture
    def real_env(self):
        """Create a real environment for integration testing (if models available)."""
        try:
            env = GSM8KEnv(max_steps=3, step_penalty=-0.05, device="cpu")
            return env
        except Exception:
            pytest.skip("PRM model not available for integration testing")
    
    def test_full_episode_flow(self, real_env):
        """Test a complete episode flow with real environment."""
        # Reset with specific problem index
        obs = real_env.reset(problem_idx=2)  # Simple problem
        assert real_env.problem == MOCK_GSM8K_PROBLEMS[2]["question"]
        
        # Simulate episode steps
        episode_steps = []
        for step in MOCK_REASONING_STEPS[2]:
            episode_steps.append((obs, step))
            obs, reward, done, info = real_env.step(step)
            if done:
                break
        
        # Test episode evaluation
        if episode_steps:
            rewards = real_env.evaluate_episode(episode_steps)
            assert len(rewards) == len(episode_steps)
            assert all(isinstance(r, float) for r in rewards)


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])