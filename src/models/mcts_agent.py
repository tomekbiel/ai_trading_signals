import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List

class MCTSAgent:
    """PyTorch Actor-Critic + Kelly criterion integration"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.001)
        
        # MCTS parameters
        self.num_simulations = 1000
        self.exploration_constant = 1.41  # UCT constant
        self.gamma = 0.99  # discount factor
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using MCTS + Actor-Critic"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get initial policy and value from networks
        action_probs = self.actor(state_tensor)
        state_value = self.critic(state_tensor)
        
        # Run MCTS simulations
        action = self.mcts_search(state, action_probs, state_value.item())
        
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(state, action)
        
        return action, action_probs[0][action].item(), kelly_fraction
    
    def mcts_search(self, state: np.ndarray, policy_probs: torch.Tensor, value: float) -> int:
        """Simplified MCTS search with neural network guidance"""
        # Root node
        root = MCTSNode(state, value, policy_probs)
        
        for _ in range(self.num_simulations):
            node = root
            
            # Selection - traverse to leaf
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child(self.exploration_constant)
            
            # Expansion - add new child if not terminal
            if not node.is_terminal():
                action_probs, state_value = self.evaluate_state(node.state)
                node = node.expand(action_probs, state_value)
            
            # Simulation - rollout to terminal
            rollout_value = self.rollout(node.state)
            
            # Backpropagation
            node.backpropagate(rollout_value, self.gamma)
        
        # Select most visited action
        visit_counts = [child.visits for child in root.children]
        return np.argmax(visit_counts)
    
    def evaluate_state(self, state: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Evaluate state using neural networks"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        state_value = self.critic(state_tensor)
        return action_probs, state_value.item()
    
    def rollout(self, state: np.ndarray, max_depth: int = 10) -> float:
        """Simple rollout with random actions"""
        current_state = state.copy()
        total_reward = 0.0
        
        for _ in range(max_depth):
            # Random action for simplicity
            action = np.random.randint(0, self.action_dim)
            
            # Simplified reward calculation
            reward = self.calculate_reward(current_state, action)
            total_reward += self.gamma ** _ * reward
            
            # Update state (simplified)
            current_state = self.update_state(current_state, action)
            
        return total_reward
    
    def calculate_reward(self, state: np.ndarray, action: int) -> float:
        """Calculate immediate reward"""
        # Simplified reward based on price change
        if len(state) > 1:
            price_change = state[-1] - state[-2] if len(state) >= 2 else 0
            
            # Action mapping: 0=hold, 1=buy, 2=sell
            if action == 1 and price_change > 0:  # buy and price goes up
                return abs(price_change)
            elif action == 2 and price_change < 0:  # sell and price goes down
                return abs(price_change)
            else:
                return -abs(price_change) * 0.5
        return 0.0
    
    def update_state(self, state: np.ndarray, action: int) -> np.ndarray:
        """Update state based on action"""
        # Simplified state update
        new_state = state.copy()
        
        # Add some noise to simulate market movement
        noise = np.random.normal(0, 0.01)
        if len(new_state) > 0:
            new_state[-1] += noise
            
        return new_state
    
    def calculate_kelly_fraction(self, state: np.ndarray, action: int) -> float:
        """Calculate Kelly criterion fraction"""
        # Simplified Kelly calculation
        if action == 0:  # hold
            return 0.0
        
        # Estimate win probability and payoff
        win_prob = self.estimate_win_probability(state, action)
        payoff = self.estimate_payoff(state, action)
        
        # Kelly formula: f* = (bp - q) / b
        # where b = payoff odds, p = win prob, q = lose prob
        if payoff > 0:
            kelly = (payoff * win_prob - (1 - win_prob)) / payoff
            return np.clip(kelly, 0.0, 1.0)  # Full Kelly capped at 1.0
        
        return 0.0
    
    def estimate_win_probability(self, state: np.ndarray, action: int) -> float:
        """Estimate probability of winning trade"""
        # Simplified - use recent price trend
        if len(state) > 5:
            recent_changes = np.diff(state[-5:])
            positive_changes = np.sum(recent_changes > 0)
            
            if action == 1:  # buy
                return positive_changes / len(recent_changes)
            elif action == 2:  # sell
                return 1.0 - (positive_changes / len(recent_changes))
        
        return 0.5  # default 50%
    
    def estimate_payoff(self, state: np.ndarray, action: int) -> float:
        """Estimate payoff odds"""
        # Simplified payoff estimation
        if len(state) > 1:
            volatility = np.std(np.diff(state[-10:])) if len(state) >= 10 else 0.01
            return 1.0 + volatility  # Simple payoff estimate
        
        return 1.0
    
    def train_step(self, states: List[np.ndarray], actions: List[int], rewards: List[float]):
        """Training step for actor-critic"""
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        
        # Calculate values
        values = self.critic(states_tensor).squeeze()
        
        # Calculate advantages
        advantages = rewards_tensor - values.detach()
        
        # Actor loss
        action_probs = self.actor(states_tensor)
        action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1))).squeeze()
        actor_loss = -(action_log_probs * advantages).mean()
        
        # Critic loss
        critic_loss = nn.MSELoss()(values, rewards_tensor)
        
        # Update networks
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        return actor_loss.item(), critic_loss.item()


class MCTSNode:
    """MCTS node implementation"""
    
    def __init__(self, state: np.ndarray, value: float, policy_probs: torch.Tensor):
        self.state = state
        self.value = value
        self.policy_probs = policy_probs
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        
    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0
    
    def is_terminal(self) -> bool:
        return False  # Simplified - never terminal
    
    def select_child(self, exploration_constant: float):
        """Select child using UCT formula"""
        best_child = None
        best_value = -float('inf')
        
        for child in self.children:
            uct_value = child.total_value / child.visits + \
                       exploration_constant * np.sqrt(np.log(self.visits) / child.visits)
            
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
                
        return best_child
    
    def expand(self, policy_probs: torch.Tensor, state_value: float):
        """Expand node with new child"""
        child = MCTSNode(self.state, state_value, policy_probs)
        self.children.append(child)
        return child
    
    def backpropagate(self, value: float, gamma: float):
        """Backpropagate value through tree"""
        self.visits += 1
        self.total_value += value
        
        for child in self.children:
            child.backpropagate(value * gamma, gamma)
