from random import sample
from typing import Any, Mapping, Optional, Sequence, Union

import copy
import itertools
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent
from skrl.resources.schedulers.torch import KLAdaptiveLR

torch.autograd.set_detect_anomaly(True)

# Define a simple Control Barrier Function (CBF) neural network
class CBFNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super(CBFNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output a single value B(x)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# [start-config-dict-torch]
CONSTRAINT_SHARED_PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },
    # Added CBF configuration parameters
    "cbf": {
        "learning_rate": 1e-3,                  # CBF learning rate
        "learning_rate_scheduler": None,        # CBF learning rate scheduler
        "learning_rate_scheduler_kwargs": {},   # CBF learning rate scheduler's kwargs

        "lambda_cbf": 0.95,      # Lambda coefficient for one-step invariance loss
        "cbf_loss_weights": {    # Weights for each CBF loss term
            "feasible": 1.0,
            "infeasible": 1.0,
            "invariance": 1.0
        },
        # Added dual variable configuration
        "dual_lr": 1e-3,          # Learning rate for dual variable
        "dual_init": 1.0          # Initial value for dual variable
    }
}
# [end-config-dict-torch]


class CONSTRAINT_SHARED_PPO(MultiAgent):
    def __init__(self,
                 possible_agents: Sequence[str],
                 models: Mapping[str, Model],
                 memories: Optional[Mapping[str, Memory]] = None,
                 observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gym.Space], Mapping[str, gymnasium.Space]]] = None,
                 action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gym.Space], Mapping[str, gymnasium.Space]]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Constraint Proximal Policy Optimization (CONSTRAINT_SHARED_PPO)

        https://arxiv.org/abs/2011.09533

        :param possible_agents: Name of all possible agents the environment could generate
        :type possible_agents: list of str
        :param models: Models used by the agents.
                       External keys are environment agents' names. Internal keys are the models required by the algorithm
        :type models: nested dictionary of skrl.models.torch.Model
        :param memories: Memories to storage the transitions.
        :type memories: dictionary of skrl.memory.torch.Memory, optional
        :param observation_spaces: Observation/state spaces or shapes (default: ``None``)
        :type observation_spaces: dictionary of int, sequence of int, gym.Space or gymnasium.Space, optional
        :param action_spaces: Action spaces or shapes (default: ``None``)
        :type action_spaces: dictionary of int, sequence of int, gym.Space or gymnasium.Space, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        _cfg = copy.deepcopy(CONSTRAINT_SHARED_PPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(possible_agents=possible_agents,
                         models=models,
                         memories=memories,
                         observation_spaces=observation_spaces,
                         action_spaces=action_spaces,
                         device=device,
                         cfg=_cfg)
        # shared index for extracting the shared variables
        self.shared_uid = next(iter(self.possible_agents))
        # models
        # all agents share the same policy and value models
        self.shared_policy = self.models[self.shared_uid].get("policy", None)
        self.shared_value = self.models[self.shared_uid].get("value", None)
        
        for uid in self.possible_agents:
            # checkpoint models
            self.checkpoint_modules[uid]["policy"] = self.shared_policy
            self.checkpoint_modules[uid]["value"] = self.shared_value

            # broadcast models' parameters in distributed runs
            if config.torch.is_distributed:
                logger.info(f"Broadcasting models' parameters")
                if self.shared_policy is not None:
                    self.shared_policy.broadcast_parameters()
                    if self.shared_value is not None and self.shared_policy is not self.shared_value:
                        self.shared_value.broadcast_parameters()

        # configuration
        self._learning_epochs = self._as_dict(self.cfg["learning_epochs"])
        self._mini_batches = self._as_dict(self.cfg["mini_batches"])
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self._as_dict(self.cfg["grad_norm_clip"])
        self._ratio_clip = self._as_dict(self.cfg["ratio_clip"])
        self._value_clip = self._as_dict(self.cfg["value_clip"])
        self._clip_predicted_values = self._as_dict(self.cfg["clip_predicted_values"])

        self._value_loss_scale = self._as_dict(self.cfg["value_loss_scale"])
        self._entropy_loss_scale = self._as_dict(self.cfg["entropy_loss_scale"])

        self._kl_threshold = self._as_dict(self.cfg["kl_threshold"])

        self._learning_rate = self._as_dict(self.cfg["learning_rate"])
        self._learning_rate_scheduler = self._as_dict(self.cfg["learning_rate_scheduler"])
        self._learning_rate_scheduler_kwargs = self._as_dict(self.cfg["learning_rate_scheduler_kwargs"])

        self._state_preprocessor = self._as_dict(self.cfg["state_preprocessor"])
        self._state_preprocessor_kwargs = self._as_dict(self.cfg["state_preprocessor_kwargs"])
        self._value_preprocessor = self._as_dict(self.cfg["value_preprocessor"])
        self._value_preprocessor_kwargs = self._as_dict(self.cfg["value_preprocessor_kwargs"])

        self._discount_factor = self._as_dict(self.cfg["discount_factor"])
        self._lambda = self._as_dict(self.cfg["lambda"])

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self._as_dict(self.cfg["time_limit_bootstrap"])

        # set up optimizer and learning rate scheduler
        if self.shared_policy is not None and self.shared_value is not None:
            if self.shared_policy is self.shared_value:
                self.shared_optimizer = torch.optim.Adam(self.shared_policy.parameters(), lr=self._learning_rate[self.shared_uid])
            else:
                self.shared_optimizer = torch.optim.Adam(itertools.chain(self.shared_policy.parameters(), self.shared_value.parameters()), lr=self._learning_rate[self.shared_uid])
            self.shared_scheduler = self._learning_rate_scheduler[self.shared_uid](self.shared_optimizer, **self._learning_rate_scheduler_kwargs[self.shared_uid]) if self._learning_rate_scheduler[self.shared_uid] else None

        if self._state_preprocessor[self.shared_uid] is not None:
            self.shared_state_preprocessor = self._state_preprocessor[self.shared_uid](**self._state_preprocessor_kwargs[self.shared_uid])
            for uid in self.possible_agents:
                self.checkpoint_modules[uid]["state_preprocessor"] = self.shared_state_preprocessor
        else:
            self.shared_state_preprocessor = self._empty_preprocessor
        if self._value_preprocessor[self.shared_uid] is not None:
            self.shared_value_preprocessor = self._value_preprocessor[self.shared_uid](**self._value_preprocessor_kwargs[self.shared_uid])
            for uid in self.possible_agents:
                self.checkpoint_modules[uid]["value_preprocessor"] = self.shared_value_preprocessor
        else:
            self.shared_value_preprocessor = self._empty_preprocessor
        
        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["optimizer"] = self.shared_optimizer

        # -------------- Initialize CBF Network --------------
        # Assume observation_spaces is a dictionary and each agent has the same observation space size
        if isinstance(self.observation_spaces[self.shared_uid], (gym.spaces.Box, gymnasium.spaces.Box)):
            # if the observation space is a Box, the size is the shape of the Box
            observation_space_size = self.observation_spaces[self.shared_uid].shape[0]
        else:
            observation_space_size = self.observation_spaces[self.shared_uid]
        self.cbf_network = CBFNetwork(input_size=observation_space_size).to(self.device)
        
        # Initialize CBF optimizer
        cbf_learning_rate = self.cfg["cbf"]["learning_rate"]
        self.cbf_optimizer = torch.optim.Adam(self.cbf_network.parameters(), lr=cbf_learning_rate)
        
        # Initialize CBF learning rate scheduler if specified
        if self.cfg["cbf"]["learning_rate_scheduler"] is not None:
            self.cbf_scheduler = self.cfg["cbf"]["learning_rate_scheduler"](
                self.cbf_optimizer, **self.cfg["cbf"]["learning_rate_scheduler_kwargs"]
            )
        else:
            self.cbf_scheduler = None

        # Get CBF lambda and loss weights
        self.cbf_lambda = self.cfg["cbf"]["lambda_cbf"]
        self.cbf_loss_weights = self.cfg["cbf"]["cbf_loss_weights"]
        # -----------------------------------------------------------

        # -------------- Initialize Dual Variable --------------
        # Define dual variable as a learnable parameter, initialized to dual_init
        self.nu_param = nn.Parameter(torch.tensor(self.cfg["cbf"]["dual_init"], device=self.device))
        self.nu = F.softplus(self.nu_param).detach()

        # Initialize dual optimizer
        self.dual_optimizer = torch.optim.Adam([self.nu_param], lr=self.cfg["cbf"]["dual_lr"])
        # -----------------------------------------------------------

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memories
        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(name="states", size=self.observation_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="advantages", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="next_states", size=self.observation_spaces[uid], dtype=torch.float32)

                # tensors sampled during training
                self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages", "next_states"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = []
        self._current_next_states = []

    def act(self, states: Mapping[str, torch.Tensor], timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policies

        :param states: Environment's states
        :type states: dictionary of torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample stochastic actions
        data = [self.shared_policy.act({"states": self.shared_state_preprocessor(states[uid])}, role="policy") for uid in self.possible_agents]

        actions = {uid: d[0] for uid, d in zip(self.possible_agents, data)}
        log_prob = {uid: d[1] for uid, d in zip(self.possible_agents, data)}
        outputs = {uid: d[2] for uid, d in zip(self.possible_agents, data)}

        self._current_log_prob = log_prob

        return actions, log_prob, outputs

    def record_transition(self,
                          states: Mapping[str, torch.Tensor],
                          actions: Mapping[str, torch.Tensor],
                          rewards: Mapping[str, torch.Tensor],
                          next_states: Mapping[str, torch.Tensor],
                          terminated: Mapping[str, torch.Tensor],
                          truncated: Mapping[str, torch.Tensor],
                          infos: Mapping[str, Any],
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: dictionary of torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: dictionary of torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: dictionary of torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: dictionary of torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: dictionary of torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: dictionary of torch.Tensor
        :param infos: Additional information about the environment
        :type infos: dictionary of any supported type
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if self.memories:
            self._current_next_states = next_states

            for uid in self.possible_agents:
                # reward shaping
                if self._rewards_shaper is not None:
                    rewards[uid] = self._rewards_shaper(rewards[uid], timestep, timesteps)

                # compute values
                values, _, _ = self.shared_value.act({"states": self.shared_state_preprocessor(states[uid])}, role="value")
                values = self.shared_value_preprocessor(values, inverse=True)

                # time-limit (truncation) boostrapping
                if self._time_limit_bootstrap[uid]:
                    rewards[uid] += self._discount_factor[uid] * values * truncated[uid]

                # storage transition in memory
                self.memories[uid].add_samples(states=states[uid], actions=actions[uid], rewards=rewards[uid], next_states=next_states[uid],
                                               terminated=terminated[uid], truncated=truncated[uid], log_prob=self._current_log_prob[uid], values=values)

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        def compute_gae(rewards: torch.Tensor,
                        dones: torch.Tensor,
                        values: torch.Tensor,
                        next_values: torch.Tensor,
                        discount_factor: float = 0.99,
                        lambda_coefficient: float = 0.95) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages
        
        all_sampled_batches = {}

        policy = self.shared_policy
        value = self.shared_value
        for uid in self.possible_agents:
            memory = self.memories[uid]
            # compute returns and advantages
            with torch.no_grad():
                value.train(False)
                last_values, _, _ = value.act({"states": self.shared_state_preprocessor(self._current_next_states[uid].float())}, role="value")
                value.train(True)
            last_values = self.shared_value_preprocessor(last_values, inverse=True)

            values = memory.get_tensor_by_name("values")
            rewards = memory.get_tensor_by_name("rewards")
            dones = memory.get_tensor_by_name("terminated")

            returns, advantages = compute_gae(rewards=rewards,
                                              dones=dones,
                                              values=values,
                                              next_values=last_values,
                                              discount_factor=self._discount_factor[self.shared_uid],
                                              lambda_coefficient=self._lambda[self.shared_uid])

            memory.set_tensor_by_name("values", self.shared_value_preprocessor(values, train=True))
            memory.set_tensor_by_name("returns", self.shared_value_preprocessor(returns, train=True))
            memory.set_tensor_by_name("advantages", advantages)

            # sample mini-batches from memory
            sampled_batches = memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches[self.shared_uid])
            all_sampled_batches[uid] = sampled_batches

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # Initialize CBF loss accumulation
        cumulative_cbf_loss = 0

        # Total number of CBF steps (for averaging)
        total_cbf_steps = 0

        # learning epochs
        for epoch in range(self._learning_epochs[self.shared_uid]):
            kl_divergences = []

            # Mini-batches loop
            for mini_batch_idx in range(self._mini_batches[self.shared_uid]):
                # concatenate mini-batches of all agents
                sampled_states = []
                sampled_actions = []
                sampled_log_prob = []
                sampled_values = []
                sampled_returns = []
                sampled_advantages = []
                next_log_prob = []
                predicted_values = []
                for uid in self.possible_agents:
                    sampled_states_uid, sampled_actions_uid, sampled_log_prob_uid, sampled_values_uid, sampled_returns_uid, sampled_advantages_uid, sampled_next_states = all_sampled_batches[uid][mini_batch_idx]
                    sampled_states_uid = self.shared_state_preprocessor(sampled_states_uid, train=not epoch)
                    _, next_log_prob_uid, _ = policy.act({"states": sampled_states_uid, "taken_actions": sampled_actions_uid}, role="policy")
                    predicted_values_uid, _, _ = value.act({"states": sampled_states_uid}, role="value")
                    sampled_states.append(sampled_states_uid)
                    sampled_actions.append(sampled_actions_uid)
                    sampled_log_prob.append(sampled_log_prob_uid)
                    sampled_values.append(sampled_values_uid)
                    sampled_returns.append(sampled_returns_uid)
                    sampled_advantages.append(sampled_advantages_uid)
                    next_log_prob.append(next_log_prob_uid)
                    predicted_values.append(predicted_values_uid)
                sampled_states = torch.cat(sampled_states, dim=0)
                sampled_actions = torch.cat(sampled_actions, dim=0)
                sampled_log_prob = torch.cat(sampled_log_prob, dim=0)
                sampled_values = torch.cat(sampled_values, dim=0)
                sampled_returns = torch.cat(sampled_returns, dim=0)
                sampled_advantages = torch.cat(sampled_advantages, dim=0)
                next_log_prob = torch.cat(next_log_prob, dim=0)
                predicted_values = torch.cat(predicted_values, dim=0)

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = next_log_prob - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # early stopping with KL divergence
                if self._kl_threshold[self.shared_uid] and kl_divergence > self._kl_threshold[self.shared_uid]:
                    logger.info(f"Early stopping at epoch {epoch}, mini-batch {mini_batch_idx} due to reaching max KL.")
                    break

                # compute entropy loss
                if self._entropy_loss_scale[self.shared_uid]:
                    entropy_loss = -self._entropy_loss_scale[self.shared_uid] * policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(next_log_prob - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip[self.shared_uid], 1.0 + self._ratio_clip[self.shared_uid])

                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                    min=-self._value_clip[self.shared_uid],
                                                                    max=self._value_clip[self.shared_uid])
                value_loss = self._value_loss_scale[self.shared_uid] * F.mse_loss(sampled_returns, predicted_values)

                # -------------- Integrated: CBF Network Training --------------
                if self.cbf_network is not None and self.memories:
                    cbf_loss_total = 0
                    for uid in self.possible_agents:
                        memory = self.memories[uid]
                        # Retrieve initial and unsafe states from memory
                        transition_states = memory.get_tensor_by_name("states")
                        next_transition_states = memory.get_tensor_by_name("next_states")
                        terminated = memory.get_tensor_by_name("terminated")
                        # convert into 2D tensor
                        transition_states = transition_states.view(-1, transition_states.size(-1))
                        next_transition_states = next_transition_states.view(-1, next_transition_states.size(-1))
                        terminated = terminated.view(-1)
                        # Filter out safe/init states and unsafe states
                        not_terminated = terminated.logical_not()
                        terminated_index = terminated.nonzero(as_tuple=False).squeeze(-1)
                        not_terminated_index = not_terminated.nonzero(as_tuple=False).squeeze(-1)
                        Dinit_states = transition_states[terminated_index] if terminated_index.numel() > 0 else None
                        Dunsafe_states = transition_states[not_terminated_index] if not_terminated_index.numel() > 0 else None
                        
                        # 1. Feasible Loss: B(x_init) ≤ 0
                        if Dinit_states is not None and Dinit_states.numel() > 0:
                            B_init = self.cbf_network(Dinit_states)
                            J_feasible = F.relu(B_init).mean()
                        else:
                            J_feasible = torch.tensor(0.0, device=self.device)

                        # 2. Infeasible Loss: B(x_unsafe) > 0
                        if Dunsafe_states is not None and Dunsafe_states.numel() > 0:
                            B_unsafe = self.cbf_network(Dunsafe_states)
                            J_infeasible = F.relu(-B_unsafe).mean()
                        else:
                            J_infeasible = torch.tensor(0.0, device=self.device)

                        # 3. Invariance Loss: B(x') - (1 - lambda) * B(x) ≤ 0
                        if transition_states is not None and next_transition_states is not None and transition_states.numel() > 0:
                            B_x = self.cbf_network(transition_states)
                            B_x_prime = self.cbf_network(next_transition_states)
                            J_invariance = F.relu(B_x_prime - (1 - self.cbf_lambda) * B_x).mean()
                        else:
                            J_invariance = torch.tensor(0.0, device=self.device)

                        # Total CBF loss as a weighted sum of the three loss terms
                        J_cbf = (
                            self.cbf_loss_weights["feasible"] * J_feasible +
                            self.cbf_loss_weights["infeasible"] * J_infeasible +
                            self.cbf_loss_weights["invariance"] * J_invariance
                        )

                        cbf_loss_total += J_cbf

                    # Average CBF loss over all agents
                    cbf_loss_total /= len(self.possible_agents)
                    cumulative_cbf_loss += cbf_loss_total.item()
                    total_cbf_steps += 1

                # -----------------------------------------------------------
                
                # ------------------ Update Dual Variable ------------------
                # Calculate the invariant loss for dual update
                # Assuming J_invariance has been computed above
                dual_loss = -self.nu * ratio * J_invariance  # Negative for gradient ascent

                self.dual_optimizer.zero_grad()
                dual_loss.mean().backward(retain_graph=True)
                self.dual_optimizer.step()
                
                self.nu = F.softplus(self.nu_param).detach()
                # -----------------------------------------------------------

                # -------------- Update Policy and Value Networks --------------
                # Define total PPO loss including the dual-scaled invariance loss
                Lagrangian_loss = policy_loss + entropy_loss + value_loss - (self.nu * ratio * J_invariance.detach()).mean()

                # Optimize policy and value networks
                self.shared_optimizer.zero_grad()
                Lagrangian_loss.backward(retain_graph=True)
                if config.torch.is_distributed:
                    policy.reduce_parameters()
                    if policy is not value:
                        value.reduce_parameters()
                if self._grad_norm_clip[self.shared_uid] > 0:
                    if policy is value:
                        nn.utils.clip_grad_norm_(policy.parameters(), self._grad_norm_clip[self.shared_uid])
                    else:
                        nn.utils.clip_grad_norm_(itertools.chain(policy.parameters(), value.parameters()), self._grad_norm_clip[self.shared_uid])
                self.shared_optimizer.step()

                # ------------------ Update CBF Network ------------------
                self.cbf_optimizer.zero_grad()
                cbf_loss_total.backward()
                if self._grad_norm_clip[self.shared_uid] > 0:
                    nn.utils.clip_grad_norm_(self.cbf_network.parameters(), self._grad_norm_clip[self.shared_uid])
                self.cbf_optimizer.step()

                # Update CBF learning rate scheduler if available
                if self.cbf_scheduler is not None:
                    self.cbf_scheduler.step()
                # -----------------------------------------------------------

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale[self.shared_uid]:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler[self.shared_uid]:
                if isinstance(self.shared_scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= (config.torch.world_size * len(self.possible_agents))
                    self.shared_scheduler.step(kl.item())
                else:
                    self.shared_scheduler.step()

        # Record PPO losses
        num_updates = self._learning_epochs[self.shared_uid] * self._mini_batches[self.shared_uid]
        self.track_data(f"Loss / Policy loss", cumulative_policy_loss / num_updates)
        self.track_data(f"Loss / Value loss", cumulative_value_loss / num_updates)
        if self._entropy_loss_scale[self.shared_uid]:
            self.track_data(f"Loss / Entropy loss", cumulative_entropy_loss / num_updates)

        self.track_data(f"Policy / Standard deviation", policy.distribution(role="policy").stddev.mean().item())
        
        if self._learning_rate_scheduler[self.shared_uid]:
            self.track_data(f"Learning / Learning rate", self.shared_scheduler.get_last_lr()[0])

        # Record CBF losses if any updates were performed
        if total_cbf_steps > 0:
            avg_cbf_loss = cumulative_cbf_loss / total_cbf_steps
            self.track_data(f"Loss / CBF Total Loss", avg_cbf_loss)
        # -----------------------------------------------------------