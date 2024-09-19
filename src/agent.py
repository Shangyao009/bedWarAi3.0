import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from collections import deque
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import sys
import signal
from utils import register_game
from game import BedWarGame, ActionId, Settings, globalConst


class SkipFrameWrapper(gym.Wrapper):
    def __init__(self, skip_frames, env):
        super().__init__(env)
        assert skip_frames > 1
        self.skip_frames = skip_frames

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        total_reward_A = reward[0]
        total_reward_B = reward[1]
        for i in range(self.skip_frames - 1):
            if terminated or truncated:
                break
            obs, reward, terminated, truncated, info = self.env.step(
                np.full_like(action, ActionId.NONE)
            )
            total_reward_A += reward[0]
            total_reward_B += reward[1]
        return obs, (total_reward_A, total_reward_B), terminated, truncated, info


class A2C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        n_envs: int,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        critic_layers = [
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),  # estimate V(s)
            nn.ReLU(),
            nn.Linear(1024, 512),  # estimate V(s)
            nn.ReLU(),
            nn.Linear(512, 1),
        ]

        actor_layers = [
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),  # estimate V(s)
            nn.ReLU(),
            nn.Linear(1024, 512),  # estimate V(s)
            nn.ReLU(),
            nn.Linear(
                512, n_actions
            ),  # estimate action logits (will be fed into a softmax later)
        ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x)  # shape: [n_envs,]
        action_logits_vec = self.actor(x)  # shape: [n_envs, n_actions]
        return (state_values, action_logits_vec)

    def select_action(
        self, x: np.ndarray, eligible_actions_mask: np.ndarray[np.bool_]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        state_values, action_logits = self.forward(x)
        # shape: [n_envs,], [n_envs, n_actions]
        eligible_actions_mask = (
            torch.Tensor(eligible_actions_mask).type(torch.bool).to(self.device)
        )
        action_logits[~eligible_actions_mask] = -float("inf")

        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax

        actions_A = action_pd.sample()

        action_log_probs = action_pd.log_prob(actions_A)
        entropy = action_pd.entropy()
        return (actions_A, action_log_probs, state_values, entropy)

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):  # start from T-2 -> 0
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


model_dir = Path("./models")
model_path = model_dir.joinpath("A2C.pt")
log_dir = Path("./logs/A2C")
n_envs = 24
skip_frames = 8
critic_lr = 3e-4
actor_lr = 1e-5
gamma = 0.999
lam = 0.99  # hyperparameter for GAE
ent_coef = 0.15  # coefficient for the entropy bonus (to encourage exploration)
n_updates = 40000
n_demo = 5
n_steps_per_update = 256  # batch size
save_every_n_updates = 200  # not recommended less than 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_size = 48
n_action = 19


def make_env(render_mode=None):
    """Make a gym environment for AsyncVectorEnv"""
    register_game()
    env = gym.make("BedWarGame-v0", render_mode=render_mode)
    env = SkipFrameWrapper(skip_frames, env)
    env = gym.wrappers.AutoResetWrapper(env=env)
    return env


def save_model(model: nn.Module, path: str):
    try:
        torch.save(model.state_dict(), path)
        return True
    except:
        print("Failed to save model")
        return False


def train(n_updates):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = None  # initialize when first logs
    _make_env = []
    for i in range(n_envs):
        _make_env.append(make_env)
    envs = gym.vector.AsyncVectorEnv(_make_env)

    agent = A2C(
        n_features=obs_size,
        n_actions=n_action,
        device=device,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        n_envs=n_envs,
    )

    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_path):
        agent.load_state_dict(
            torch.load(model_path.__str__(), weights_only=True, map_location=device)
        )
        print("Loaded model from", model_path.__str__())

    def save_model_and_exit(signal_received, frame):
        print("Saving model")
        done = save_model(agent, model_path.__str__())
        if done:
            print("Model saved")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_model_and_exit)
    signal.signal(signal.SIGTERM, save_model_and_exit)

    critic_losses_A = deque(maxlen=100)
    critic_losses_B = deque(maxlen=100)
    actor_losses_A = deque(maxlen=100)
    actor_losses_B = deque(maxlen=100)
    entropies_A = deque(maxlen=100)
    entropies_B = deque(maxlen=100)
    rewards_A_deque = deque(maxlen=100)
    rewards_B_deque = deque(maxlen=100)

    # training loop
    states, infos = envs.reset()
    for update_i in tqdm(range(1, 1 + n_updates)):
        # data required for one update
        ep_value_preds_A = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_rewards_A = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_action_log_probs_A = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_value_preds_B = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_rewards_B = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_action_log_probs_B = torch.zeros(n_steps_per_update, n_envs, device=device)

        masks = torch.zeros(n_steps_per_update, n_envs, device=device)

        for step in range(n_steps_per_update):
            # play n_steps_per_update steps in the environment
            states_A = states["A"]  # shape: [n_envs, obs_size]
            states_B = states["B"]  # shape: [n_envs, obs_size]
            eligible_actions_mask_A = np.stack(infos["valid_actions_mask_A"])
            # shape [n_envs, n_actions]
            eligible_actions_mask_B = np.stack(infos["valid_actions_mask_B"])

            actions_A, action_log_probs_A, state_value_preds_A, entropy_A = (
                agent.select_action(states_A, eligible_actions_mask_A)
            )
            actions_B, action_log_probs_B, state_value_preds_B, entropy_B = (
                agent.select_action(states_B, eligible_actions_mask_B)
            )

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            states, rewards, terminated, truncated, infos = envs.step(
                (actions_A.cpu().numpy(), actions_B.cpu().numpy())
            )
            # rewards_A, rewards_B = rewards
            rewards_A = rewards[:, 0]
            rewards_B = rewards[:, 1]

            ep_value_preds_A[step] = torch.squeeze(state_value_preds_A)
            ep_rewards_A[step] = torch.tensor(rewards_A, device=device)
            ep_action_log_probs_A[step] = action_log_probs_A

            ep_value_preds_B[step] = torch.squeeze(state_value_preds_B)
            ep_rewards_B[step] = torch.tensor(rewards_B, device=device)
            ep_action_log_probs_B[step] = action_log_probs_B

            masks[step] = ~torch.tensor(terminated, device=device)

        # calculate the losses for actor and critic
        critic_loss_A, actor_loss_A = agent.get_losses(
            ep_rewards_A,
            ep_action_log_probs_A,
            ep_value_preds_A,
            entropy_A,
            masks,
            gamma,
            lam,
            ent_coef,
            device,
        )

        critic_loss_B, actor_loss_B = agent.get_losses(
            ep_rewards_B,
            ep_action_log_probs_B,
            ep_value_preds_B,
            entropy_B,
            masks,
            gamma,
            lam,
            ent_coef,
            device,
        )
        critic_loss = critic_loss_A + critic_loss_B
        actor_loss = actor_loss_A + actor_loss_B
        # update the actor and critic networks
        agent.update_parameters(critic_loss, actor_loss)

        if update_i % save_every_n_updates == 0:
            save_model(agent, model_path.__str__())

        # log the losses and entropy
        critic_losses_A.append(critic_loss_A.item())
        critic_losses_B.append(critic_loss_B.item())
        actor_losses_A.append(actor_loss_A.item())
        actor_losses_B.append(actor_loss_B.item())
        entropies_A.append(entropy_A.mean().item())
        entropies_B.append(entropy_B.mean().item())
        rewards_A_deque.append(ep_rewards_A.mean().item())
        rewards_B_deque.append(ep_rewards_B.mean().item())

        reward_A = float(np.mean(rewards_A_deque).item())
        reward_B = float(np.mean(rewards_B_deque).item())
        actor_loss_A = float(np.mean(actor_losses_A).item())
        actor_loss_B = float(np.mean(actor_losses_B).item())
        critic_loss_A = float(np.mean(critic_losses_A).item())
        critic_loss_B = float(np.mean(critic_losses_B).item())
        entropy_A = float(np.mean(entropies_A).item())
        entropy_B = float(np.mean(entropies_B).item())
        # print(
        #     f"{reward_A=}, {reward_B=}, {actor_loss_A=}, {actor_loss_B=}, {critic_loss_A=}, {critic_loss_B=}, {entropy_A=}, {entropy_B=}"
        # )
        if writer is None:
            writer = SummaryWriter(
                f"{log_dir.__str__()}/{time_str}_lr_{actor_lr}_{critic_lr}"
            )
        writer.add_scalar("actor_loss_A", actor_loss_A, update_i)
        writer.add_scalar("actor_loss_B", actor_loss_B, update_i)
        writer.add_scalar("critic_loss_A", critic_loss_A, update_i)
        writer.add_scalar("critic_loss_B", critic_loss_B, update_i)
        writer.add_scalar("entropy_A", entropy_A, update_i)
        writer.add_scalar("entropy_B", entropy_B, update_i)
        writer.add_scalar("reward_A", reward_A, update_i)
        writer.add_scalar("reward_B", reward_B, update_i)


def demo(n_episodes):
    import pygame
    import sys

    env = BedWarGame(render_mode="human")
    model = A2C(
        n_features=obs_size,
        n_actions=n_action,
        device=device,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        n_envs=1,
    )

    if model_path.exists():
        model.load_state_dict(
            torch.load(model_path.__str__(), weights_only=True, map_location=device)
        )
        print("Loaded model from", model_path.__str__())
    model.eval()

    skip_frames = 4
    for i in range(1, 1 + n_episodes):
        state, infos = env.reset()
        while True:
            if env.get_ticks() % skip_frames == 0:
                state_A = state["A"]
                state_B = state["B"]
                eligible_actions_mask_A = infos["valid_actions_mask_A"]
                eligible_actions_mask_B = infos["valid_actions_mask_B"]
                action_A, _, _, _ = model.select_action(
                    state_A, eligible_actions_mask_A
                )
                action_B, _, _, _ = model.select_action(
                    state_B, eligible_actions_mask_B
                )
                state, rewards, terminated, truncated, infos = env.step(
                    (action_A.cpu().numpy(), action_B.cpu().numpy())
                )
                # print(f"reward_A :{rewards[0]}")
                # print(f"reward_B :{rewards[1]}")
            else:
                state, _, terminated, truncated, _ = env.step(
                    (ActionId.NONE, ActionId.NONE)
                )
            env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if terminated or truncated:
                env.render()
                break


if __name__ == "__main__":

    is_demo = False
    for arg in sys.argv:
        if arg == "--demo":
            is_demo = True
    print(f"{device=}")
    if is_demo:
        demo(n_episodes=n_demo)
    else:
        train(n_updates=n_updates)
