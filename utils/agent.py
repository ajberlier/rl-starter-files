import torch

import utils

import torch_ac.algos.oc


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, arch, argmax=False, num_envs=1):

        self.arch = arch
        self.argmax = argmax
        self.num_envs = num_envs

        if self.arch.recurrent:
            self.memories = torch.zeros(self.num_envs, self.arch.memory_size, device=utils.device)

        self.arch.to(utils.device)
        self.arch.eval()

    def get_actions(self, obss):
        preprocessed_obss = self.arch.preprocess_obss(obss, device=utils.device)

        with torch.no_grad():
            if self.arch.recurrent:
                if self.arch.__class__.__name__ == 'OCModel':
                    dist, _, option_dist, self.memories = self.arch(preprocessed_obss, self.memories)
                elif self.arch.__class__.__name__ == 'ACModel':
                    dist, _, self.memories = self.arch(preprocessed_obss, self.memories)
            else:
                if self.arch.__class__.__name__ == 'OCModel':
                    dist, _, option_dist = self.arch(preprocessed_obss)
                elif self.arch.__class__.__name__ == 'ACModel':    
                    dist, _ = self.arch(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.arch.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=utils.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
