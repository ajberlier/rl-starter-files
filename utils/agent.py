import torch

import utils


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, acmodel, argmax=False, num_envs=1):

        self.acmodel = acmodel
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=utils.device)

        self.acmodel.to(utils.device)
        self.acmodel.eval()

    def get_actions(self, obss):
        preprocessed_obss = self.acmodel.preprocess_obss(obss, device=utils.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=utils.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
