import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
import gymnasium
import re

import utils


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, env_obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        self.env_obs_space = env_obs_space
        self.obs_space = None  # setup in setup_obss_preprocessor
        self.preprocess_obss = None  # setup in setup_obss_preprocessor
        self.setup_obss_preprocessor()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = self.obs_space["image"][0]
        m = self.obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(self.obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @staticmethod
    def preprocess_images(images, device=None):
        # Bug of Pytorch: very slow if not first converted to numpy array
        images = np.array(images)
        return torch.tensor(images, device=device, dtype=torch.float)

    @staticmethod
    def preprocess_texts(texts, vocab, device=None):
        var_indexed_texts = []
        max_text_len = 0

        for text in texts:
            tokens = re.findall("([a-z]+)", text.lower())
            var_indexed_text = np.array([vocab[token] for token in tokens])
            var_indexed_texts.append(var_indexed_text)
            max_text_len = max(len(var_indexed_text), max_text_len)

        indexed_texts = np.zeros((len(texts), max_text_len))

        for i, indexed_text in enumerate(var_indexed_texts):
            indexed_texts[i, :len(indexed_text)] = indexed_text

        return torch.tensor(indexed_texts, device=device, dtype=torch.long)

    def setup_obss_preprocessor(self):
        # Check if obs_space is an image space
        if isinstance(self.env_obs_space, gymnasium.spaces.Box):
            obs_space = {"image": self.env_obs_space.shape}

            def preprocess_obss(obss, device=None):
                return torch_ac.DictList({
                    "image": self.preprocess_images(obss, device=device)
                })

        # Check if it is a MiniGrid observation space
        elif isinstance(self.env_obs_space, gymnasium.spaces.Dict) and "image" in self.env_obs_space.spaces.keys():
            obs_space = {"image": self.env_obs_space.spaces["image"].shape, "text": 100}

            vocab = utils.format.Vocabulary(obs_space["text"])

            def preprocess_obss(obss, device=None):
                return torch_ac.DictList({
                    "image": self.preprocess_images([obs["image"] for obs in obss], device=device),
                    "text": self.preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
                })

            preprocess_obss.vocab = vocab

        else:
            raise ValueError("Unknown observation space: " + str(self.env_obs_space))

        self.obs_space = obs_space
        self.preprocess_obss = preprocess_obss

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
