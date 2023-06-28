import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Création des couches du réseau
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    # Passe en avant à travers le réseau
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Sauvegarde les poids du modèle
    def save_weights(self, path):
        torch.save(self.state_dict(), path)

class DQNAgent:
    def __init__(self, input_dim, output_dim, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,target_update_freq=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Création des modèles et optimiseur
        self.model = DQN(input_dim, output_dim).to(self.device)
        self.target_model = DQN(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Paramètres
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.action_size = action_size
        self.target_update_freq = target_update_freq

    # Sélection d'une action selon epsilon-greedy
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 4)
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)
        return np.argmax(q_values.detach().numpy())
    
    # Ajout d'une expérience à la mémoire
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Apprentissage à partir d'un échantillon d'expériences stockées dans la mémoire de l'agent
    def replay(self, batch_size):

        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convertir les listes en tenseurs PyTorch et les déplacer vers l'appareil approprié (GPU ou CPU)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Calculer les valeurs Q pour les états et les états suivants
        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        target_q_values = q_values.clone()

        # Mettre à jour les valeurs Q cibles pour les actions effectuées
        for i in range(batch_size):
            target_q_values[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i]).detach() * torch.logical_not(dones[i])

        # Calculer la perte en utilisant l'erreur quadratique moyenne (MSE) entre les valeurs Q et les valeurs Q cibles
        loss = nn.MSELoss()(q_values, target_q_values)

        # Effectuer la rétropropagation et mettre à jour les poids du modèle
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mettre à jour epsilon pour diminuer progressivement l'exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Sauvegarde les poids du modèle dans un fichier
    def save(self, name):
        self.model.save_weights(name)
    
    # Charge les poids du modèle à partir d'un fichier
    def load(self, name):
        self.model.load_state_dict(torch.load(name))
    
    # Prédit l'action optimale pour un état donné
    def act(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        act_values = self.model(state)
        action = np.argmax(act_values.cpu().data.numpy())
        return action
    
    # Met à jour le modèle cible avec les poids du modèle principal
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
