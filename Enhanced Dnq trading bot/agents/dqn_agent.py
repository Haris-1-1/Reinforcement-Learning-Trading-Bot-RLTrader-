import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. DAS DUELING NEURONALE NETZWERK ---
class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQNNetwork, self).__init__()
        
        # Feature Extraction Layer (Gemeinsam)
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Schutz gegen Overfitting
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Stream 1: Value (Wie gut ist der Marktzustand generell?)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Gibt EINEN Wert zurück (Zustandswert)
        )
        
        # Stream 2: Advantage (Wie viel besser ist eine Aktion als die andere?)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # Gibt Werte für ALLE Aktionen zurück
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Kombinieren: Q(s,a) = V(s) + (A(s,a) - Mean(A(s,a)))
        qvals = values + (advantages - advantages.mean())
        return qvals

# --- 2. DER AGENT ---
class Agent:
    def __init__(self, state_size, action_size, window_size, is_eval=False):
        self.state_size = state_size
        self.action_size = action_size
        self.window_size = window_size
        
        # Input Dimension für das Netz = Window Size * Features pro Step
        self.input_dim = state_size * window_size
        
        self.memory = deque(maxlen=100000)  # Großes Gedächtnis
        self.inventory = []
        
        self.is_eval = is_eval
        
        # Hyperparameter
        self.gamma = 0.99         # Discount Factor (Zukunftsgewichtung)
        self.epsilon = 1.0        # Start-Exploration (100% Zufall)
        self.epsilon_min = 0.05   # Mindest-Exploration (5% Zufall)
        self.epsilon_decay = 0.99995 # Sehr langsamer Decay für komplexes Lernen
        self.learning_rate = 0.0001
        self.batch_size = 64
        
        # Hardware-Beschleunigung
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- DOUBLE DQN SETUP ---
        # 1. Policy Net: Trifft Entscheidungen
        self.policy_net = DuelingDQNNetwork(self.input_dim, action_size).to(self.device)
        # 2. Target Net: Bewertet Entscheidungen (stabil)
        self.target_net = DuelingDQNNetwork(self.input_dim, action_size).to(self.device)
        
        # Synchronisieren am Start
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_network(self):
        """Kopiert Gewichte vom Policy-Net ins Target-Net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        """Wählt eine Aktion basierend auf dem Zustand (Epsilon-Greedy)"""
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Zustand vorbereiten
        state = torch.FloatTensor(state).view(1, -1).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            
        return np.argmax(q_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        """Speichert Erfahrung im Replay Buffer"""
        # Flache Arrays speichern um Speicher zu sparen
        state = state.flatten()
        next_state = next_state.flatten()
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Trainiert das Netzwerk mit Erfahrungen (Double DQN Logic)"""
        if len(self.memory) < self.batch_size:
            return

        # Mini-Batch ziehen
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Daten vorbereiten (Batch-Verarbeitung ist viel schneller)
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)

        # --- DOUBLE DQN MAGIC ---
        # 1. Policy Net wählt die beste Aktion für den nächsten Zustand
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            
            # 2. Target Net berechnet den Q-Wert für DIESE Aktion
            next_q_values = self.target_net(next_states)
            # Wir nehmen den Wert, der zur Aktion vom Policy Net gehört
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Bellman Gleichung
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Aktuelle Vorhersage des Policy Nets
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Loss berechnen und Backpropagation
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping (verhindert explodierende Gradienten)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()

        # Epsilon Decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay