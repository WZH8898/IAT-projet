from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.dqn_agent import DQNAgent
import numpy as np
from collections import deque
import random

def main():
    num_episodes = 501 # Nombre d'épisodes pour l'entraînement
    max_steps_per_episode = 10000 # Nombre maximum d'étapes par épisode
    batch_size = 64 # Taille du lot pour l'entraînement
    
    game = SpaceInvaders(display=False)
    state_size = len(game.get_state()) # Taille de l'état
    action_size = game.na 
    controller = DQNAgent(state_size, game.na, game.na)
    # controller = RandomAgent(game.na)
    # controller = KeyboardController()

    # Paramètres d'exploration
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    # Create lists to store reward history and average reward history
    reward_history = []
    average_reward_history = []
    
    # Boucle d'apprentissage
    for episode in range(num_episodes):
        # Réinitialisez l'environnement et obtenez l'état initial
        state = game.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            # Sélectionnez une action
            action = controller.select_action(state)

            # Appliquez l'action et obtenez le nouvel état, la récompense et l'indicateur "terminé"
            next_state, reward, done = game.step(action)
            total_reward += reward

            # Mémorisez l'expérience dans la mémoire de l'agent
            controller.remember(state, action, reward, next_state, done)

            # Mettez à jour l'état actuel
            state = next_state

            # Mettre à jour le modèle cible
            if episode % controller.target_update_freq == 0:
                controller.update_target_model()

            # Si la partie est terminée, sortez de la boucle d'étapes
            if done:
                break

        # Entraînez l'agent avec un échantillon d'expériences
        controller.replay(batch_size)

        # Mettre à jour epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        controller.epsilon = epsilon

        # Add total reward to reward history and compute average reward
        reward_history.append(total_reward)
        average_reward = np.mean(reward_history[-100:])
        average_reward_history.append(average_reward)

        # Imprimer le total des récompenses et average recompense pour 100 épisodes precedents
        print(f"Episode: {episode}, Total reward: {total_reward}, Average reward (last 100): {average_reward}")

        # Sauvegardez le modèle à intervalles réguliers
        if episode % 50 == 0:
            controller.save(f"dqn_agent_model_{episode}.pt")
    
    # Test de l'agent avec le dernière modèle
    print("TEST COMMENCE")

    test_epsilon = 0.05
    game = SpaceInvaders(display=True)

    controller.load(f"dqn_agent_model_500.pt")
    controller.epsilon = test_epsilon
    game = SpaceInvaders(display=True)
    state = game.reset()
    total_test_reward = 0

    while True:
        action = controller.act(state) # Sélectionner une action
        state, reward, is_done = game.step(action) # Appliquer l'action et obtenir le nouvel état, la récompense et l'indicateur "terminé"
        total_test_reward += reward # Mettre à jour la récompense totale
        sleep(0.0001)

        if is_done:
            print(f"Test score: {total_test_reward}")
            break

if __name__ == '__main__':
    main()

