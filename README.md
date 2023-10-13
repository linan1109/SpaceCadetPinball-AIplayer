# SpaceCadetPinball-AI
An AI player that can learn from the game and reach a high score, using or will use Computer Vision, Reinforcement Learning, and Genetic Algorithm.
-  Vision: Recognize the pinball's coordinates and speed vector, the score, and the states of every frame during the play.
- Reinforcement Learning: During one game, train the AI to play the higher score using information gathered from the Vision part.
- Genetic Algorithm: Choose the best players after each generation (game round), and use crossover (average parameters) and mutation (add random noise) to produce new generation.

# To do list
- [x] Basic game control
- [x] Basic game information extraction using cv2
- [x] genetic algorithm design and implementation
- [] genetic algorithm training
- [] game control using trained model
- [] reinforcement learning design and implementation
- [] reinforcement learning training

# Demo
![Game information extraction](./demo/demo.gif)

# limitation for now

Because the window location is based on window relation, so it can not use with full screen mode.
