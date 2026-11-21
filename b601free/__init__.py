"""Real-robot code for the Seeed reBot B601-RS follower + reBot Arm 102 leader, without LeRobot.

Units everywhere (q, action, joint limits): follower MOTOR-FRAME degrees, i.e. exactly what the Seeed
plugin's get_observation() reports (math.degrees(state.pos)). See the repository README.
"""
