"""
Script name: randomize_conds.py

Purpose:
    Generate pseudorandomized condition files for each participant
    in the AI Perception experiment. The script balances social
    and nonsocial topics across runs and agents, ensuring that each
    agent appears with both topic types in different blocks.

Inputs:
    - conds.csv : Base configuration file with columns:
        ['run', 'agent', 'topic', 'social']
        where 'social' = 1 (social topic) or 0 (nonsocial topic).

Parameters:
    - num_participants : Total number of participants to generate
      pseudorandomized condition files for (default = 45).

Outputs:
    - config/conds_sXX.csv :
      One pseudorandomized condition file per participant,
      where XX is the participant number (01–45).
      Each file contains balanced and randomized topic–agent
      assignments with trial order per run.

Usage:
    Run from the command line:
        python randomize_conds.py

    This will generate a directory called
    `config/` containing one CSV per participant.

Author: Rachel C. Metzgar
Date: 2025-02-12
"""


import pandas as pd
import random
import os

# Number of participants and input configuration file
num_participants = 30
config_file = "config/conds.csv"

def pseudorandomize_conditions(config_file, num_participants):
    # Load the original conditions file
    df = pd.read_csv(config_file)

    # Separate social and nonsocial topics
    social_topics = df[df['social'] == 1]
    nonsocial_topics = df[df['social'] == 0]

    # Ensure the output directory exists
    output_dir = "config"
    os.makedirs(output_dir, exist_ok=True)

    for participant in range(1, num_participants + 1):
        pseudorandomized = []

        # Loop through each run (1 to 5)
        for run in df['run'].unique():
            run_social = social_topics[social_topics['run'] == run].copy()
            run_nonsocial = nonsocial_topics[nonsocial_topics['run'] == run].copy()

            # Shuffle topics and agents
            social_topics_list = run_social['topic'].tolist()
            nonsocial_topics_list = run_nonsocial['topic'].tolist()
            random.shuffle(social_topics_list)
            random.shuffle(nonsocial_topics_list)

            agents = df['agent'].unique().tolist()
            random.shuffle(agents)

            # Initialize Block 1 and Block 2
            block_1 = []
            block_2 = []

            # Assign topics to agents in Block 1 and the opposite valence in Block 2
            for agent in agents:
                # Randomly decide valence for Block 1
                if random.choice([True, False]):
                    topic_1 = social_topics_list.pop()
                    topic_2 = nonsocial_topics_list.pop()
                    block_1.append({'run': run, 'agent': agent, 'topic': topic_1, 'social': 1})
                    block_2.append({'run': run, 'agent': agent, 'topic': topic_2, 'social': 0})
                else:
                    topic_1 = nonsocial_topics_list.pop()
                    topic_2 = social_topics_list.pop()
                    block_1.append({'run': run, 'agent': agent, 'topic': topic_1, 'social': 0})
                    block_2.append({'run': run, 'agent': agent, 'topic': topic_2, 'social': 1})

            # Shuffle order within each block
            random.shuffle(block_1)
            random.shuffle(block_2)

            # Assign trial order within each block
            for i, trial in enumerate(block_1):
                trial['order'] = i + 1
            for i, trial in enumerate(block_2):
                trial['order'] = i + 1 + len(block_1)

            # Add the trials for this run to the full pseudorandomized list
            pseudorandomized.extend(block_1 + block_2)

        # Convert the pseudorandomized list to a DataFrame
        pseudorandomized_df = pd.DataFrame(pseudorandomized)

        # Save the pseudorandomized DataFrame to a CSV file for the current participant
        output_file = os.path.join(output_dir, f"conds_s{participant:02d}.csv")
        pseudorandomized_df.to_csv(output_file, index=False)

# Run the function to generate pseudorandomized conditions
pseudorandomize_conditions(config_file, num_participants)
