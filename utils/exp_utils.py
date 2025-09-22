"""
Script: experiment_utils.py

Purpose:
    Helper functions for running the AI Perception experiment in PsychoPy.
    Provides utilities for:
        - Logging event timestamps,
        - Loading conversation prompts,
        - Handling scanner/experimenter synchronization,
        - Collecting participant feedback (quality and connectedness ratings).

Inputs:
    - globalClock   : PsychoPy core.Clock instance
    - trials        : PsychoPy TrialHandler instance
    - thisExp       : PsychoPy ExperimentHandler instance
    - win           : PsychoPy visual.Window instance
    - topic         : str, conversation topic name (for load_prompt)
    - run, trial_num: int, identifiers for feedback logging

Outputs:
    - Event timestamps logged to `trials` and PsychoPy log.
    - Feedback ratings stored in `thisExp`.
    - Prompt text (string) loaded from topic file.

Usage:
    from utils.experiment_utils import (
        log_event_time,
        load_prompt,
        wait_for_scanner_signal,
        wait_for_experimenter,
        collect_feedback
    )

Author: Rachel C. Metzgar
Date: 2025-02-12
"""

import os
from psychopy import visual, core, event, logging


def log_event_time(event: str, globalClock, trials):
    """Log the event timestamp to both TrialHandler and PsychoPy log."""
    eventOnsetTime = globalClock.getTime()
    trials.addData(event, eventOnsetTime)
    logging.exp(f"Event: {event} at {eventOnsetTime}")


def load_prompt(topic: str) -> str:
    """Load the prompt for the given topic from a text file."""
    prompt_file = f"utils/prompts/{topic}.txt"
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            return f.read()
    return f"Error: {topic} prompt not found."


def wait_for_scanner_signal(win, globalClock, trials):
    """Wait for scanner signal (five '=' key presses) before starting run."""
    waiting_text = visual.TextStim(
        win,
        text="Waiting for scanner signal to start the next run...",
        pos=(0, 0), height=20, color=(0, 0, 0)
    )
    waiting_text.draw()
    win.flip()

    signal_count = 0
    while signal_count < 5:
        keys = event.getKeys()
        if "q" in keys:
            raise KeyboardInterrupt
        elif "equal" in keys:
            signal_count += 1
            print(f"Scanner signal count: {signal_count}")
        elif keys:
            signal_count = 0
            print("Non-equal key pressed, resetting count.")
        core.wait(0.1)

    log_event_time("run_start_time", globalClock, trials)
    print("Scanner signal received. Starting run.")


def wait_for_experimenter(win, globalClock, trials):
    """Wait for experimenter to press 'x' before continuing to scanner signal."""
    print("Waiting for the experimenter to press 'x'...")
    waiting_text = visual.TextStim(
        win,
        text="Experimenter: Press 'x' to continue to the next scanner signal.",
        pos=(0, 0), height=20, color=(0, 0, 0)
    )
    waiting_text.draw()
    win.flip()

    while True:
        keys = event.getKeys(keyList=["x", "q"])
        if "x" in keys:
            log_event_time("experimenter_ready", globalClock, trials)
            print("Experimenter pressed 'x'. Proceeding to scanner signal.")
            break
        elif "q" in keys:
            raise KeyboardInterrupt
        core.wait(0.1)


def collect_feedback(win, thisExp, run: int, trial_num: int):
    """Show participant feedback windows for quality and connectedness ratings."""
    # Quality rating
    quality_text = visual.TextStim(
        win,
        text="Rate conversation quality (1=poor, 4=high):",
        pos=(0, 50), height=30, color=(-1, -1, -1)
    )
    quality_rating = None
    while quality_rating not in ["1", "2", "3", "4"]:
        quality_text.draw()
        win.flip()
        keys = event.waitKeys(keyList=["1", "2", "3", "4"])
        if keys:
            quality_rating = keys[0]

    print(f"Conversation Quality Rating: {quality_rating}")
    thisExp.addData("Run", run)
    thisExp.addData("Trial", trial_num)
    thisExp.addData("Quality", quality_rating)

    # Connectedness rating
    connectedness_text = visual.TextStim(
        win,
        text="Rate connectedness with partner (1=disconnected, 4=connected):",
        pos=(0, 50), height=30, color=(-1, -1, -1)
    )
    connectedness_rating = None
    while connectedness_rating not in ["1", "2", "3", "4"]:
        connectedness_text.draw()
        win.flip()
        keys = event.waitKeys(keyList=["1", "2", "3", "4"])
        if keys:
            connectedness_rating = keys[0]

    print(f"Connectedness Rating: {connectedness_rating}")
    thisExp.addData("Connectedness", connectedness_rating)
