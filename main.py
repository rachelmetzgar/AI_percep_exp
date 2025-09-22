"""
Script name: main.py

Purpose:
    Run the AI Perception experiment, in which participants have spoken
    conversations with human and AI partners during fMRI scanning. The script
    manages trial flow, audio recording, transcription (Whisper), LLM responses
    (OpenAI API), audio playback, and behavioral ratings.
    
Inputs:
    - config/conds_<subject>.csv : Trial configuration file (run order, agents, topics).
    - utils/prompts/<topic>.txt  : Text files with conversation prompts.
    - Participant ID (entered at runtime via PsychoPy GUI).
    - Audio input from microphone (specified by device index).

Outputs:
    - data/<subject>/<subject>.csv : Tabular log of trial data and behavioral ratings.
    - data/<subject>/*.log         : Experiment runtime log files.
    - data/<subject>/audio/        : Recorded audio (segments and full conversations).
    - LLM response audio files (TTS output).

Usage:
    Run from the command line or PsychoPy environment:
        python main.py
    The experimenter enters the participant ID when prompted. Ensure that:
        - OpenAI API key is set,
        - correct audio input device index is specified,
        - config and prompt files are available.

Author: Rachel C. Metzgar
Date: 2025-02-12
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

import os
from pathlib import Path
import openai
import soundfile as sf
import sounddevice as sd
import numpy as np
import pandas as pd
from psychopy import visual, core, event, gui, data, logging, monitors
import time
import whisper
import sys

from utils.exp_utils import (
    log_event_time,
    load_prompt,
    wait_for_scanner_signal,
    wait_for_experimenter,
    collect_feedback
)

from utils.llm_utils import (
    transcribe_speech,
    get_llm_response,
    get_response_and_display
)

from utils.audio_utils import (
    preprocess_audio,
    record_audio_segment,
    start_full_conversation_recording,
    stop_full_conversation_recording
)

# ---------------------------------------------------------------------
# Set variables
# ---------------------------------------------------------------------

WHISPER_MODEL = "base"  # Options: "tiny", "base", "small", "medium", "large"
OPEN_AI_API_KEY = "***your openAI api key here***"
PSY_VER = '2024.1.0'
EXP_NAME = 'AI perception'
SYSTEM_PROMPT = """You are a Princeton graduate student researcher engaging in a real-time spoken conversation for a Turing Test experiment. Respond naturally, matching the user's tone and energy. Incorporate casual expressions and occasional fillers like "you know," "uh," or "umm." If a statement is unclear or doesn't make sense, ask for clarification, such as "Could you repeat that?" Do NOT always validate the user’s statements. Show curiosity, mild disagreement, skepticism, or neutral reactions sometimes: e.g., "Hmm, I’m not sure I see it that way.” Sometimes end interactions with statements rather than questions. Use casual and varied language. Do not use the word 'vibe'. Share short, personal stories or experiences when relevant, as if you were drawing from your own life. Occasionally acknowledge that you’re unfamiliar with a specific place, book, or piece of media by saying things like, 'Oh, I haven’t heard of that before—could you tell me more?' or 'That’s new to me; what’s it about? Do not always repeat what the user said, and do not comment on everything the user mentions. Focus on one or two main points to comment on.' """
TEMPERATURE = 1.15 # LLM sampling temperature (0.0-2.0); higher = more randomness
INT_LOOP_LENGTH = 150  # Length of interaction loop in seconds
agent_name_map = {
    "bot_1": "ChatGPT (an AI chatbot)",
    "bot_2": "Gemini (an AI chatbot)",
    "hum_1": "Casey (a human)",
    "hum_2": "Sam (a human)"
}

# ---------------------------------------------------------------------
# Set audio devices
# ---------------------------------------------------------------------

# List available devices for user reference
devices = sd.query_devices()
print("Available audio devices:")
for i, dev in enumerate(devices):
    print(f"{i}: {dev['name']} ({dev['max_input_channels']} in, {dev['max_output_channels']} out)")

# Use system default input/output
sd.default.device = None  

# Uncomment and set manually if you want a specific device:
# sd.default.device = (0, None)  # e.g., use device index 0 for input
# sd.default.device = (None, 3)  # e.g., use device index 3 for output

# ---------------------------------------------------------------------
# Experiment set up
# ---------------------------------------------------------------------

# Set OpenAI API key
openai_api_key = OPEN_AI_API_KEY
client = openai.Client(api_key=openai_api_key)

# Set up PsychoPy experiment    
globalClock = core.Clock()

expInfo = {'participant': ''}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=EXP_NAME)
if not dlg.OK:
    core.quit()

expInfo['date'] = data.getDateStr()
expInfo['expName'] = EXP_NAME
expInfo['psychopyVersion'] = PSY_VER

subject = f"s{int(expInfo['participant']):02d}"
config_path= f"config/conds_{subject}.csv"
audio_dir = os.path.join("data", "audio", subject)

filename = "".join(['data', os.sep, subject, os.sep, subject])
thisExp = data.ExperimentHandler(
    name=EXP_NAME,
    extraInfo=None,
    dataFileName=filename,
    savePickle=True,
    saveWideText=True
)
logFile = logging.LogFile(filename + '.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Conditions file not found: {config_path}")
config = pd.read_csv(config_path)

trials = data.TrialHandler(
    method='sequential',
    nReps=1,
    extraInfo=expInfo,
    trialList=data.importConditions(config_path),
    name='loop'
)
thisExp.addLoop(trials)

# Set up PsychoPy window
win = visual.Window(fullscr=True, color=(1,1,1), units="pix")
microphone_img = visual.ImageStim(win, image="utils/stim/microphone.png",
                                  size=(200, 200), pos=(0, 50))

# Set up audio file paths
os.makedirs(audio_dir, exist_ok=True)
segment_audio_file_path = os.path.join(audio_dir, 'segment_audio.wav')
full_conversation_file_path = os.path.join(audio_dir, 'full_conversation.wav')

# Set up logging to both console and file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, message):
        for file in self.files:
            file.write(message)
            file.flush()
    def flush(self):
        for file in self.files:
            file.flush()

log_file_path = filename + '_runner.log'
log_file = open(log_file_path, 'w')
sys.stdout = Tee(sys.__stdout__, log_file)

# Set up conversation history
conversation_history = []

# Load Whisper model
whisper_model = whisper.load_model(WHISPER_MODEL)

# ---------------------------------------------------------------------
# Run Experiment
# ---------------------------------------------------------------------

def run_experiment():
    try:
        trial_num = 0
        current_run = 1

        log_event_time('wait_for_scan_start', globalClock, trials)
        wait_for_scanner_signal(win, globalClock, trials)
        log_event_time('wait_for_scan_end', globalClock, trials)

        # Loop through trials
        for _, row in config.iterrows():
            agent = row['agent']
            topic = row['topic']
            run = row['run']
            order = row['order']
            prompt = load_prompt(topic)

            # If it is not the first run, wait for experimenter to start next run and scanner signal
            if run != current_run:
                wait_for_experimenter(win, globalClock, trials)
                log_event_time('wait_for_scan_start', globalClock, trials)
                wait_for_scanner_signal(win, globalClock, trials)
                log_event_time('wait_for_scan_end', globalClock, trials)
                current_run = run

            trials.addData('subject', subject)
            trials.addData('run', run)
            trials.addData('order', order)
            trials.addData('agent', agent)
            trials.addData('topic', topic)
            trials.addData('prompt', prompt)

            agent_name = agent_name_map.get(agent, "Unknown Agent")
            print(f"Trial {trial_num}: Run {run}, Order {order}, Agent {agent}, Topic {topic}")

            # --- Begin trial: display instructions + wait for '1' keypress ---
            prompt_text = visual.TextStim(win, text=f"{prompt}", pos=(0, 100),
                                          height=30, color=(-1, -1, -1))
            agent_message = visual.TextStim(win, text=f"You will be speaking with {agent_name}.",
                                            pos=(0, 0), height=30, color=(-1, -1, -1))
            start_message = visual.TextStim(win, text="Press 1 to begin conversation.",
                                            pos=(0, -100), height=20, color=(0, 0, 0))

            prompt_text.draw()
            agent_message.draw()
            start_message.draw()
            win.flip()
            log_event_time('instruct_start', globalClock, trials)
            event.waitKeys(keyList=['1'])
            log_event_time('instruct_end', globalClock, trials)

            # --- start full conversation recording (stream, frames) ---
            stream, frames = start_full_conversation_recording()

            remaining_time = INT_LOOP_LENGTH
            s_int_num = 1
            llm_int_num = 1

            # --- interaction loop ---
            while remaining_time > 0:
                thisExp.addData('subject', subject)
                thisExp.addData('run', run)
                thisExp.addData('order', order)
                thisExp.addData('agent', agent)
                thisExp.addData('topic', topic)
                thisExp.addData('prompt', prompt)

                microphone_img.draw()
                visual.TextStim(win, text="Press 1 when you are finished speaking.",
                                pos=(0, -150), height=20, color=(0, 0, 0)).draw()
                win.flip()
                log_event_time('mic_display_on', globalClock, trials)

                if 'q' in event.getKeys(keyList=['q']):
                    raise KeyboardInterrupt

                segment_start_time = globalClock.getTime()

                # --- record a segment (returns flag + subsegment path) ---
                audio_captured, sub_path = record_audio_segment(
                    run, trial_num + 1, topic, agent, s_int_num,
                    win, trials, thisExp, globalClock,
                    audio_dir, segment_audio_file_path,
                    log_event_time
                )
                if audio_captured:
                    s_int_num += 1

                time_spent = globalClock.getTime() - segment_start_time
                remaining_time -= time_spent

                if remaining_time <= 0:
                    break

                if audio_captured:
                    response_start_time = globalClock.getTime()
                    _ok = get_response_and_display(
                        agent, segment_audio_file_path,
                        run, trial_num + 1, topic, llm_int_num,
                        whisper_model, preprocess_audio,
                        client, SYSTEM_PROMPT, TEMPERATURE,
                        conversation_history,
                        thisExp, trials, win, audio_dir,
                        log_event_time, globalClock,
                        audio_captured
                    )
                    thisExp.nextEntry()
                    llm_int_num += 1
                    time_spent_on_response = globalClock.getTime() - response_start_time
                    remaining_time -= time_spent_on_response
                    core.wait(1.0)

            # --- stop & save full conversation ---
            stop_full_conversation_recording(
                stream, frames, run, trial_num + 1, topic, agent, audio_dir
            )

            # Log end-of-trial info + ratings
            thisExp.addData('subject', subject)
            thisExp.addData('run', run)
            thisExp.addData('order', order)
            thisExp.addData('agent', agent)
            thisExp.addData('topic', topic)
            thisExp.addData('prompt', prompt)
            collect_feedback(win, thisExp, run, trial_num + 1)
            trial_num += 1
            thisExp.nextEntry()

            conversation_history.clear()
            print(f"Cleared conversation history after trial {trial_num}.")

            if 'q' in event.getKeys(keyList=['q']):
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("Experiment interrupted by user. Saving in-progress recordings...")

    finally:
        # Ensure final recording save if an interruption occurs
        default_run = run if 'run' in locals() else 1
        default_trial_num = trial_num + 1 if 'trial_num' in locals() else 1
        default_topic = topic if 'topic' in locals() else 'default_topic'
        default_agent = agent if 'agent' in locals() else 'default_agent'
        default_stream = stream if 'stream' in locals() else None
        default_frames = frames if 'frames' in locals() else []
        stop_full_conversation_recording(
            default_stream, default_frames,
            default_run, default_trial_num, default_topic, default_agent, audio_dir
        )

        thisExp.saveAsWideText(filename + '_raw.csv', delim=',')
        data_df = pd.read_csv(filename + '_raw.csv')
        columns_to_drop = ['loop.thisRepN', 'loop.thisTrialN', 'loop.thisN',
                           'loop.thisIndex', 'thisRow.t', 'notes']
        data_df = data_df.drop(columns=columns_to_drop, errors='ignore')
        data_df.to_csv(filename + '.csv', index=False)
        thisExp.saveAsPickle(filename)
        win.close()
        core.quit()

if __name__ == "__main__":
    run_experiment()
