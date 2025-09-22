"""
Module: audio_utils.py

Purpose:
    Helper functions for recording and managing participant/LLM audio in the
    AI Perception experiment. Handles:
        - Preprocessing audio for Whisper
        - Recording participant speech segments
        - Managing continuous full-conversation recordings
        - Generating unique audio file names for saving

Functions:
    preprocess_audio(input_file, output_file)
    record_audio_segment(...)
    start_full_conversation_recording()
    stop_full_conversation_recording(...)
    get_full_conversation_filename(...)

Inputs (passed explicitly):
    - run, trial_num, topic, agent : identifiers for trial and agent
    - s_int_num                    : sub-segment index within trial
    - win                          : PsychoPy visual.Window instance
    - trials                       : PsychoPy TrialHandler (for logging metadata)
    - thisExp                      : PsychoPy ExperimentHandler (for logging)
    - globalClock                  : PsychoPy core.Clock (for timestamps)
    - audio_dir                    : path to subject's audio directory
    - segment_audio_file_path      : path for saving raw segment
    - log_event_time_fn            : function to log events with timestamps

Outputs:
    - Sub-segment audio files saved under audio_dir/trials/sub
    - Full-conversation audio files saved under audio_dir/trials
    - PsychoPy logs updated with audio metadata
    - Returns boolean flag indicating whether significant audio was captured

Usage:
    from utils.audio_utils import (
        preprocess_audio,
        record_audio_segment,
        start_full_conversation_recording,
        stop_full_conversation_recording,
        get_full_conversation_filename
    )

Author: Rachel C. Metzgar
Date: 2025-02-12
"""

import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from psychopy import visual, core, event


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def preprocess_audio(input_file: str, output_file: str) -> str:
    """Preprocess audio by ensuring it is mono and resampled to 16 kHz."""
    data, rate = sf.read(input_file)

    # Convert to mono if necessary
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Save the processed audio (Whisper will handle resampling if needed)
    sf.write(output_file, data, rate)
    return output_file


# ---------------------------------------------------------------------
# Segment recording
# ---------------------------------------------------------------------

def record_audio_segment(run: int, trial_num: int, topic: str, agent: str, s_int_num: int,
                         win, trials, thisExp, globalClock,
                         audio_dir: str, segment_audio_file_path: str,
                         log_event_time_fn):
    """
    Record a speech segment for transcription and save a sub-segment.

    Returns:
        (bool, str) : (audio_captured flag, sub_segment_audio_file_path or None)
    """
    print("Starting audio recording...")
    log_event_time_fn('sub_speech_start', globalClock, trials)

    trials_sub_dir = os.path.join(audio_dir, 'trials', 'sub')
    os.makedirs(trials_sub_dir, exist_ok=True)

    sub_segment_audio_file_path = os.path.join(
        trials_sub_dir,
        f"run{run:02d}_{trial_num}_{topic}_{agent}_{s_int_num}_sub.wav"
    )

    audio_frames = []
    with sd.InputStream(samplerate=44100, channels=1, dtype='float32', blocksize=4096) as stream:
        while True:
            frame, overflowed = stream.read(int(44100 * 0.5))
            if overflowed:
                print("Overflow! Audio data has been lost.")
            audio_frames.append(frame)

            keys = event.getKeys(keyList=['1', 'q'])
            if '1' in keys:
                log_event_time_fn('sub_speech_end', globalClock, trials)
                print("1 pressed: Stopping recording")
                win.flip()
                log_event_time_fn('mic_display_off', globalClock, trials)
                break
            if 'q' in keys:
                core.quit()

    if not audio_frames:
        print("No audio frames captured.")
        return False, None

    audio = np.vstack(audio_frames)
    audio_energy = np.sqrt(np.mean(audio**2))
    print(f"Recorded audio energy: {audio_energy}, frames: {len(audio_frames)}")

    if audio_energy > 0.01:
        sf.write(segment_audio_file_path, audio, 44100)
        core.wait(0.1)
        sf.write(sub_segment_audio_file_path, audio, 44100)
        core.wait(0.1)
        print(f"Recording saved as {sub_segment_audio_file_path}")

        trials.addData('sub_segment_name', sub_segment_audio_file_path)
        trials.addData('sub_input', segment_audio_file_path)
        return True, sub_segment_audio_file_path
    else:
        print("No significant audio detected.")
        thisExp.addData('audio_captured', 'No audio detected')
        no_audio_msg = visual.TextStim(
            win,
            text="No significant audio detected. Please try again.",
            pos=(0, 0), height=20, color=(1, -1, -1)
        )
        no_audio_msg.draw()
        win.flip()
        thisExp.nextEntry()
        core.wait(2)
        return False, None


# ---------------------------------------------------------------------
# Full conversation recording
# ---------------------------------------------------------------------

def start_full_conversation_recording():
    """
    Start recording the entire conversation.

    Returns:
        (stream, frames) : handle to the InputStream and list of audio frames
    """
    frames = []

    def callback(indata, frames_count, time_info, status):
        if status:
            print(status)
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=44100, channels=1,
                            callback=callback, blocksize=4096)
    try:
        stream.start()
        print("Started full conversation recording.")
    except Exception as e:
        print(f"Error initializing full conversation stream: {e}")
    return stream, frames


def stop_full_conversation_recording(stream, frames: list,
                                     run: int, trial_num: int, topic: str, agent: str,
                                     audio_dir: str):
    """Stop and save the full conversation recording for the current trial."""
    print("Stopping full conversation recording...")

    if stream:
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            print(f"Error stopping full conversation stream: {e}")

    if frames:
        full_conversation_data = np.vstack(frames)
        full_conversation_filename = get_full_conversation_filename(
            run, trial_num, topic, agent, audio_dir
        )
        sf.write(full_conversation_filename, full_conversation_data, 44100)
        print(f"Full conversation saved as {full_conversation_filename}")
    else:
        print("No frames recorded, skipping save.")


def get_full_conversation_filename(run: int, trial_num: int, topic: str, agent: str,
                                   audio_dir: str) -> str:
    """Generate a filename for each full conversation recording."""
    trials_dir = os.path.join(audio_dir, 'trials')
    os.makedirs(trials_dir, exist_ok=True)
    return os.path.join(trials_dir, f"run{run:02d}_{trial_num}_{topic}_{agent}.wav")
