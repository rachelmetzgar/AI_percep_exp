# AI_percep_exp
Experiment code for AI Perception study — fMRI + real-time LLM conversations

This repository contains the code for an fMRI/behavioral experiment studying how people perceive AI versus human partners in naturalistic conversation. Participants engage in real-time spoken interactions with both humans and chatbots, while their speech is recorded, transcribed (via Whisper), and responded to by large language models (via the OpenAI API). The experiment was implemented in [PsychoPy](https://www.psychopy.org/). Version: 2024.1.0.

---

## Features
- Real-time audio recording with [`sounddevice`](https://python-sounddevice.readthedocs.io/).
- Speech-to-text transcription using [OpenAI Whisper](https://github.com/openai/whisper).
- LLM-generated conversational responses via the [OpenAI API](https://platform.openai.com/docs/).
- Customizable prompts and trial configuration (see `config/` and `utils/prompts/`).
- PsychoPy experiment flow with trial logging, event timing, and participant ratings.

---

## Repository Structure
```
AI_percep_exp/
├── main.py                 # Main experiment script
├── randomize_conds.py      # Script for generating pseudorandomized trial configs
├── utils/                  # Helper modules
│   ├── audio_utils.py
│   ├── llm_utils.py
│   ├── exp_utils.py
│   ├── stim/               # Images used in the experiment
│   └── prompts/            # Prompts for the conversations
├── config/                 # Trial configuration files (CSV)
│   └── pseudorandomized/   # (optional) Generated configs (one per participant)
├── data/                   # Output directory for audio and logs
├── environment.yml         # Clean conda environment file
├── environment.lock.yml    # Full, pinned conda environment (reproducibility)
├── requirements.txt        # pip dependencies
└── README.md               # Project description and setup instructions
```

---

## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/rachelmetzgar/AI_percep_exp.git
cd AI_percep_exp

# Recommended (clean, minimal env)
conda env create -f environment.yml
conda activate ai_percep

# Exact versions (reproducibility)
conda env create -f environment.lock.yml
conda activate ai_percep
```

Alternatively, for pip users:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the experiment from the command line:

```bash
python main.py
```

- A PsychoPy dialog will prompt for participant ID.  
- Audio will be recorded via the system default microphone.  
- Prompts are loaded from `utils/prompts/` and trial order from `config/`.  
- Data (logs, audio, transcripts) will be saved in `data/<subject>/`.

---

## Generating Condition Files

Before running the experiment, you can generate pseudorandomized condition files for each participant using:

```bash
python randomize_conds.py
```

This will create participant-specific CSV files in `config/` (or `config/pseudorandomized/` if set in the script).

---

## Procedure

- At the start of each run, the experimenter waits for the scanner trigger and presses `x` to advance.  
- Participants are presented with a topic prompt on screen and told which partner they will be speaking with (human or chatbot).  
- The participant speaks while their voice is recorded.  
- After pressing `1`, the audio segment is transcribed (Whisper) and passed to the LLM for a response, which is then played back.  
- This cycle repeats until the run time ends (~150 seconds).  
- After each conversation, participants rate conversation quality and connectedness on a 1–4 scale.  
- Data (logs, audio files, transcriptions, and ratings) are automatically saved in `data/<subject>/`.  

---

## License

This project is released under the [MIT License](LICENSE).  
You are free to use, modify, and distribute the code with attribution.

---

## Acknowledgments

- Developed as part of dissertation research on human–AI interaction and perception of mind.  
- Thanks to the PsychoPy and OpenAI teams for open-source tools that made this possible.
