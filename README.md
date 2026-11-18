# PathSnap – Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR Campus), Section H.

| Name | SRN |
|---|---|
| Samvith Vaddiparthi | PES1UG24AM468 |
| Vikas MS | PES1UG24AM469 |
| Aditya A Kirankumar | PES1UG24AM470 |
| Haneesh E | PES1UG24AM471 |

> Work in progress – full install / run / deployment instructions are added as each stage lands.
> Work split and module contract: [`docs/INTERFACE.md`](docs/INTERFACE.md).

## Quick start (development)
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # macOS: brew install libomp
export PATHSNAP_DATA="$HOME/Downloads/Student folder"               # contains train/ and student_test/
.venv/bin/python scripts/01_inventory.py                            # Stage 1 dataset inventory
```

## AI-assistance disclosure
AI coding assistants were used during development (code scaffolding, debugging and drafting).
All experiments were actually executed by the team; every reported number is produced by the scripts in this repository.
