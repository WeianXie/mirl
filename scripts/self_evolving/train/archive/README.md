# Archived launchers

Launchers for experiments that are no longer active, moved here on 2026-11-21 to keep
`train/` down to the scripts in current use. Nothing outside `archive/` runs them.

What's here:

- Diagnosis/MCQ self-improve runs on PubMedQA, PMC-VQA and MIMIC-IV rare (`run_self_evolving*`,
  `run_multi_agent_*`, `run_qwen36_27b_selfimprove_*`, `run_qwen36_27b_gptjudge_s4.sh`, `run_baseline_grpo.sh`)
- SFT, distillation and on-policy distillation (`*_sft*`, `*_opd*`, `run_gemma4_e4b_sft.sh`)
- Reward-evolution and Stage-4 ablations (`run_qwen36_27b_evolve_reward.sh`, `run_ablation_*`)
- CLIMB multimodal runs (`run_climb_*`), Kimi runs (`launch_kimi_*`), LoRA and DAPO variants
- `run_sft_then_rl_pipeline.sh`: the unattended SFT → RL driver for server 1 (was in
  `scripts/self_evolving/`; it only runs scripts that are now in this folder)

Experiment-specific server launchers that went with these runs are in `../../serve/archive/`.

**Running one again:** these scripts call each other by their old paths
(`scripts/self_evolving/train/...`, `scripts/self_evolving/serve/...`), so they won't run from
here as they are. Move the script, and any script it calls, back with `git mv`, or check out
commit `b01884ee`, the last commit before the move.
