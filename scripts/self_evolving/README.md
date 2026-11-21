# scripts/self_evolving

The top level holds the HealthBench Pro pipeline. Everything else is sorted into folders by purpose.

## HealthBench Pro (top level)

| File | What it does |
|---|---|
| `generation_server.py` | Proposer server: writes tasks and rubrics into a pool, serves `/sample`, takes `/report` |
| `self_evolving_dataset.py` | verl dataset class that pulls tasks from the generation server |
| `novelty.py` | Best-of-N novelty selection for the proposer |
| `report_novelty.py` | Control vs Best-of-N comparison report |
| `analyze_proposer.py` | Offline proposer analysis from generation-server logs |
| `healthbench_gen.py`, `preprocess_healthbench_gen.py` | Offline HealthBench-style task generation, then conversion to a training parquet |
| `make_retrieval_traces.py` | Retrieval warm-start SFT traces |
| `run_checkpoint_healthbench_sweep.py`, `aggregate_healthbench_results.py`, `recompute_official_aggregations.py` | Checkpoint eval sweep and result tables |

Launchers: `train/slurm_b200_healthbench.sbatch` (training), `train/slurm_b200_eval_healthbench.sbatch`
(eval), `train/slurm_cpu_healthbench_proposer.sbatch` (proposer only, no GPU).

## Folders

| Folder | Contents |
|---|---|
| `train/` | Training launchers. Old ones are in `train/archive/` |
| `serve/` | Servers and their launchers: embedding, TRAPI proxy, generation server, vLLM, summarizer, BioBERT, CLIMB files. Old ones are in `serve/archive/` |
| `eval/` | HealthBench Pro eval at the top, plus `eval/medcalc/` and `eval/medthinkvqa/` |
| `kb/` | Knowledge base: the retrieval code the generation server uses, plus build, re-index, WikiDoc and coverage-probe scripts |
| `mimic/` | MIMIC-IV rare-disease project: preprocessing, seed balancing, HF checkpoint backup, reward-evolution harness. `mimic/eval/` has the eval sweeps, baselines, re-judging and tables; `mimic/results/` has the result files |
| `data_prep/` | Preprocessing for PubMedQA, PMC-VQA and CLIMB |
| `sft/` | SFT and distillation dataset classes, the distillation trace builder, and checks for an external on-policy-distillation teacher |
| `analysis/` | W&B run caching and summaries |
| `notes/` | Example generated questions |

Many scripts outside HealthBench Pro default to paths under `/scratch/sheng/...` from the servers they
were written on, so set their path arguments before running them on this cluster.
