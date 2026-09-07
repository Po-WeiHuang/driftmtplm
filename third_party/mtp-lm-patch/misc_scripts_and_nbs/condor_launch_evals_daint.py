"""Generates the HTCondor eval-sweep files under condor/:

    condor/eval_reproduce_sweep.sh     (executable; parameterized by
                                         TASK, K_TOKS, STRATEGY args)
    condor/eval_reproduce_sweep.sub    (queues one job per row of the
                                         .items file below)
    condor/eval_reproduce_sweep.items  (task,k_toks,strategy combinations)

This is the HTCondor analogue of misc_scripts_and_nbs/launch_evals_daint.py
in mtp-lm (which targets CSCS Daint's SLURM + launch_daint.py instead), for
sweeping the reproduce_full checkpoint's lm-eval-harness accuracy over
multiple tasks and multiple MTP (k_toks, strategy) settings. Edit the config
below and rerun this script to regenerate all three files; do not hand-edit
the generated files directly.
"""
import csv
import argparse
import os
import shutil
import stat
import subprocess
from itertools import product

# fmt: off

CONDOR_DIR = "/home/huangp/aiproj/driftmtplm/condor"
# the generated .sh/.sub/.items live in condor/eval/; logs stay in condor/logs/.
EVAL_DIR = os.path.join(CONDOR_DIR, "eval")
LOG_DIR = os.path.join(CONDOR_DIR, "logs")

# if True, condor_submit the generated .sub after writing it. Left False by
# default so you can inspect condor/eval_reproduce_sweep.items first.
SUBMIT = False

# --- fixed across the whole sweep (same checkpoint being reproduced as
# condor/eval_reproduce_gsm8k.sh) ---
CKPT_DIR = "/data/snoplus/weiiiiiii/aiproj/driftmtplm/outputs/reproduce/step-00060000"
STEP_NUM = 60000
RUN_NAME_BASE = "l3_magpie_metamath"
EVAL_OUTPUT_BASE = "/data/snoplus/weiiiiiii/aiproj/driftmtplm/outputs/lm_eval/reproduce_step-00060000"
ACCELERATE_CONFIG = "/home/huangp/aiproj/driftmtplm/third_party/lm-evaluation-harness-mtp-lm-patch/evaluation/config_hub/accelerate_config_1N.yaml"
DEFAULT_MTP_CFG = "/home/huangp/aiproj/driftmtplm/third_party/lm-evaluation-harness-mtp-lm-patch/evaluation/config_hub/default_mtp.yaml"

parser = argparse.ArgumentParser(
    description="Generate and optionally submit the lm-eval HTCondor sweep."
)
parser.add_argument("--ckpt-dir", default=CKPT_DIR)
parser.add_argument("--step-num", type=int, default=STEP_NUM)
parser.add_argument("--run-name-base", default=RUN_NAME_BASE)
parser.add_argument("--eval-output-base", default=EVAL_OUTPUT_BASE)
parser.add_argument("--accelerate-config", default=ACCELERATE_CONFIG)
parser.add_argument("--default-mtp-cfg", default=DEFAULT_MTP_CFG)
parser.add_argument(
    "--save_condordir",
    default=None,
    help="Also copy the generated .sh, .sub, and .items files into this directory.",
)
args = parser.parse_args()

# Keep the constants above as the documented defaults while allowing callers
# to override every checkpoint/evaluation path and run metadata from the CLI.
CKPT_DIR = args.ckpt_dir
STEP_NUM = args.step_num
RUN_NAME_BASE = args.run_name_base
EVAL_OUTPUT_BASE = args.eval_output_base
ACCELERATE_CONFIG = args.accelerate_config
DEFAULT_MTP_CFG = args.default_mtp_cfg

# --- swept over ---
TASKS = [
    "gsm8k_cot_singleshot",
    "aime25",
    "bbh_cot_fewshot",
    # "ifeval",
    # "gpqa_main_cot_n_shot",
]

# --- per-task `until` stop strings ---
# CLI --gen_kwargs REPLACES (not merges with) the gen_kwargs mapping in
# default_mtp.yaml -- EvaluatorConfig.from_cli does a flat dict.update of CLI
# args over the YAML config -- so the job script has to re-emit every MTP
# gen_kwarg itself, `until` included. Values copied from
# mtp-lm/misc_scripts_and_nbs/launch_evals_daint.py's task list.
EOS_UNTIL = "</s>+<|end_of_text|>+<|eot_id|>+<|endoftext|>+<|im_end|>"
UNTIL_BY_TASK = {
    # cot tasks stop at the next "Q:" so the model doesn't roll on into a
    # self-generated follow-up question; bbh's 3-shot prompt is blank-line
    # separated, so it also stops at "\n\n" (the literal backslash-n pair is
    # what lm_eval's gen_kwargs parser unescapes into a real newline).
    "gsm8k_cot_singleshot": f"Q:+{EOS_UNTIL}",
    "bbh_cot_fewshot": f"\\n\\n+Q:+{EOS_UNTIL}",
}
# everything else (aime25, ...) just stops at the EOS-ish tokens
DEFAULT_UNTIL = EOS_UNTIL

# the non-swept MTP gen_kwargs from default_mtp.yaml, minus `until`
BASE_GEN_KWARGS = (
    "do_sample=False,do_mtp=True,include_prompt=True,"
    "return_mtp_result_dict=True,mask_id=128259,eos_id=128009+128001"
)

# (k_toks, strategy) pairs. strategy=None means static/full-k acceptance
# (matches default_mtp.yaml's default). "conf_adapt+<threshold>" is the
# confidence-adaptive acceptance strategy -- see
# third_party/lm-evaluation-harness-mtp-lm-patch/README.md's "Accept: static
# vs. conf_adapt" section. Grid copied from launch_evals_daint.py's sweep.
K_STRATEGY = [
    (1, None),
    (2, None),
    (3, None),
    (4, None),
    (5, None),
    (16, "conf_adapt+0.995"),
    (16, "conf_adapt+0.99"),
    (16, "conf_adapt+0.98"),
    (16, "conf_adapt+0.97"),
    (16, "conf_adapt+0.96"),
    (16, "conf_adapt+0.95"),
    (16, "conf_adapt+0.9"),
    (16, "conf_adapt+0.87"),
    (16, "conf_adapt+0.85"),
    (16, "conf_adapt+0.80"),
    (16, "conf_adapt+0.75"),
    (16, "conf_adapt+0.70"),
    (16, "conf_adapt+0.65"),
    (16, "conf_adapt+0.6"),
]

REQUEST_GPUS = 1
REQUEST_CPUS = 4
REQUEST_MEMORY = "32G"

# fmt: on

SH_PATH = os.path.join(EVAL_DIR, "eval_reproduce_sweep.sh")
SUB_PATH = os.path.join(EVAL_DIR, "eval_reproduce_sweep.sub")
ITEMS_PATH = os.path.join(EVAL_DIR, "eval_reproduce_sweep.items")

UNTIL_CASES = "".join(
    f'    {task}) UNTIL="{until}" ;;\n' for task, until in UNTIL_BY_TASK.items()
)

SH_TEMPLATE = f"""\
#!/usr/bin/env bash
# Executable launched by HTCondor for the reproduce_full checkpoint's MTP
# lm-eval-harness sweep over tasks x (k_toks, strategy) (single GPU,
# batch_size=1 - MTP generation doesn't support batching).
#
# Generated by third_party/mtp-lm-patch/misc_scripts_and_nbs/condor_launch_evals_daint.py
# -- do not hand-edit, rerun that script instead.
set -euo pipefail

TASK="$1"
K_TOKS="$2"
STRATEGY="$3"   # "none" for no strategy override (static/full-k acceptance)

export RUN_NAME={RUN_NAME_BASE}
export STEP_NUM={STEP_NUM}

source /data/snoplus/weiiiiiii/aiproj/driftmtplm/miniforge/etc/profile.d/conda.sh
conda activate /data/snoplus/weiiiiiii/aiproj/driftmtplm/.venv

STRAT_TAG="${{STRATEGY//+/@}}"
: "${{EVAL_OUTPUT_DIR:={EVAL_OUTPUT_BASE}/${{TASK}}_k${{K_TOKS}}_strat${{STRAT_TAG}}}}"
# Explicitly override model_args.pretrained from default_mtp.yaml so the
# generated script evaluates the checkpoint selected by --ckpt-dir.
: "${{CKPT_DIR:={CKPT_DIR}}}"

# `until` depends on the task's prompt format -- see UNTIL_BY_TASK in the
# generator. The full MTP gen_kwargs set has to be repeated here because a CLI
# --gen_kwargs replaces default_mtp.yaml's gen_kwargs mapping wholesale rather
# than merging into it.
case "$TASK" in
{UNTIL_CASES}    *) UNTIL="{DEFAULT_UNTIL}" ;;
esac

GEN_KWARGS="{BASE_GEN_KWARGS},until=${{UNTIL}},k_toks=${{K_TOKS}}"
if [ "$STRATEGY" != "none" ]; then
    GEN_KWARGS="${{GEN_KWARGS}},strategy=${{STRATEGY}}"
fi

accelerate launch --config_file {ACCELERATE_CONFIG} -m lm_eval run \\
    --config {DEFAULT_MTP_CFG} \\
    --model_args pretrained="${{CKPT_DIR}}",dtype=float32 \\
    --tasks "${{TASK}}" \\
    --gen_kwargs "${{GEN_KWARGS}}" \\
    --apply_chat_template \\
    --fewshot_as_multiturn \\
    --output_path "${{EVAL_OUTPUT_DIR}}"

exec python -u /home/huangp/aiproj/driftmtplm/third_party/mtp-lm/litgpt/scripts/push_lmeval_metrics_to_wandb.py \\
--run_dir "${{EVAL_OUTPUT_DIR}}" \\
--hf_tokenizer_path "${{CKPT_DIR}}" \\
--wandb_args name=${{RUN_NAME}}_${{TASK}}_k${{K_TOKS}}_strat${{STRAT_TAG}},project=singleshot-evals,step=${{STEP_NUM}},tags=condor+stepwise+manual_pusher \\
--dry_run=False
"""

SUB_TEMPLATE = f"""\
universe   = vanilla
executable = {SH_PATH}

# --- GPU / resource requests ---
request_gpus   = {REQUEST_GPUS}
request_cpus   = {REQUEST_CPUS}
request_memory = {REQUEST_MEMORY}

# Same resource profile as eval_reproduce_gsm8k.sh: single 8B checkpoint, no
# teacher model, and MTP generation forces batch_size=1 regardless of task or
# k_toks/strategy, so every job in this sweep fits on any pool GPU (A100 or
# H100). See condor/gpu_status.sh to check current GPU availability.

# --- environment ---
getenv     = false
should_transfer_files = NO
# paths in the job script live under /data/snoplus, which is on shared
# storage mounted on the execute nodes, so no file transfer is needed.

arguments = "$(task) $(k_toks) $(strategy)"

# --- logging ---
log    = {LOG_DIR}/eval_reproduce_sweep.$(task)_k$(k_toks)_strat$(strategy).$(Cluster).log
output = {LOG_DIR}/eval_reproduce_sweep.$(task)_k$(k_toks)_strat$(strategy).$(Cluster).out
error  = {LOG_DIR}/eval_reproduce_sweep.$(task)_k$(k_toks)_strat$(strategy).$(Cluster).err

# One job per (task, k_toks, strategy) combination, listed in
# eval_reproduce_sweep.items. Regenerate both files by rerunning
# third_party/mtp-lm-patch/misc_scripts_and_nbs/condor_launch_evals_daint.py.
queue task,k_toks,strategy from eval_reproduce_sweep.items
"""

combos = [
    (task, k_toks, strategy if strategy is not None else "none")
    for task, (k_toks, strategy) in product(TASKS, K_STRATEGY)
]

os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

with open(SH_PATH, "w") as f:
    f.write(SH_TEMPLATE)
os.chmod(SH_PATH, os.stat(SH_PATH).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

with open(ITEMS_PATH, "w") as f:
    writer = csv.writer(f)
    for row in combos:
        writer.writerow(row)

with open(SUB_PATH, "w") as f:
    f.write(SUB_TEMPLATE)

if args.save_condordir is not None:
    os.makedirs(args.save_condordir, exist_ok=True)
    for generated_path in (SH_PATH, SUB_PATH, ITEMS_PATH):
        shutil.copy2(generated_path, args.save_condordir)
    print(f"copied generated Condor files to {args.save_condordir}")

print(f"wrote {SH_PATH}")
print(f"wrote {ITEMS_PATH} ({len(combos)} jobs)")
print(f"wrote {SUB_PATH}")
for row in combos:
    print("  ", row)

if SUBMIT:
    subprocess.run(["condor_submit", SUB_PATH], cwd=EVAL_DIR, check=True)
else:
    print(f"SUBMIT=False, not submitting. To submit: condor_submit {SUB_PATH}")
