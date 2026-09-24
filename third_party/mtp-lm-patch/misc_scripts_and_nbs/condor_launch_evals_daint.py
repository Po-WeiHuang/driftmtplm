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
REPO_ROOT = "/home/huangp/aiproj/driftmtplm"

# --- controlled-rollout diagnostic (docs/controlled_rollout_plan.md Phase 4) ---
# Off by default: it needs a litgpt-format student AND a frozen teacher, which a
# plain accuracy sweep does not have. Enable with --controlled-rollout and give
# it both checkpoints.
#
# NOTE these are *litgpt* checkpoint dirs (lit_model.pth + model_config.yaml) --
# NOT the HF dir that --ckpt-dir points at for `--model_args pretrained=`. They
# are different directories and conflating them fails at load.
CONTROLLED_ROLLOUT = False
LITGPT_CKPT_DIR = None
TEACHER_CKPT_DIR = None
CR_TRUNCATION_LENGTH = 160   # hparams.singleshot.truncation_length
CR_MASK_REGION_CT = 5        # hparams.singleshot.mask_region_ct -> region_width = 32
CR_MICRO_BATCH_SIZE = 32     # hparams.train.micro_batch_size
CR_OFFSET = 0                # region-grid alignment; abs(offset) < P
CR_N_BINS = 15               # ECE bins (Guo et al. 2017 convention)
CR_MAX_POOL_SAMPLES = None   # None = use the full per-horizon population
CR_SEED = 0
CR_LIMIT = None              # cap documents, for a quick smoke run

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
    "--controlled-rollout",
    action="store_true",
    default=CONTROLLED_ROLLOUT,
    help="also run the controlled-rollout diagnostic after lm_eval, into the same run dir.",
)
parser.add_argument(
    "--litgpt-ckpt-dir",
    default=LITGPT_CKPT_DIR,
    help="student litgpt checkpoint dir (lit_model.pth + model_config.yaml). "
         "NOT --ckpt-dir, which is the HF dir. Required with --controlled-rollout.",
)
parser.add_argument(
    "--teacher-ckpt-dir",
    default=TEACHER_CKPT_DIR,
    help="frozen teacher litgpt checkpoint dir. Required with --controlled-rollout.",
)
parser.add_argument("--cr-truncation-length", type=int, default=CR_TRUNCATION_LENGTH,
                    help="hparams.singleshot.truncation_length (default %(default)s)")
parser.add_argument("--cr-mask-region-ct", type=int, default=CR_MASK_REGION_CT,
                    help="hparams.singleshot.mask_region_ct (default %(default)s)")
parser.add_argument("--cr-micro-batch-size", type=int, default=CR_MICRO_BATCH_SIZE,
                    help="documents per forward pass (default %(default)s)")
parser.add_argument("--cr-offset", type=int, default=CR_OFFSET,
                    help="region-grid alignment, abs(offset) < P (default %(default)s)")
parser.add_argument("--cr-n-bins", type=int, default=CR_N_BINS,
                    help="ECE bin count (default %(default)s)")
parser.add_argument("--cr-max-pool-samples", type=int, default=CR_MAX_POOL_SAMPLES,
                    help="cap the per-horizon population for MMD/Sinkhorn (default: no cap)")
parser.add_argument("--cr-seed", type=int, default=CR_SEED,
                    help="only matters when --cr-max-pool-samples is set (default %(default)s)")
parser.add_argument("--cr-limit", type=int, default=CR_LIMIT,
                    help="evaluate only the first N documents (smoke runs)")
parser.add_argument("--repo-root", default=REPO_ROOT,
                    help="repo root, used for PYTHONPATH=<root>/src in the job script")
parser.add_argument(
    "--save_condordir",
    default=None,
    help="Write the generated .sh, .sub and .items into this directory instead of "
         "condor/eval/. The .sub's executable points at the .sh beside it, so each "
         "directory is a self-contained, independently submittable set.",
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
REPO_ROOT = args.repo_root
CONTROLLED_ROLLOUT = args.controlled_rollout
LITGPT_CKPT_DIR = args.litgpt_ckpt_dir
TEACHER_CKPT_DIR = args.teacher_ckpt_dir
CR_TRUNCATION_LENGTH = args.cr_truncation_length
CR_MASK_REGION_CT = args.cr_mask_region_ct
CR_MICRO_BATCH_SIZE = args.cr_micro_batch_size
CR_OFFSET = args.cr_offset
CR_N_BINS = args.cr_n_bins
CR_MAX_POOL_SAMPLES = args.cr_max_pool_samples
CR_SEED = args.cr_seed
CR_LIMIT = args.cr_limit

if CONTROLLED_ROLLOUT and not (LITGPT_CKPT_DIR and TEACHER_CKPT_DIR):
    parser.error(
        "--controlled-rollout needs both --litgpt-ckpt-dir and --teacher-ckpt-dir "
        "(litgpt-format dirs with lit_model.pth + model_config.yaml, not the HF --ckpt-dir)."
    )

# --- swept over ---
TASKS = [
    "gsm8k_cot_singleshot",
    #"aime25",
    #"bbh_cot_fewshot",
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
    #"bbh_cot_fewshot": f"\\n\\n+Q:+{EOS_UNTIL}",
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

# --save_condordir, when given, IS the output directory -- not a copy
# destination. Generating straight into it is what keeps the .sub's `executable`
# pointing at the .sh beside it; copying instead left the copy's `executable`
# baked to condor/eval/, so submitting it silently ran whatever model that
# directory happened to hold. It also stops a run for one checkpoint from
# overwriting condor/eval/ for another -- each model gets its own directory.
if args.save_condordir is not None:
    EVAL_DIR = args.save_condordir

SH_PATH = os.path.join(EVAL_DIR, "eval_reproduce_sweep.sh")
SUB_PATH = os.path.join(EVAL_DIR, "eval_reproduce_sweep.sub")
ITEMS_PATH = os.path.join(EVAL_DIR, "eval_reproduce_sweep.items")

# The controlled-rollout step, spliced into the job script only when enabled.
# `|| CONTROLLED_FAILED=1` is load-bearing: `set -euo pipefail` is on, so without
# it a crash here aborts the job BEFORE the pusher runs and the free-rollout
# accuracy numbers are lost. A diagnostic must never be able to destroy a
# benchmark result.
#
# $TASK / $K_TOKS / $STRATEGY / $EVAL_OUTPUT_DIR are the SAME shell variables the
# lm_eval invocation above uses -- that is what enforces "controlled rollout must
# match free rollout", with no second place to edit and get wrong.
if CONTROLLED_ROLLOUT:
    _cr_optional = ""
    if CR_MAX_POOL_SAMPLES is not None:
        _cr_optional += f" \\\n    --max-pool-samples {CR_MAX_POOL_SAMPLES}"
    if CR_LIMIT is not None:
        _cr_optional += f" \\\n    --limit {CR_LIMIT}"
    CONTROLLED_ROLLOUT_BLOCK = f"""
# --- controlled-rollout diagnostic (docs/controlled_rollout_plan.md) ---------
# Writes controlled_rollout_<timestamp>.json into the SAME $EVAL_OUTPUT_DIR that
# lm_eval wrote results_*.json into, so one pusher call logs both rollouts into
# one wandb run. NOTE --student-checkpoint is a litgpt dir, not $CKPT_DIR (HF).
CONTROLLED_FAILED=0
PYTHONPATH="{REPO_ROOT}/src:${{PYTHONPATH:-}}" python -u -m driftmtp.eval.condrollouteval \\
    --config {DEFAULT_MTP_CFG} \\
    --enabled \\
    --student-checkpoint "{LITGPT_CKPT_DIR}" \\
    --teacher-checkpoint "{TEACHER_CKPT_DIR}" \\
    --task "${{TASK}}" \\
    --k-toks "${{K_TOKS}}" \\
    --strategy "${{STRATEGY}}" \\
    --out-dir "${{EVAL_OUTPUT_DIR}}" \\
    --truncation-length {CR_TRUNCATION_LENGTH} \\
    --mask-region-ct {CR_MASK_REGION_CT} \\
    --micro-batch-size {CR_MICRO_BATCH_SIZE} \\
    --offset {CR_OFFSET} \\
    --n-bins {CR_N_BINS} \\
    --seed {CR_SEED}{_cr_optional} || CONTROLLED_FAILED=1
if [ "$CONTROLLED_FAILED" -ne 0 ]; then
    echo "WARNING: controlled rollout failed; pushing free-rollout metrics only." >&2
fi
"""
else:
    CONTROLLED_ROLLOUT_BLOCK = ""

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
{CONTROLLED_ROLLOUT_BLOCK}
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

print(f"wrote {SH_PATH}")
print(f"wrote {ITEMS_PATH} ({len(combos)} jobs)")
print(f"wrote {SUB_PATH}")
for row in combos:
    print("  ", row)

if SUBMIT:
    subprocess.run(["condor_submit", SUB_PATH], cwd=EVAL_DIR, check=True)
else:
    print(f"SUBMIT=False, not submitting. To submit: condor_submit {SUB_PATH}")
