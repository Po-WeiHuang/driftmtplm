# `lm-evaluation-harness-mtp-lm` — 介紹與使用指南

## 1. 這是什麼

[`third_party/lm-evaluation-harness-mtp-lm`](../lm-evaluation-harness-mtp-lm) 是
[`jwkirchenbauer/lm-evaluation-harness-mtp-lm`](https://github.com/jwkirchenbauer/lm-evaluation-harness-mtp-lm)
的一份 vendored clone，它是 [EleutherAI 的 `lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
的一個 fork，只加了最少量的修補，用來讓標準的評測迴圈能夠驅動
[`third_party/mtp-lm`](../mtp-lm)（`litgpt/transformers_local/{llama,qwen3}/modeling_*.py`）裡實作的
**MTP（Multi-Token Prediction，多詞元預測）生成**——一個非自迴歸（non-autoregressive）的自訂 `generate()`
路徑——並把 MTP 特有的推論統計數據（tokens/sec、平均接受的 `k`、forward-eval 次數……）輸出到
`outputs/**/*.jsonl` 與 W&B。

它**不是**一組自己的評測任務或指標——任務定義（`gsm8k`、`lambada_openai` 等）、prompt 模板、指標實作全都是上游
EleutherAI 的原始碼。這個 fork 只動了以下幾個檔案：

| 檔案 | 加了什麼 |
|---|---|
| [`lm_eval/models/huggingface.py`](../lm-evaluation-harness-mtp-lm/lm_eval/models/huggingface.py) | 讓程式可以接受 `model.generate()` 回傳一個 `dict`（而不只是 `Tensor`），把它拆開、照常把 `token_ids` decode 成文字，並把 MTP 的附加欄位重新掛回結果上，供後續記錄使用。 |
| [`lm_eval/evaluator.py`](../lm-evaluation-harness-mtp-lm/lm_eval/evaluator.py) | 在計分**之前**先把 MTP 附加欄位從每個回應中剝離（讓 `process_results` 完全看不到它們），並把它們存進 `log_samples` 輸出裡的 `example["mtp_results"]`，寫進 JSONL。 |
| [`lm_eval/models/sglang_causallms.py`](../lm-evaluation-harness-mtp-lm/lm_eval/models/sglang_causallms.py) | 對 sglang-server backend 做等價的處理。 |
| `lm_eval/_cli/*`（參數解析） | 支援用 list / `+` 分隔的方式輸入複合型的 `gen_kwargs`，例如 `strategy=["conf_adapt",0.9]` 或 `eos_id=128009+128001`。 |

這個資料夾（`lm-evaluation-harness-mtp-lm-patch`）是 [`third_party/mtp-lm-patch`](../mtp-lm-patch) 的姊妹目錄——
也就是說，這裡是用來放這個 vendored harness 專屬的本機 patch／設定／腳本的地方，就像 `mtp-lm-patch`
在 `third_party/mtp-lm` 之上疊加編號的 `.patch` 檔和 `config_hub/` 覆寫一樣。除了這份指南以外，目前是空的。

下面提到的 MTP `gen_kwargs` 預設值與 `accelerate` 設定檔，實際上都放在
[`third_party/mtp-lm/config_hub/lm_eval/`](../mtp-lm/config_hub/lm_eval)，例如 `default_mtp.yaml`、
`gsm8k_mtp.yaml`、`gsm8k_mtp8_ca90.yaml`、`accelerate_config_1N.yaml`。

## 2. 快速上手

### 安裝

```bash
# （在一個已經先 `pip install -e '.[all]'` 過 third_party/mtp-lm 的環境裡）
pip install -e third_party/lm-evaluation-harness-mtp-lm
```

這會註冊兩個等價的 console entry point：`lm-eval` 與 `lm_eval`，都定義在
[`pyproject.toml`](../lm-evaluation-harness-mtp-lm/pyproject.toml) 裡指向
`lm_eval._cli.harness:cli_evaluate`（也可以用 `python -m lm_eval` 執行）。CLI 有三個子命令
（`lm_eval/_cli/{run,ls,validate}.py`）：`run`（執行評測）、`ls`（列出任務）、`validate`（檢查任務設定）。
如果你省略子命令（例如直接 `lm_eval --model hf ...`），系統會自動幫你補上 `run` 以維持向下相容
（`lm_eval/_cli/harness.py:48-51`）。

### 最小非 MTP 煙霧測試

```bash
lm_eval run --model hf --model_args pretrained=gpt2 --tasks lambada_openai --batch_size 8
```

### 單一 process 的 MTP 執行

```bash
lm_eval run \
  --config third_party/mtp-lm/config_hub/lm_eval/gsm8k_mtp.yaml \
  --tasks gsm8k
```

`gsm8k_mtp.yaml` 設定了 `model: hf`、`model_args.trust_remote_code: true`、`batch_size: 1`，以及下面會說明的
MTP `gen_kwargs` 區塊（`do_mtp: true`、`k_toks`、`mask_id`、`strategy`、`eos_id`、`include_prompt`）。CLI 參數
（`--tasks`、`--model_args`、`--gen_kwargs` 等）會覆寫／合併到 `--config` 所設定的內容之上——細節見
`lm_eval/config/evaluate_config.py`。

## 3. 各個環節分別在哪裡發生

### 3a. Prompt 組裝（Prompt formulation）

這一整段都是上游原生的 harness 程式碼（`lm_eval/api/task.py`，`ConfigurableTask` 類別），MTP 這個 patch
完全沒有動它：

- **`ConfigurableTask.fewshot_context`**（`lm_eval/api/task.py:927`）負責組出完整的 prompt：先放 system
  instruction／description，接著透過 `self.sampler` 取樣出 `num_fewshot` 個 few-shot 的問答輪次，並用
  `doc_to_text` / `doc_to_target` / `build_qa_turn`（約 `task.py:975-1010`）渲染出來，最後才接上待評測文件
  本身的問題（不含答案）。可以渲染成純文字，或在加上 `--apply_chat_template` 後透過 tokenizer 的
  chat template 渲染（見 `HFLM.apply_chat_template`，`models/huggingface.py:1548`）。
- **`ConfigurableTask.doc_to_text` / `doc_to_target` / `doc_to_choice`**（`task.py:1191` 附近）負責從該任務的
  YAML（例如 `lm_eval/tasks/gsm8k/gsm8k.yaml`）解析出對應的 Jinja／格式化字串欄位。
- **`ConfigurableTask.construct_requests`**（`task.py:1353`）把 `(doc, ctx)` 轉成 `Instance` 物件。對於
  `generate_until` 類型的任務（也就是 MTP 執行時用的類型），arguments tuple 的組裝發生在
  **`task.py:1394`**：`arguments = (ctx, deepcopy(self.config.generation_kwargs))`——這正是任務（或 CLI）
  的 `gen_kwargs`（包含 `do_mtp`）被附加到每一個 request 上的確切位置。

### 3b. 模型生成（tokenize → 補上 MTP 遮罩 → 前向傳播 → 接受策略 → decode 並輸出統計）

這才是這個 fork 真正動到的部分，橫跨 harness（`HFLM`）與模型專案本身（`mtp-lm`）。

| 步驟 | 檔案 : 行數 |
|---|---|
| **Tokenize** 整批 context | `HFLM.generate_until` 裡呼叫 `tok_batch_encode()`，位於 [`lm_eval/models/huggingface.py:1456`](../lm-evaluation-harness-mtp-lm/lm_eval/models/huggingface.py#L1456) |
| **呼叫模型的 `generate()`** | `HFLM._model_generate`（[`huggingface.py:970-1006`](../lm-evaluation-harness-mtp-lm/lm_eval/models/huggingface.py#L970)）呼叫 `self.model.generate(input_ids=context, ..., **generation_kwargs)`，其中 `generation_kwargs` 仍然帶著任務 `gen_kwargs` 裡的 `do_mtp`、`k_toks`、`mask_id`、`strategy`、`eos_id`、`include_prompt`、`return_mtp_result_dict`。 |
| **走標準路徑還是 MTP 路徑** | 因為該 checkpoint 的 `config.json` 是用有註冊 `trust_remote_code` 的自訂類別推上 Hub 的（`LlamaForCausalLM.register_for_auto_class(...)`，見 [`mtp-lm/litgpt/transformers_local/llama/modeling_llama.py:957-960`](../mtp-lm/litgpt/transformers_local/llama/modeling_llama.py#L957)），所以 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` 載入的其實是*那個*類別被覆寫過的 `.generate()`（[`modeling_llama.py:501`](../mtp-lm/litgpt/transformers_local/llama/modeling_llama.py#L501)）。它會檢查 `do_mtp`、嚴格驗證／過濾 kwargs，然後轉呼叫 `_mtp_generate`。（Qwen3 在 `modeling_qwen3.py` 裡是同樣的作法。） |
| **補上 MTP 遮罩位置** | `_extend_w_mask`（[`modeling_llama.py:855`](../mtp-lm/litgpt/transformers_local/llama/modeling_llama.py#L855)），在 `k_toks > 1` 時，於 `_mtp_generate` 迴圈內每一輪都會呼叫（`modeling_llama.py:665`）。它會在目前序列後面補上 `k_toks - 1` 個佔位／遮罩 token（用 `mask_id`，或是一個範圍 `[min_mask_id, max_mask_id)`）——這些就是模型在單次前向傳播中要一併「填空」的額外槽位，也就是多詞元預測（multi-token prediction）真正的機制所在。 |
| **執行前向傳播** | `_mtp_next_tokens`（[`modeling_llama.py:877`](../mtp-lm/litgpt/transformers_local/llama/modeling_llama.py#L877)）呼叫 `outputs = model(**model_inputs)`——單次前向傳播就同時算出 prompt 加上所有遮罩槽位、共 `k_toks` 個位置的 logits，並沿用跨迴圈迭代保留的 KV cache（`model_kwargs["past_key_values"]`，每一步結束後會被裁切）。 |
| **接受策略：static 或 `conf_adapt`** | 同一個函式，`modeling_llama.py:887-935`。`strategy is None`（即「static」）**永遠接受全部 `k_toks` 個預測出的 token**，直接取 `argmax`（`:888`）。`strategy[0] == "conf_adapt"` 則會先算出每個位置 top-1 的 softmax 信心值（`_top1_confidence`，`:845`），只接受**信心值持續 ≥ `strategy[1]`（閾值）的最長連續前綴**（`:889-927`）；`conf_adapt_sample@1` 這個變體則會在退回到 `k=1` 時改用取樣（sampling）決定最後一個 token。`strategy[0] == "random"` 則是每一步隨機抽樣要接受的 `k`。 |
| **回傳 decode 過的文字，以及生成效率統計** | `_mtp_generate` 在 [`modeling_llama.py:806-818`](../mtp-lm/litgpt/transformers_local/llama/modeling_llama.py#L806) 組出 `mtp_result_dict`——內含 `token_ids`、`tps`（每秒 token 數）、`avg_effective_k`、`effective_k_values`（每一步實際接受的 token 數）、`num_fwd_evals`、`t_prefill`、`t_gen`——**但前提是 `gen_kwargs` 裡有設定 `return_mtp_result_dict: true`**；否則它只會回傳原始的 token tensor。回到 harness 這邊，`HFLM.generate_until`（`huggingface.py:1479-1533`）會偵測到回傳的是 `dict`，把 `token_ids` 取出、用 `tok_decode` 加上 `postprocess_generated_text`（`models/utils.py:726`）decode 成文字，再把 MTP 附加欄位逐筆掛回去。接著 `evaluator.py:660-716` 會在計分前把這些附加欄位剝除（確保計分邏輯完全看不到它們），並歸檔到記錄樣本 JSONL 裡的 `example["mtp_results"]`。 |

**要注意的地方：** 目前 checked-in 的預設檔（`mtp-lm/config_hub/lm_eval/` 底下的 `default_mtp.yaml`、
`gsm8k_mtp.yaml`、`gsm8k_mtp8_ca90.yaml`）**都沒有**設定 `return_mtp_result_dict: true`——只有
`mtp-lm/README.md` 裡的範例指令有加（`--gen_kwargs ...,return_mtp_result_dict=True,...`）。沒設定的話，
MTP 生成與計分都還是會正常運作，只是不會記錄 `tps`／`avg_effective_k` 這些統計數據。如果你想要這些數據，
記得在 yaml 裡加上 `return_mtp_result_dict: true`（或用 `--gen_kwargs ...,return_mtp_result_dict=True`）。

### 3c. 計分（Scoring）

同樣是原生上游程式碼：

- **`ConfigurableTask.process_results`**（`lm_eval/api/task.py:1441`，`generate_until` 分支在 `task.py:1564`）
  用 `self._metric_fn_list` 裡的每個指標（例如 `exact_match`、`acc`）把 decode 出的文字跟
  `doc_to_target(doc)` 做比對——實作在
  [`lm_eval/api/metrics.py`](../lm-evaluation-harness-mtp-lm/lm_eval/api/metrics.py)（`exact_match_fn`、
  `acc_fn`、`f1_score`、`bleu` 等，透過 `@register_metric` 註冊）。
- 每份文件算出的指標值會累積進 `task_output.sample_metrics`，在 `evaluator.py:719-720` 的主迴圈裡完成。
- **`evaluator_utils.consolidate_results`**（`evaluator_utils.py:312`）把這些數值彙總成最終的任務層級數字，
  而 `lm_eval.utils.make_table` 則負責把結果彙整表格印出來／存進 `results.json`。

## 4. 如何讓 `accelerate launch` 跑起來

`HFLM.__init__` 一定會建立一個 `accelerate.Accelerator()`（`huggingface.py:133`），它會自動偵測目前是不是在
`accelerate` launcher 底下執行（透過 launcher 設定的環境變數）——不需要改任何程式碼，只要用正確的方式啟動就好。
以下是能正常運作、來自 `mtp-lm/README.md`「Evaluation」章節的標準指令：

```bash
accelerate launch --config_file third_party/mtp-lm/config_hub/lm_eval/accelerate_config_1N.yaml -m lm_eval run \
  --config third_party/mtp-lm/config_hub/lm_eval/default_mtp.yaml \
  --model_args pretrained=$RUN_OUTPUT_DIR/$CKPT_SUBDIR,dtype=float32 \
  --tasks gsm8k_cot_singleshot \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs do_sample=False,do_mtp=True,include_prompt=True,return_mtp_result_dict=True,"until=Q:+</s>+<|end_of_text|>+<|eot_id|>+<|endoftext|>+<|im_end|>",mask_id=128259,eos_id=128009+128001,k_toks=1 \
  --output_path $EVAL_OUTPUT_DIR
```

要讓這個指令真的跑起來，以下幾點必須對齊：

1. **`batch_size` 一定要維持 `1`。** `_mtp_generate` 會強制斷言只支援單筆生成
   （`modeling_llama.py:576-577`：`NotImplementedError("MTP generation currently only supports single-example
   generation (no batching)")`）。這在搭配 `accelerate launch` 時完全沒問題，甚至是理想的用法——因為
   `accelerate` 給你的是*資料並行*（data parallelism）：`num_processes` 個 GPU 各自載入一份完整的模型，
   各自獨立處理評測資料集裡互不重疊的一部分，而不是在同一個 process 裡做 batching。所有 checked-in 的
   MTP yaml 檔都已經設定 `batch_size: 1`。
2. **要用 `accelerate_config_1N.yaml`，不要用 `accelerate_config_fsdp_1N.yaml`。** 前者是單純的
   `MULTI_GPU`（資料並行，每個 process 一份完整模型）——這正是 MTP 生成程式碼設計時假設的情境。FSDP 模型
   切分是另一個不相關的使用情境（把一個單一 GPU裝不下的模型切分到多個 process 上）；上游 harness 的
   README 本身就警告過基本的資料並行啟動方式跟 FSDP 切分不相容，而且 `_mtp_generate` 手動操作 KV cache 的
   方式（`model_kwargs["past_key_values"].crop(...)`，`modeling_llama.py:747`）也假設整個 cache 是由單一
   process 獨自擁有。只有在模型真的裝不下單一張 GPU 時，才考慮用 `parallelize=True` / FSDP，而且要更謹慎地
   驗證結果。
3. **必須設定 `trust_remote_code: true`**（在 yaml 的 `model_args` 裡，或用
   `--model_args trust_remote_code=True`）。Hub 上的 MTP checkpoint 會連同權重一起附上自己的
   `modeling_llama.py`／`modeling_qwen3.py`（透過 §3b 提到的 `register_for_auto_class`）；`transformers`
   只有在 `trust_remote_code=True` 時才會執行那份附帶的程式碼——也才會真的認得並執行 `do_mtp=True`。
   沒開的話會悄悄地退回成原版的 `LlamaForCausalLM`／`Qwen3ForCausalLM`，任何 `do_mtp`／`k_toks`／`strategy`
   之類的參數都會被忽略（或直接報錯，因為原版 HF 的 `generate()` 根本不認得這些參數）。
4. **`do_sample` 必須維持 `False`。** MTP 路徑在 `do_sample=True` 時會直接丟出例外
   （`modeling_llama.py:554-555`）——這裡的 MTP 生成走的是貪婪／信心閾值的邏輯，而不是取樣（唯一的例外是
   `conf_adapt_sample@1` 這個變體裡，最後一個 token 內部會用取樣決定，但那跟 `do_sample` 這個參數無關）。
5. **對於任何帶有巢狀 list 的值（例如 `strategy: ["conf_adapt", 0.9]`），建議優先用 `--config <yaml>`，
   而不是手動在 CLI 上組出 `gen_kwargs`。** CLI 解析器確實支援複合型的值（用 `+` 串接的 list、
   類 JSON 的值——見 `lm_eval/_cli/utils.py`），就像上面 README 範例裡的 `until=...` 和
   `eos_id=128009+128001` 一樣，但很容易因為引號沒處理好而出錯。`mtp-lm/config_hub/lm_eval/` 底下
   checked-in 的 yaml 檔已經把 `k_toks`／`strategy`／`mask_id`／`eos_id` 正確地編碼好了——直接用
   `--config` 指到其中一個檔案，然後只覆寫每次跑會變的部分（`--model_args`、`--tasks`、`--output_path`）就好。
6. **讓 accelerate 設定檔裡的 `num_processes` 跟你實際擁有的 GPU 數量對上。**
   `accelerate_config_1N.yaml` 預設是 `num_processes: 4`（針對 4-GPU 節點調過的值）；如果你的 GPU 數量不同，
   用 `accelerate launch --num_processes N --config_file ...` 覆寫，或直接修改該 yaml。
7. **要在你用來啟動的那個環境裡安裝好這個 harness fork。** `accelerate launch -m lm_eval run ...`
   需要每一個被啟動的 process 都能 import 到 `lm_eval`——也就是要在啟用中的環境裡執行
   `pip install -e third_party/lm-evaluation-harness-mtp-lm`（以及模型端的
   `pip install -e 'third_party/mtp-lm[all]'`），做法可參考
   `mtp-lm/install_torch_210_cuda_129_singleshot.sh`。如果在 `accelerate launch` 底下看到
   `ModuleNotFoundError: No module named 'lm_eval'`，幾乎都代表你不在那個環境裡（例如用的是系統內建的
   Python，而不是專案的 `.venv`）。

如果你只是想先確認 launcher 本身有沒有接好，還不想管 MTP 的細節，可以先拿掉 `--config`／MTP 相關的
`gen_kwargs`，跑一個純粹的資料並行煙霧測試：

```bash
accelerate launch --config_file third_party/mtp-lm/config_hub/lm_eval/accelerate_config_1N.yaml -m lm_eval run \
  --model hf --model_args pretrained=gpt2 --tasks lambada_openai --batch_size 1
```

如果這個能順利跑完，且 log 裡有回報 `world_size > 1`，就代表 accelerate／launch 這一層沒問題，剩下的問題
就是 MTP 特有的部分了（上面第 1–5 點）。
