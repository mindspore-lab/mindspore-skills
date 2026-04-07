from pathlib import Path
import json
import logging
import sys
import time

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class StepMetricsCallback(TrainerCallback):
    def __init__(self):
        self.step_start_time = None
        self.step_times = []
        self.records = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start_time is None:
            return
        step_time = time.perf_counter() - self.step_start_time
        self.step_times.append(step_time)
        self.step_start_time = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return

        step = state.global_step
        if step <= 0 or len(self.step_times) < step:
            return

        step_time = self.step_times[step - 1]
        steps_per_second = 1.0 / step_time if step_time > 0 else None

        self.records.append(
            {
                "step": step,
                "loss": logs["loss"],
                "train_steps_per_second": round(steps_per_second, 6) if steps_per_second is not None else None,
            }
        )


output_dir = Path("qwen3-finetuned")
output_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    force=True,
)

logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info("Loading dataset...")
dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")


def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )


logger.info("Tokenizing dataset...")
dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)

logger.info("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id
model = model.to("npu")

training_args = TrainingArguments(
    output_dir=str(output_dir),
    max_steps=20,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    gradient_checkpointing=False,
    bf16=True,
    learning_rate=2e-5,
    logging_strategy="steps",
    logging_steps=1,
    logging_first_step=True,
    report_to="none",
    log_level="info",
    eval_strategy="no",
    save_strategy="no",
    disable_tqdm=False,
    skip_memory_metrics=True,
)

metrics_callback = StepMetricsCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[metrics_callback],
)

logger.info("Starting training...")
trainer.train()

with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics_callback.records, f, ensure_ascii=False, indent=2)

logger.info("Training finished.")
logger.info("Saved step metrics to %s", (output_dir / "metrics.json").resolve())
