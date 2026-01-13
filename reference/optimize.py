import torch
import re
import argparse
import os, sys
import pickle
import random
import yaml
import gc
from datasets import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ScoreFlow.DPOtrainer import DPOTrainer
from ScoreFlow.DPOconfig import DPOConfig
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback
from utils.llm_manager import LLMManager
import shutil
from metagpt.logs import logger
from utils.path_utils import (
    ensure_dir,
    finetuned_epoch_dir,
    preference_data_file,
    legacy_preference_data_file,
    resolve_existing_path,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def find_checkpoint_in_epoch_dir(epoch_dir, str_id):
    subdirs = [os.path.join(epoch_dir, d) for d in os.listdir(epoch_dir)
               if os.path.isdir(os.path.join(epoch_dir, d)) and d.startswith(str_id)]
    if len(subdirs) == 1:
        return subdirs[0]
    elif len(subdirs) == 0:
        raise FileNotFoundError(
            f"No checkpoint directory found in '{epoch_dir}' starting with {str_id}.")
    else:
        raise ValueError(
            f"Multiple checkpoint directories found in '{epoch_dir}', expected only one: {subdirs}")


def training(epoch, optimize_config, zcp, data_set):

    next_epoch = str(int(epoch) + 1)

    # load model from huggingface
    model = AutoModelForCausalLM.from_pretrained(
        optimize_config["model_path"],
        dtype=torch.bfloat16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(optimize_config["model_path"])

    ensure_dir(finetuned_epoch_dir(data_set, zcp, next_epoch))

    # get LoRA
    if epoch == "0":
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.01,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
    else:
        checkpoint_path = os.path.join(
            finetuned_epoch_dir(data_set, zcp, epoch), "checkpoint")
        checkpoint_path = resolve_existing_path(
            checkpoint_path,
            [os.path.join("scoreflow_workspace", "finetuned",
                          zcp, epoch, "checkpoint")],
        )
        model = PeftModel.from_pretrained(model, checkpoint_path)
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        model.train()

    # get preference data
    preference_file = resolve_existing_path(
        preference_data_file(data_set, zcp, epoch),
        [legacy_preference_data_file(zcp, epoch)],
    )
    with open(preference_file, 'rb') as file:
        data = pickle.load(file)
    random.shuffle(data)
    dataset = Dataset.from_list(data)

    dpo_args = DPOConfig(output_dir=finetuned_epoch_dir(data_set, zcp, next_epoch),
                        #  logging_steps=10,
                         logging_steps=1,  # NOTE: modified for RAPID test
                         save_steps=140000,
                         per_device_train_batch_size=1,
                         per_device_eval_batch_size=1,
                         num_train_epochs=1,  # Total 2k steps
                         #  num_train_epochs=0.01,  # NOTE: modified for RAPID test
                         use_score=True
                         )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = DPOTrainer(model=model,
                         args=dpo_args,
                         processing_class=tokenizer,
                         train_dataset=dataset
                         )
    trainer.train()


def merge(epoch, optimize_config, zcp, data_set, train_only=False):
    next_epoch = str(int(epoch) + 1)
    base_model_path = optimize_config["model_path"]
    target_dir = finetuned_epoch_dir(data_set, zcp, next_epoch)
    peft_model_path = find_checkpoint_in_epoch_dir(target_dir, "checkpoint-")
    output_model_path = target_dir + "/merged"

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    # Merge
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    target_path = os.path.join(target_dir, "checkpoint")
    if os.path.exists(target_path):
        # 如果目标文件夹已存在，先删除再移动，实现覆盖
        logger.warning(
            f"Target folder {target_path} already exists. Overwriting...")
        shutil.rmtree(target_path)
        logger.info(f"Target folder {target_path} overwritten.")
    os.rename(peft_model_path, target_path)

    # 合并后删除上上个 epoch 目录（节省空间，同时防止当前optimize有误）
    # 如果是 train_only 模式，不删除旧目录，保留所有微调结果
    # if not train_only:
    #     delete_dir = finetuned_epoch_dir(data_set, zcp, int(next_epoch) - 2)
    #     if os.path.exists(delete_dir):
    #         logger.info(f"Removing old epoch directory: {delete_dir}")
    #         shutil.rmtree(delete_dir)
    #         logger.info(f"Old epoch directory {delete_dir} removed.")
    # else:
    #     logger.info(
    #         "train_only mode: skipping deletion of old epoch directory to preserve all fine-tuning results.")


def run_optimize(epoch: int, data_set: str, zcp: str = "accuracy", train_only: bool = False):
    """供其他模块调用的优化入口，等价于原 main 的主体逻辑。"""
    epoch_str = str(epoch)

    # load model configurations
    with open("config/optimize_config.yaml", "r") as file:
        optimize_config = yaml.safe_load(file)

    os.environ["CUDA_VISIBLE_DEVICES"] = LLMManager._allocate_gpus(llm_memory=65.0)[
        "cuda_visible_devices"]

    training(epoch_str, optimize_config, zcp, data_set)

    # merge LoRA to base model
    merge(epoch_str, optimize_config, zcp, data_set, train_only=train_only)


def main():
    # input epoch(0, 1, 2...)
    parser = argparse.ArgumentParser(
        description="Process command-line arguments")
    parser.add_argument("--dataset", type=str, default="GSM8K",
                        help="Value for Dataset")
    parser.add_argument("--epoch", type=str, default="0",
                        help="Value for Epoch")
    parser.add_argument("--zcp", type=str, default="accuracy",
                        help="ZCP for evaluation")
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Train only mode: preserve all fine-tuning results (do not delete old epoch directories)",
    )

    args = parser.parse_args()

    run_optimize(
        epoch=int(args.epoch),
        data_set=args.dataset,
        zcp=args.zcp,
        train_only=args.train_only,
    )


if __name__ == "__main__":
    main()
