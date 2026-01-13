from metagpt.logs import logger
from utils.llm_manager import get_global_llm_manager
from evaluate import run_evaluate
from generate import run_generate
import argparse
import asyncio
import os
import subprocess
import sys
from termcolor import cprint

os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


async def run_inference_epoch(
    dataset: str,
    epoch: int,
    zcp: str,
    limit: int | None,
    full_infer: bool,
    agent_max_output_tokens: int,
) -> None:
    """单个 epoch 的 inference：generate + evaluate。"""
    effective_limit = None if full_infer else limit
    cprint(
        f"[ScoreFlowRunner] {'='*10} Inference epoch {epoch} {'='*10}", color="blue")

    cprint(
        f"[ScoreFlowRunner] Start Generate | Epoch {epoch} | Task inference | Dataset {dataset} | ZCP {zcp} | Limit {effective_limit} | Agent Max Output Tokens {agent_max_output_tokens}", color="blue")
    await run_generate(
        data_set=dataset,
        task_type="inference",
        epoch=epoch,
        data_limit=effective_limit,
        zcp=zcp,
        agent_max_output_tokens=agent_max_output_tokens,
    )

    cprint(
        f"[ScoreFlowRunner] Start Evaluate | Epoch {epoch} | Task inference | Dataset {dataset} | ZCP {zcp} | Limit {effective_limit} | Agent Max Output Tokens {agent_max_output_tokens}", color="blue")
    await run_evaluate(
        data_set=dataset,
        task_type="inference",
        epoch=epoch,
        limit=effective_limit,
        zcp=zcp,
        agent_max_output_tokens=agent_max_output_tokens,
    )


async def run_optimize_epoch(
    dataset: str,
    epoch: int,
    zcp: str,
    limit: int | None,
    agent_max_output_tokens: int,
    train_only: bool = False,
) -> None:
    """单个 epoch 的 optimize：generate + evaluate + DPO 优化。"""
    cprint(f"[ScoreFlowRunner] Optimize epoch {epoch}", color="blue")

    cprint(
        f"[ScoreFlowRunner] Start Generate | Epoch {epoch} | Task optimize | Dataset {dataset} | ZCP {zcp} | Limit {limit} | Agent Max Output Tokens {agent_max_output_tokens}", color="blue")
    await run_generate(
        data_set=dataset,
        task_type="optimize",
        epoch=epoch,
        data_limit=limit,
        zcp=zcp,
        agent_max_output_tokens=agent_max_output_tokens,
    )

    cprint(
        f"[ScoreFlowRunner] Start Evaluate | Epoch {epoch} | Task optimize | Dataset {dataset} | ZCP {zcp} | Limit {limit} | Agent Max Output Tokens {agent_max_output_tokens}", color="blue")
    await run_evaluate(
        data_set=dataset,
        task_type="optimize",
        epoch=epoch,
        limit=limit,
        zcp=zcp,
        agent_max_output_tokens=agent_max_output_tokens,
    )
    get_global_llm_manager().clear_all()
    cprint("All LLM models have been cleared!", color="green")
    logger.info("All LLM models have been cleared!")

    cprint(
        f"[ScoreFlowRunner] Start Optimize | Epoch {epoch} | ZCP {zcp}", color="blue")
    # 在独立子进程中运行优化，避免主进程持有CUDA上下文
    optimize_script = os.path.join(os.path.dirname(__file__), "optimize.py")
    cmd = [
        sys.executable,
        optimize_script,
        "--dataset",
        dataset,
        "--epoch",
        str(epoch),
        "--zcp",
        zcp,
    ]
    if train_only:
        cmd.append("--train_only")
    subprocess.run(cmd, check=True)


async def run_scoreflow_pipeline(
    dataset: str,
    start_epoch: int,
    max_epoch: int,
    zcp: str,
    limit: int | None,
    full_infer: bool,
    agent_max_output_tokens: int,
    infer_only: bool = False,
    train_only: bool = False,
) -> None:
    """完整的 ScoreFlow epoch 循环，参考 scripts/main.sh 的逻辑。"""
    mode_str = "Inference Only" if infer_only else (
        "Train Only" if train_only else "Full Pipeline")
    cprint(
        f"[ScoreFlowRunner] Start ScoreFlow | Mode={mode_str} | Dataset={dataset} | Epoch={start_epoch}→{max_epoch} | ZCP={zcp}",
        color="green",
    )

    if infer_only:
        # 只执行 inference
        for epoch in range(start_epoch, max_epoch + 1):
            await run_inference_epoch(
                dataset=dataset,
                epoch=epoch,
                zcp=zcp,
                limit=limit,
                full_infer=full_infer,
                agent_max_output_tokens=agent_max_output_tokens,
            )
    elif train_only:
        # 只执行 optimize (train)
        for epoch in range(start_epoch, max_epoch + 1):
            await run_optimize_epoch(
                dataset=dataset,
                epoch=epoch,
                zcp=zcp,
                limit=limit,
                agent_max_output_tokens=agent_max_output_tokens,
                train_only=train_only,
            )
    else:
        # 完整的 pipeline：inference + optimize
        for epoch in range(start_epoch, max_epoch):
            await run_inference_epoch(
                dataset=dataset,
                epoch=epoch,
                zcp=zcp,
                limit=limit,
                full_infer=full_infer,
                agent_max_output_tokens=agent_max_output_tokens,
            )
            await run_optimize_epoch(
                dataset=dataset,
                epoch=epoch,
                zcp=zcp,
                limit=limit,
                agent_max_output_tokens=agent_max_output_tokens,
                train_only=False,
            )

        # 最后一个 epoch 只做 inference，对应 scripts/main.sh 的尾部逻辑
        await run_inference_epoch(
            dataset=dataset,
            epoch=max_epoch,
            zcp=zcp,
            limit=limit,
            full_infer=full_infer,
            agent_max_output_tokens=agent_max_output_tokens,
        )

    cprint("[ScoreFlowRunner] ScoreFlow pipeline completed.", color="green")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Non-stop ScoreFlow runner (Python version of scripts/main.sh).")
    parser.add_argument("--dataset", type=str,
                        default="GSM8K", help="Dataset to use")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Start epoch (inclusive)")
    parser.add_argument("--max_epoch", type=int, default=3,
                        help="Max epoch (inclusive only for final inference)")
    parser.add_argument("--zcp", type=str,
                        default="accuracy", help="ZCP value")
    parser.add_argument("--limit", type=int, default=72,
                        help="Limit for data points (optimize / inference when not full)")
    parser.add_argument(
        "--full_infer",
        action="store_true",
        help="Use full inference (ignore limit for inference) when this flag is set",
    )
    parser.add_argument(
        "--agent_max_output_tokens",
        type=int,
        default=1024,
        help="Max output tokens for single agent (pass to LLM manager)",
    )
    parser.add_argument(
        "--infer_only",
        action="store_true",
        help="Only run inference (generate + evaluate), skip optimization",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Only run training (optimize: generate + evaluate + DPO), skip inference",
    )

    args = parser.parse_args()

    # 检查参数互斥性
    if args.infer_only and args.train_only:
        parser.error("--infer_only and --train_only cannot be used together")

    asyncio.run(
        run_scoreflow_pipeline(
            dataset=args.dataset,
            start_epoch=args.start_epoch,
            max_epoch=args.max_epoch,
            zcp=args.zcp,
            limit=args.limit,
            full_infer=args.full_infer,
            agent_max_output_tokens=args.agent_max_output_tokens,
            infer_only=args.infer_only,
            train_only=args.train_only,
        )
    )

    get_global_llm_manager().clear_all()
    cprint("All LLM models have been cleared!", color="green")
    logger.info("All LLM models have been cleared!")


if __name__ == "__main__":
    main()
