from difflib import SequenceMatcher
from metagpt.logs import logger
from utils.VLLMAdapter import VLLMAdapter
from vllm import SamplingParams
import vllm
import aiofiles
import json
import os
import importlib
import re
import yaml
import asyncio
import aiofiles
import pickle
from tqdm import tqdm
import ScoreFlow.params
import argparse
import time
import traceback

from termcolor import cprint

from utils.llm_manager import get_global_llm_manager
from utils.path_utils import (
    build_model_name,
    ensure_dir,
    finetuned_dir,
    temp_generate_dir,
    workflow_output_dir,
    workflow_output_file,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# OPTIONAL_LLM = True
OPTIONAL_LLM = False

GRAPH_NUM = 8
num_epoch = 10
generate_batch_size = MAX_CONCURRENT_TASKS // GRAPH_NUM  # (7473 * 5% = 374)
# generate_batch_size = 748 # (7473 * 10% = 748)
# generate_batch_size = 374 # (7473 * 5% = 374)
# generate_batch_size = 512 # Qwen2.5-7B-Instruct 8 min 44 s
# generate_batch_size = 160 # Qwen2.5-7B-Instruct 2 min 38 s


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load dataset


def load_data(file_path):
    data = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def get_sampling_params(temperature):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=1000,
        stop=["</graph>"]
    )
    return sampling_params

# simialrity ratio between generated workflow and template workflow


def similarity_ratio(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

# must make sure the generated worklow is executable, and pass certain conditions, including: 1. time limit 2. modification


async def test_if_runable(num_epoch, sub_index_i, i, generated_text, problem_type, fail_list, PYTHON_START, PYTHON_END, sim_threshold, TEMP_AVOID, TEST_PROMPT, NO_EXCEPTION_LIST, temp_file_dir):
    error_info = None
    try:
        # extract python script
        graph_content = re.search(
            r"<graph>(.*?)</graph>", generated_text, re.DOTALL)
        if graph_content is None:
            error_info = "Format error"
            logger.info(f"Error for datapoint {sub_index_i}: Format error")
            fail_list.append((i, error_info, generated_text))
            return (False, error_info, generated_text)
        class_script = graph_content.group(1).strip()
        extract_graph_script = re.search(
            r"async def run_workflow\(self\)(.*?)return", class_script, re.DOTALL)
        if extract_graph_script is None:
            error_info = "Format error"
            logger.info(f"Error for datapoint {sub_index_i}: Format error")
            fail_list.append((i, error_info, generated_text))
            return (False, error_info, generated_text)

        # if there is requirement on level of modification (based on template)
        extract_graph_script = extract_graph_script.group(1).strip()
        extract_TEMP_AVOID = re.search(
            r"async def run_workflow\(self\)(.*?)return", TEMP_AVOID, re.DOTALL).group(1).strip()
        similar_score = similarity_ratio(
            extract_graph_script, extract_TEMP_AVOID)
        if similar_score >= sim_threshold:
            error_info = "The similarity to the given template is too high, requiring further modifications."
            logger.info(
                f"Error for datapoint {sub_index_i}: Nearly no modification")
            fail_list.append((i, error_info, generated_text))
            return (False, error_info, generated_text)

        python_script = PYTHON_START + class_script + PYTHON_END

        for no_exception_char in NO_EXCEPTION_LIST:
            if no_exception_char in python_script:
                error_info = f"Contains prohibited code snippet: {no_exception_char}"
                logger.info(
                    f"Error for datapoint {sub_index_i}: Contain no_exception_char {no_exception_char}")
                fail_list.append((i, error_info, generated_text))
                return (False, error_info, generated_text)

        async with asyncio.Semaphore(MAX_OPEN_FILES):
            # write and load workflow in .py
            async with aiofiles.open(temp_file_dir + f"/graph_{num_epoch}_{sub_index_i}.py", mode='w') as graph_file:
                await graph_file.write(python_script)
        workflows_path = temp_file_dir.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{workflows_path}.graph_{num_epoch}_{sub_index_i}"
        graph_module = __import__(graph_module_name, fromlist=[""])
        graph_class = getattr(graph_module, "Workflow")

        # test if executable
        if problem_type == None:
            graph_class = graph_class(problem=TEST_PROMPT)
        else:
            graph_class = graph_class(problem=TEST_PROMPT[problem_type])
        await graph_class()

        if os.path.exists(temp_file_dir + f"/graph_{num_epoch}_{sub_index_i}.py"):
            os.remove(temp_file_dir + f"/graph_{num_epoch}_{sub_index_i}.py")
        else:
            logger.warning(
                f"Warning for datapoint {sub_index_i}: {temp_file_dir}/graph_{num_epoch}_{sub_index_i}.py not found")
        return (True, None, None)

    except Exception as e:
        error_info = f"Runtime error: {str(e)}"
        logger.info(f"Error for datapoint {sub_index_i}: {e}")
        # if 'validation error' in str(e) or 'NameError' in str(e): # 可接受的workflow运行错误
        #     fail_list.append(i)
        #     os.remove(temp_file_dir + f"/graph_{num_epoch}_{sub_index_i}.py")
        #     return False

        ALLOWED_ERROR_LIST = [
            'has already been used and cannot be reused', 'invalid syntax', 'attribute']
        raise_error_signal = True
        for error_s in ALLOWED_ERROR_LIST:
            if error_s in str(e):
                raise_error_signal = False
                break
        # if e is not None: # Always raise error
        # if len(str(e)) == 0:
        # if 'has already been used and cannot be reused.' in str(e):

        # if raise_error_signal:
        #     print("========== MAIN ERROR ==========")
        #     print("Exception object:", repr(e))
        #     import traceback
        #     traceback.print_exc()
        #     if 'retry' in str(e):
        #         print(repr(e.last_attempt.exception()))
        #     print("========================================")
        #     exit(1)

        # if 'validation error' in str(e):
        #     raise e
        fail_list.append((i, error_info, generated_text))

        # os.remove(temp_file_dir + f"/graph_{num_epoch}_{sub_index_i}.py")
        # 将出错的脚本移动到以报错信息命名的文件夹
        error_dir = os.path.join(temp_file_dir, str(e).replace(
            '/', '_').replace('\\', '_').replace(':', '_')[:100])
        ensure_dir(error_dir)
        target_file = os.path.join(
            error_dir, f"graph_{num_epoch}_{sub_index_i}.py")
        # 处理重名：若已存在则加序号
        counter = 1
        original_target = target_file
        while os.path.exists(target_file):
            name, ext = os.path.splitext(original_target)
            target_file = f"{name}_{counter}{ext}"
            counter += 1
        os.rename(temp_file_dir +
                  f"/graph_{num_epoch}_{sub_index_i}.py", target_file)

        # 保留最近10个错误脚本：按创建时间排序，删除更早的
        all_error_files = []
        # 遍历错误目录下所有 .py 文件
        for fname in os.listdir(error_dir):
            if fname.endswith('.py'):
                fpath = os.path.join(error_dir, fname)
                try:
                    # 获取文件创建时间
                    ctime = os.path.getctime(fpath)
                    all_error_files.append((ctime, fpath))
                except OSError:
                    continue
        # 按创建时间升序排序
        all_error_files.sort(key=lambda x: x[0])
        # 如果超过10个，删除最早的
        while len(all_error_files) > 10:
            _, oldest_path = all_error_files.pop(0)
            try:
                os.remove(oldest_path)
            except OSError:
                pass

        return (False, error_info, generated_text)


async def get_fail_list(num_epoch, sub_index, sub_generated_results, sub_type_list, PYTHON_START, PYTHON_END, sim_threshold, TEMP_AVOID, TEST_PROMPT, NO_EXCEPTION_LIST, temp_file_dir):
    tasks = []
    fail_list = []
    for i, generated_text in enumerate(sub_generated_results):
        tasks.append(test_if_runable(num_epoch, sub_index[i], i, generated_text, sub_type_list[i], fail_list,
                     PYTHON_START, PYTHON_END, sim_threshold, TEMP_AVOID, TEST_PROMPT, NO_EXCEPTION_LIST, temp_file_dir))
    await asyncio.gather(*tasks)
    if len(fail_list) == len(sub_generated_results):
        logger.error("All generated workflows are failed")
        # raise Exception("All generated workflows are failed")
    # fail_list 现在包含 (index, error_info, generated_text) 元组
    return fail_list


async def generate_graphs(generator_model_name, data, generator_sampling_params, num_epoch, graph_num, benchmark, data_set, temp_file_dir):

    # load prompts and conditions
    prompt_module = importlib.import_module(
        f"ScoreFlow.scripts.{data_set}.conditions{'_llm' if OPTIONAL_LLM else ''}")
    PYTHON_END = prompt_module.PYTHON_END.format(
        time=TIME_FOR_GRAPH_TO_RESPONSE)
    sim_threshold = prompt_module.sim_threshold
    PYTHON_START = prompt_module.PYTHON_START
    TEMP_AVOID = prompt_module.TEMP_AVOID
    TEST_PROMPT = prompt_module.TEST_PROMPT
    NO_EXCEPTION_LIST = prompt_module.NO_EXCEPTION_LIST
    START_PROMPT = prompt_module.START_PROMPT
    END_PROMPT = prompt_module.END_PROMPT

    # generate prompts
    prompts = []
    type_list = []
    for problem in data:
        optimize_prompt = START_PROMPT + \
            benchmark.get_graph_input_text(problem) + END_PROMPT
        prompts = prompts + [optimize_prompt]*graph_num
        if "problem_type" in problem:
            type_list = type_list + [problem["problem_type"]]*graph_num
        else:
            type_list = type_list + [None]*graph_num

    # generate workflows
    generated_results = []

    # High-performance LLM inference engine: fast, high concurrency, and low memory consumption.
    # vllm model
    tasks = [get_global_llm_manager().generate(generator_model_name, prompt,
                                               sampling_params=generator_sampling_params) for prompt in prompts]
    outputs = await asyncio.gather(*tasks)
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_text = generated_text + "</graph>"
        generated_results.append(generated_text)

    sub_generated_results = generated_results
    sub_type_list = type_list
    sub_index = [i for i in range(len(generated_results))]
    # 存储每个索引对应的错误历史，用于多轮问答改进
    error_history = {}  # {index: [(error_info, generated_text), ...]}

    while num_epoch >= 0:

        # get failed workflows
        fail_list = await get_fail_list(num_epoch, sub_index, sub_generated_results, sub_type_list, PYTHON_START, PYTHON_END, sim_threshold, TEMP_AVOID, TEST_PROMPT, NO_EXCEPTION_LIST, temp_file_dir)
        if len(fail_list) == 0:
            break

        # fail_list 现在包含 (relative_index, error_info, generated_text) 元组
        # 需要转换为原始索引
        fail_list.sort(key=lambda x: x[0])
        fail_percentage = 100 * len(fail_list) / len(sub_index)
        logger.info(
            f"epoch: {num_epoch}, fail_percentage: {fail_percentage:.1f}%, length of fail_list: {len(fail_list)}. total: {len(sub_index)}.")

        # 提取失败项的原始索引、错误信息和生成结果
        original_fail_indices = []
        fail_error_info_list = []
        fail_generated_texts = []

        for relative_idx, error_info, generated_text in fail_list:
            original_idx = sub_index[relative_idx]
            original_fail_indices.append(original_idx)
            fail_error_info_list.append(error_info)
            fail_generated_texts.append(generated_text)

            # 更新错误历史
            if original_idx not in error_history:
                error_history[original_idx] = []
            error_history[original_idx].append((error_info, generated_text))

        sub_index = original_fail_indices
        sub_type_list = [type_list[i] for i in sub_index]
        original_prompts = [prompts[i] for i in sub_index]

        # 检查是否需要使用多轮对话格式
        use_chat_template = VLLMAdapter.needs_chat_template(
            generator_model_name)

        # 构建包含错误反馈的改进提示
        improved_prompts = []
        for idx, (original_prompt, error_info, prev_generated_text) in enumerate(zip(original_prompts, fail_error_info_list, fail_generated_texts)):
            # 获取该索引的所有历史错误
            history = error_history[sub_index[idx]]

            if use_chat_template:
                # 构建多轮对话格式
                conversation = []

                # 第一轮：原始提示（user）
                conversation.append(
                    {"role": "user", "content": original_prompt})

                # 添加历史对话轮次
                for hist_idx, (hist_error, hist_text) in enumerate(history[-5:]):
                    # Assistant 的回复（上一轮生成的代码）
                    prev_graph_match = re.search(
                        r"<graph>(.*?)</graph>", hist_text, re.DOTALL)
                    if prev_graph_match:
                        assistant_content = prev_graph_match.group(1).strip()
                    else:
                        assistant_content = hist_text
                    conversation.append(
                        {"role": "assistant", "content": f"<graph>\n{assistant_content}\n</graph>"})

                    # User 的错误反馈
                    error_feedback = ERROR_FEEDBACK_HEADER
                    error_feedback += f"{ERROR_FEEDBACK_ERROR_PREFIX}{hist_error}\n\n"

                    # 如果是多轮失败，添加历史错误信息
                    if hist_idx < len(history[-5:]) - 1:
                        error_feedback += ERROR_FEEDBACK_HISTORY_HEADER
                        for round_num, (prev_hist_error, _) in enumerate(history[-5:][:hist_idx]):
                            error_feedback += ERROR_FEEDBACK_ROUND_PREFIX.format(
                                round_num=round_num) + f"{prev_hist_error}\n"
                        error_feedback += "\n"

                    error_feedback += ERROR_FEEDBACK_TAIL

                    conversation.append(
                        {"role": "user", "content": error_feedback})

                improved_prompts.append(conversation)
            else:
                # 构建拼接格式（原有方式）
                error_feedback = ERROR_FEEDBACK_CONCAT_HEADER
                error_feedback += f"{ERROR_FEEDBACK_ERROR_PREFIX}{error_info}\n\n"

                # 如果是多轮失败，添加历史错误信息
                if len(history) > 1:
                    error_feedback += ERROR_FEEDBACK_HISTORY_HEADER
                    for round_num, (hist_error, hist_text) in enumerate(history[:-1], 1):
                        error_feedback += ERROR_FEEDBACK_ROUND_PREFIX.format(
                            round_num=round_num) + f"{hist_error}\n"
                    error_feedback += "\n"

                error_feedback += ERROR_FEEDBACK_CONCAT_CODE_HEADER
                error_feedback += ERROR_FEEDBACK_CONCAT_GRAPH_START
                # 提取上一轮生成的 graph 内容
                prev_graph_match = re.search(
                    r"<graph>(.*?)</graph>", prev_generated_text, re.DOTALL)
                if prev_graph_match:
                    error_feedback += prev_graph_match.group(1).strip() + "\n"
                else:
                    error_feedback += prev_generated_text + "\n"
                error_feedback += ERROR_FEEDBACK_CONCAT_GRAPH_END

                error_feedback += ERROR_FEEDBACK_TAIL + "\n"

                # 将错误反馈添加到原始提示后
                improved_prompt = original_prompt + error_feedback
                improved_prompts.append(improved_prompt)

        # generate workflows for those failed again with improved prompts
        sub_generated_results = []
        tasks = [get_global_llm_manager().generate(generator_model_name, prompt,
                                                   sampling_params=generator_sampling_params) for prompt in improved_prompts]
        # tasks = [get_global_llm_manager().generate(generator_model_name, prompt, sampling_params=generator_sampling_params) for prompt in original_prompts]
        outputs = await asyncio.gather(*tasks)
        for output in outputs:
            generated_text = output.outputs[0].text
            generated_text = generated_text + "</graph>"
            sub_generated_results.append(generated_text)

        # renew the failed ones
        for i, original_idx in enumerate(sub_index):
            generated_results[original_idx] = sub_generated_results[i]
            type_list[original_idx] = sub_type_list[i]

        num_epoch -= 1

    return generated_results


async def run_generate(
    data_set: str,
    task_type: str,
    epoch: int,
    data_limit: int | None = None,
    zcp: str = "accuracy",
    agent_max_output_tokens: int = 1024,
):
    """供其他模块调用的生成入口，等价于原 main 的主体逻辑。"""
    bench_dic = ScoreFlow.params.bench_dic

    if task_type == "optimize":
        data_set_type = "validate"  # dataset type
        graph_num = GRAPH_NUM  # number of workflow to generate
    elif task_type == "inference":
        data_set_type = "test"
        graph_num = 1
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # output directory for workflow files
    output_dir = workflow_output_file(data_set, zcp, epoch, task_type)

    # dataset path
    file_path = "data/" + \
        bench_dic[data_set]["benchmark_name"] + "_" + data_set_type + ".jsonl"

    # if is the first epoch
    is_first = epoch == 0

    # load model configurations
    with open("config/generator_config.yaml", "r") as file:
        generator_config = yaml.safe_load(file)

    # config for generator
    generator_model_name = generator_config["model"]
    generator_temperature = generator_config["temperature"]

    # temp file directory
    temp_file_dir = temp_generate_dir(data_set, zcp)

    ensure_dir(workflow_output_dir(data_set, zcp))
    ensure_dir(temp_file_dir)
    ensure_dir(finetuned_dir(data_set, zcp))

    # import corresponding benchmark module
    sub_path = bench_dic[data_set]["benchmark_name"]
    benchmark_module_path = f"ScoreFlow.benchmark.{sub_path}"
    module = importlib.import_module(benchmark_module_path)
    benchmark = getattr(module, bench_dic[data_set]["module_name"])
    benchmark = benchmark(name=data_set, file_path="data", log_path="")

    data = load_data(file_path)

    # Apply data limit if specified
    if data_limit is not None:
        data = data[:data_limit]

    get_global_llm_manager(
        agent_max_output_tokens=agent_max_output_tokens,
        max_expected_requests=min(len(data) * graph_num, MAX_CONCURRENT_TASKS),
    )

    # load generator
    generator_model_name = (
        generator_model_name
        if is_first
        else build_model_name(generator_model_name, data_set, zcp, epoch)
    )
    generator_sampling_params = get_sampling_params(generator_temperature)

    start_time = time.time()
    generated_results = []
    for start in range(0, len(data), generate_batch_size):
        _start_time = time.time()
        end = min(start + generate_batch_size, len(data))
        batch_data = data[start:end]
        batch_results = await generate_graphs(
            generator_model_name,
            batch_data,
            generator_sampling_params,
            num_epoch,
            graph_num,
            benchmark,
            data_set,
            temp_file_dir,
        )
        generated_results.extend(batch_results)
        cprint(
            f"Generator-LLM 处理数据 {start} 到 {end} 条完毕（总 {len(data)} 条），已完成 {end/len(data)*100:.2f} %，本次耗时 {time.time() - _start_time:.2f} 秒，共耗时 {time.time() - start_time:.2f} 秒，预计剩余 {(len(data) - end) / end * (time.time() - start_time):.2f} 秒",
            color="green",
        )
        logger.info(
            f"Generator-LLM 处理数据 {start} 到 {end} 条完毕（总 {len(data)} 条），已完成 {end/len(data)*100:.2f} %，本次耗时 {time.time() - _start_time:.2f} 秒，共耗时 {time.time() - start_time:.2f} 秒，预计剩余 {(len(data) - end) / end * (time.time() - start_time):.2f} 秒"
        )

    # output generated workflows
    final_dataset = []
    for i in range(len(generated_results)):
        final_dataset.append([data[int(i / graph_num)], generated_results[i]])
    with open(output_dir, "wb") as f:
        pickle.dump(final_dataset, f)
    cprint("Generation Process Done!", color="green")
    logger.info("Generation Process Done!")

    get_global_llm_manager().clear_all_finetuned_llms()


async def main():
    """CLI entrypoint：保留原有命令行用法。"""
    # input dataset, task type (optimize/inference), and epoch(0, 1, 2...)
    parser = argparse.ArgumentParser(
        description="Process command-line arguments")
    parser.add_argument("--dataset", type=str,
                        required=True, help="Value for Dataset")
    parser.add_argument("--task", type=str, required=True,
                        help="Value for Task")
    parser.add_argument("--epoch", type=str, required=True,
                        help="Value for Epoch")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of data points to process")
    parser.add_argument("--zcp", type=str, default="accuracy",
                        help="ZCP for evaluation")
    parser.add_argument(
        "--agent_max_output_tokens",
        type=int,
        default=1024,
        help="Max output tokens for single agent",
    )
    args = parser.parse_args()

    await run_generate(
        data_set=args.dataset,
        task_type=args.task,
        epoch=int(args.epoch),
        data_limit=args.limit,
        zcp=args.zcp,
        agent_max_output_tokens=args.agent_max_output_tokens,
    )


if __name__ == "__main__":
    asyncio.run(main())

    get_global_llm_manager().clear_all()
    logger.info("All LLM models have been cleared!")
