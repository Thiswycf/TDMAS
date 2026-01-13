import re
import os
import argparse
import importlib
import importlib.util
import pickle
import asyncio
import time
import ScoreFlow.params
from termcolor import cprint
import datetime
from difflib import SequenceMatcher
import numpy as np
from metagpt.logs import logger
import traceback
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from utils.common import TIME_FOR_GRAPH_TO_RESPONSE, MAX_CONCURRENT_TASKS, TIME_TO_READ_SCRIPT
from utils.llm_manager import get_global_llm_manager
from utils.path_utils import (
    ensure_dir,
    evaluation_output_dir,
    evaluation_output_file,
    legacy_evaluation_output_file,
    legacy_preference_data_file,
    legacy_workflow_output_file,
    preference_data_dir,
    preference_data_file,
    resolve_existing_path,
    solve_rate_file,
    temp_eval_dir,
    workflow_output_dir,
    workflow_output_file,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def sample_data(p_r_data, N, f):
    prob_list = []
    for i in range(len(p_r_data)):
        w = p_r_data[i]["chosen_score"]
        l = p_r_data[i]["rejected_score"]
        prob_list.append(f(w, l))
    sum_prob = sum(prob_list)
    prob_list = [p/sum_prob for p in prob_list]
    sampled_elements = np.random.choice(
        p_r_data, size=N, replace=True, p=prob_list)
    return sampled_elements.tolist()


def similarity_ratio(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


def load_graph(question_id, graph_id, workflows_path: str):
    from time import sleep, time
    # 构建文件路径（使用原始路径，不转换为模块名）
    graph_file_path = os.path.join(
        workflows_path, f"graph_{question_id}_{graph_id}.py")
    start_time = time()
    while time() - start_time < TIME_TO_READ_SCRIPT:
        if os.path.exists(graph_file_path):
            break
        sleep(0.01)
    assert os.path.exists(
        graph_file_path), f"Graph file {graph_file_path} not found"
    try:
        # 使用 importlib.util 从文件路径直接加载模块
        spec = importlib.util.spec_from_file_location(
            f"graph_{question_id}_{graph_id}",
            graph_file_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for {graph_file_path}")
        graph_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graph_module)
        graph_class = getattr(graph_module, "Workflow")
        return graph_class
    except Exception as e:
        logger.info(f"Error loading graph: {e}")
        raise


@retry(stop=stop_after_attempt(1), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
async def _configure_graph(graph, problem=None):
    return graph(problem=problem)

# get postprocessor function


def load_postprocessor(workflows_path: str, name):
    workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
    ext_module_name = f"{workflows_path}." + name
    try:
        ext_module = __import__(ext_module_name, fromlist=[""])
        ext_class = getattr(ext_module, "Workflow")
        return ext_class
    except ImportError as e:
        logger.info(f"Error loading graph: {e}")
        raise


async def _configure_postprocessor(extraction, llm_name):
    return extraction(llm_name=llm_name)


async def get_scores(work_dir, data_set, llm_name, max_concurrent_tasks, i, question_start, question_end, post_dir, use_judger, use_extraction, data, benchmark, vali_num=3, zcp=None):
    prompt_module = importlib.import_module(
        f"ScoreFlow.scripts.{data_set}.conditions")
    PYTHON_END = prompt_module.PYTHON_END.format(
        time=TIME_FOR_GRAPH_TO_RESPONSE)
    PYTHON_START = prompt_module.PYTHON_START
    TEMP_AVOID = prompt_module.TEMP_AVOID

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    graph_scores = []
    tasks = []
    extraction = load_postprocessor(post_dir, "extraction")
    judger = load_postprocessor(post_dir, "judger")
    if use_judger:
        configured_judger = await _configure_postprocessor(judger, llm_name)
    else:
        configured_judger = None
    if use_extraction:
        configured_extraction = await _configure_postprocessor(extraction, llm_name)
    else:
        configured_extraction = None

    for question_id in range(question_start, question_end):
        problem = data[i][0]
        graph = []
        while (1):
            graph.append(data[i][1])
            i += 1
            if i == len(data):
                break
            i_id = benchmark.get_problem_id(data[i][0])
            i_last_id = benchmark.get_problem_id(data[i-1][0])
            if i_id != i_last_id:
                break

        for graph_id, response_text in enumerate(graph):
            graph_file_path = None
            try:
                # Extract graph content
                graph_content = re.search(
                    r"<graph>(.*?)</graph>", response_text, re.DOTALL)
                python_script = PYTHON_START + \
                    graph_content.group(1).strip() + PYTHON_END
                graph_file_path = os.path.join(
                    work_dir, f"graph_{question_id}_{graph_id}.py")
                with open(graph_file_path, mode='w') as graph_file:
                    graph_file.write(python_script)
                optimizer_graph = load_graph(question_id, graph_id, work_dir)
                # input_text = benchmark.get_input_text(problem)
                # configured_graph = await _configure_graph(optimizer_graph, input_text)

                # async def sem_evaluate(question_id, graph_id, rep_id, problem, configured_extraction, configured_judger, configured_graph):
                async def sem_evaluate(question_id, graph_id, rep_id, problem, configured_extraction, configured_judger, optimizer_graph, zcp=None):
                    # async with semaphore:
                    try:
                        # evaluate workflow
                        # results = await benchmark.evaluate_problem(problem, configured_extraction, configured_judger, configured_graph)
                        results = await benchmark.evaluate_problem(problem, configured_extraction, configured_judger, optimizer_graph, zcp=zcp)
                        # print(f'Get result in question {question_id}, graph {graph_id}, rep {rep_id}, results {results}')
                        graph_scores.append(
                            [question_id, graph_id, rep_id, results[3], results[0], results[1], results[2]])
                    except Exception as e:
                        # logger.info(f"Error when running: {e}")
                        raise e

                # evaluate vali_num times for each task
                for rep_id in range(vali_num):
                    # tasks.append(sem_evaluate(question_id, graph_id, rep_id, problem, configured_extraction, configured_judger, configured_graph))
                    tasks.append(sem_evaluate(question_id, graph_id, rep_id, problem,
                                 configured_extraction, configured_judger, optimizer_graph, zcp))

            except Exception as e:
                logger.info(f"Error when loading graph: {e}")
            finally:
                if graph_file_path and os.path.exists(graph_file_path):
                    os.remove(graph_file_path)

        if i == len(data):
            break

    await asyncio.gather(*tasks)

    return graph_scores


async def run_get_scores(data_set, task_type, epoch, parallel_id, limit, zcp, agent_max_output_tokens, vali_num):
    bench_dic = ScoreFlow.params.bench_dic

    max_concurrent_tasks = 50
    use_extraction = True

    if task_type == "optimize":
        use_judger = False
        graph_num = 8
    elif task_type == "inference":
        use_judger = True
        graph_num = 1
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    input_dir = resolve_existing_path(
        workflow_output_file(data_set, zcp, epoch, task_type),
        [legacy_workflow_output_file(zcp, epoch, task_type)],
    )  # input directory of workflow files
    # output directory of evaluation feedback files
    output_dir = evaluation_output_file(
        data_set, zcp, epoch, parallel_id, task_type)

    # batch_size for evaluation, similar to generate_batch_size in generate.py
    evaluate_batch_size = MAX_CONCURRENT_TASKS // (graph_num * vali_num) if (
        graph_num * vali_num) > 0 else MAX_CONCURRENT_TASKS

    # temp file directory
    work_dir = temp_eval_dir(data_set, zcp)

    # import corresponding benchmark module
    sub_path = bench_dic[data_set]["benchmark_name"]
    benchmark_module_path = f"ScoreFlow.benchmark.{sub_path}"
    module = importlib.import_module(benchmark_module_path)
    benchmark = getattr(module, bench_dic[data_set]["module_name"])
    benchmark = benchmark(name=data_set, file_path="data", log_path="")

    post_dir = "ScoreFlow/scripts/" + data_set
    ensure_dir(work_dir)
    ensure_dir(evaluation_output_dir(data_set, zcp))

    # load workflow data
    with open(input_dir, "rb") as f:
        data = pickle.load(f)

    # 若data[i][0]没有'id'的键，则自行分配。
    # 为每个data条目分配唯一问题id（递增），如果已存在则保留
    next_auto_id = 0
    for entry in data:
        problem = entry[0]
        if not isinstance(problem, dict):
            continue
        if 'id' not in problem:
            problem['id'] = f"auto_{next_auto_id}"
            next_auto_id += 1

    len_data = int(len(data)/graph_num)
    start, end = 0, min(len_data, limit) if limit is not None else len_data

    get_global_llm_manager(agent_max_output_tokens=agent_max_output_tokens, max_expected_requests=min(
        (end - start) * vali_num * graph_num, MAX_CONCURRENT_TASKS))

    # evaluate workflows to get scores
    score_results = []
    start_time = time.time()
    for question_start in range(start, end, evaluate_batch_size):
        _start_time = time.time()
        question_end = min(question_start + evaluate_batch_size, end)

        # get the start position i for current batch
        i = 0
        question_id = 0
        while (1):
            if question_id == question_start:
                break
            while (1):
                i += 1
                if i >= len(data):
                    break
                i_id = benchmark.get_problem_id(data[i][0])
                i_last_id = benchmark.get_problem_id(data[i-1][0])
                if i_id != i_last_id:
                    break
            if i >= len(data):
                break
            question_id += 1

        # 使用 await 而不是 asyncio.run()，避免在已有事件循环中调用 asyncio.run()
        graph_scores = await get_scores(work_dir, data_set, 'Qwen3-8B', max_concurrent_tasks, i, question_start, question_end, post_dir, use_judger, use_extraction, data, benchmark, vali_num=vali_num, zcp=zcp if task_type == "optimize" else "accuracy")
        score_results.extend(graph_scores)
        elapsed_time = time.time() - start_time
        batch_time = time.time() - _start_time
        total_questions = end - start
        completed_questions = question_end - start
        remaining_questions = end - question_end
        progress_pct = (completed_questions / total_questions *
                        100) if total_questions > 0 else 0
        estimated_remaining = (remaining_questions / completed_questions *
                               elapsed_time) if completed_questions > 0 else 0

        cprint(f'Evaluator-LLM 处理问题 {question_start} 到 {question_end} 完毕（总 {total_questions} 个问题），已完成 {progress_pct:.2f} %，本次耗时 {batch_time:.2f} 秒，共耗时 {elapsed_time:.2f} 秒，预计剩余 {estimated_remaining:.2f} 秒', color="green")
        logger.info(
            f"Evaluator-LLM 处理问题 {question_start} 到 {question_end} 完毕（总 {total_questions} 个问题），已完成 {progress_pct:.2f} %，本次耗时 {batch_time:.2f} 秒，共耗时 {elapsed_time:.2f} 秒，预计剩余 {estimated_remaining:.2f} 秒")

    # output score information
    with open(output_dir, 'wb') as f:
        pickle.dump(score_results, f)


def generate_preference_data(bench_dic, data_set, epoch, task_type, zcp, limit, vali_num=3):

    # load benchmark module
    sub_path = bench_dic[data_set]["benchmark_name"]
    benchmark_module_path = f"ScoreFlow.benchmark.{sub_path}"
    module = importlib.import_module(benchmark_module_path)
    benchmark = getattr(module, bench_dic[data_set]["module_name"])
    benchmark = benchmark(name=data_set, file_path="", log_path="")

    if task_type == "inference":
        post = "-test"
        graph_num = 1
    elif task_type == "optimize":
        post = ""
        graph_num = 8

    workflow_path = resolve_existing_path(
        workflow_output_file(data_set, zcp, epoch, task_type),
        [legacy_workflow_output_file(zcp, epoch, task_type)],
    )
    with open(workflow_path, 'rb') as file:
        data = pickle.load(file)

    # 若data[i][0]没有'id'的键，则自行分配。
    # 为每个data条目分配唯一问题id（递增），如果已存在则保留
    next_auto_id = 0
    for entry in data:
        problem = entry[0]
        if not isinstance(problem, dict):
            continue
        if 'id' not in problem:
            problem['id'] = f"auto_{next_auto_id}"
            next_auto_id += 1

    score_file = resolve_existing_path(
        evaluation_output_file(data_set, zcp, epoch, "0", task_type),
        [legacy_evaluation_output_file(zcp, epoch, "0", task_type)],
    )
    with open(score_file, 'rb') as f:
        loaded_list0 = pickle.load(f)
    # with open(evaluation_output_file(data_set, zcp, epoch, "1", task_type), 'rb') as f:
    #     loaded_list1 = pickle.load(f)
    # with open(evaluation_output_file(data_set, zcp, epoch, "2", task_type), 'rb') as f:
    #     loaded_list2 = pickle.load(f)
    # score_list = loaded_list0 + loaded_list1 + loaded_list2
    score_list = loaded_list0
    n_data = int(len(data)/graph_num)

    # load prompts and conditions
    prompt_module = importlib.import_module(
        f"ScoreFlow.scripts.{data_set}.conditions")
    START_PROMPT = prompt_module.START_PROMPT
    END_PROMPT = prompt_module.END_PROMPT
    TEMP_AVOID = prompt_module.TEMP_AVOID
    sim_threshold = prompt_module.sim_threshold

    avg_score_list = []
    for i in range(n_data):
        sub_score_list = [sub_l for sub_l in score_list if sub_l[0] == i]
        graph_list = list(set([sub_l[1] for sub_l in sub_score_list]))
        if len(graph_list) != graph_num:
            print("graph_num not right here: ", i, len(graph_list))
        sub_avg_score_list = []
        for graph in graph_list:
            avg_score = [sub_l[3]
                         for sub_l in sub_score_list if sub_l[1] == graph]
            if len(avg_score) != vali_num:
                print("vali_num not right here:", i, graph)
            avg_score = sum(avg_score)/vali_num
            sub_avg_score_list.append([i, graph, avg_score])
        avg_score_list = avg_score_list + sub_avg_score_list

    if task_type == "inference":
        cprint(
            f"Average Test Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list) * 100:.4f}%", color="green")
        logger.info(
            f"Average Test Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list) * 100:.4f}%")
        # 将平均通过率写入结果文件
        result_dir = evaluation_output_dir(data_set, zcp)
        ensure_dir(result_dir)
        result_file = solve_rate_file(data_set, zcp, task_type)
        with open(result_file, "a") as f:
            f.write(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | Limit {limit} | Epoch {epoch} | Test Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list) * 100:.4f}%\n")
        return
    elif zcp == 'accuracy':
        cprint(
            f"Average Train Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list) * 100:.4f}%", color="green")
        logger.info(
            f"Average Train Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list) * 100:.4f}%")
        # 将平均通过率写入结果文件
        result_dir = evaluation_output_dir(data_set, zcp)
        ensure_dir(result_dir)
        result_file = solve_rate_file(data_set, zcp, task_type)
        with open(result_file, "a") as f:
            f.write(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | Limit {limit} | Epoch {epoch} | Train Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list) * 100:.4f}%\n")

    # check if requirement holds
    i = 0
    question_id = 0
    list_aviod = []
    while (1):
        if question_id == n_data:
            break
        graphs = []
        while (1):
            graphs.append(data[i][1])
            i += 1
            if i == len(data):
                break
            i_id = benchmark.get_problem_id(data[i][0])
            i_last_id = benchmark.get_problem_id(data[i-1][0])
            if i_id != i_last_id:
                break

        for j in range(len(graphs)):
            graph = graphs[j]
            graph_script = re.search(
                r"<graph>(.*?)</graph>", graph, re.DOTALL).group(1).strip()
            extract_graph_script_search = re.search(
                r"async def run_workflow\(self\)(.*?)return", graph_script, re.DOTALL)
            if extract_graph_script_search is None:
                extract_graph_script = re.search(
                    r"async def run_workflow\(self\)(.*?)return", TEMP_AVOID, re.DOTALL).group(1).strip()
            else:
                extract_graph_script = extract_graph_script_search.group(
                    1).strip()
            extract_TEMP_AVOID = re.search(
                r"async def run_workflow\(self\)(.*?)return", TEMP_AVOID, re.DOTALL).group(1).strip()
            similar_score = similarity_ratio(
                extract_graph_script, extract_TEMP_AVOID)
            if similar_score >= sim_threshold:
                list_aviod.append(
                    [question_id, j, extract_graph_script, similar_score])
        question_id += 1
    list_aviod = sorted(list_aviod, key=lambda x: (x[3]))
    list_id_aviod = [[sub_l[0], sub_l[1]] for sub_l in list_aviod]
    avg_score_list = [sub_l for sub_l in avg_score_list if [
        sub_l[0], sub_l[1]] not in list_id_aviod]

    # organize data structure
    pre_rej_list = []
    for i in range(n_data):
        sub_data = [sub_l for sub_l in avg_score_list if sub_l[0] == i]
        sub_data = sorted(sub_data, key=lambda x: x[2])
        check_list = []
        for j in range(len(sub_data)):
            for k in range(j+1, len(sub_data)):
                score_w = sub_data[k][2]
                score_l = sub_data[j][2]
                check_list.append(
                    [i, sub_data[j][1], sub_data[k][1], score_w, score_l])
        pre_rej_list = pre_rej_list + check_list

    # obtain raw preference data
    i = 0
    data_id = 0
    pre_rej_data = []
    while (i < n_data):
        optimize_prompt = START_PROMPT + \
            benchmark.get_graph_input_text(data[data_id][0]) + END_PROMPT
        graph = []
        while (1):
            graph.append(data[data_id][1])
            data_id += 1
            if data_id == len(data):
                break
            i_id = benchmark.get_problem_id(data[data_id][0])
            i_last_id = benchmark.get_problem_id(data[data_id-1][0])
            if i_id != i_last_id:
                break
        sub_data = [sub_l for sub_l in pre_rej_list if sub_l[0] == i]
        for sub_l in sub_data:
            if sub_l[2] >= len(graph) or sub_l[1] >= len(graph):
                continue
            chosen = graph[sub_l[2]]
            chosen_score = sub_l[3]
            rejected = graph[sub_l[1]]
            rejected_score = sub_l[4]
            dic = {"prompt": optimize_prompt, "chosen": chosen, "rejected": rejected,
                   "chosen_score": chosen_score, "rejected_score": rejected_score}
            if chosen_score != rejected_score:
                pre_rej_data.append(dic)
        i += 1

    # enhance sampling distribution by function d(x, y)
    if data_set == "HumanEval":
        n_sample = 600
    elif data_set == "AIME24":
        n_sample = 200
    elif data_set == "GPQA":
        n_sample = 1000
    else:
        n_sample = 2000

    sampled_pre_rej_data = sample_data(
        pre_rej_data, n_sample, lambda x, y: (x - y)**3)  # d(x, y) = (x - y)^3

    # output preference data
    output_dir = preference_data_dir(data_set, zcp)
    ensure_dir(output_dir)
    with open(preference_data_file(data_set, zcp, epoch), "wb") as f:
        pickle.dump(sampled_pre_rej_data, f)


async def run_evaluate(
    data_set: str,
    task_type: str,
    epoch: int,
    limit: int | None = None,
    zcp: str = "accuracy",
    agent_max_output_tokens: int = 1024,
):
    """供其他模块调用的评估入口，等价于原 main 的主体逻辑。"""
    bench_dic = ScoreFlow.params.bench_dic
    vali_num = 3 if zcp is None or zcp == 'accuracy' else 1

    # we use parallelization to speed up the evaluation process.
    # Now integrated directly instead of using subprocess
    parallel_id = 0
    await run_get_scores(
        data_set=data_set,
        task_type=task_type,
        epoch=int(epoch),
        parallel_id=parallel_id,
        limit=limit,
        zcp=zcp,
        agent_max_output_tokens=agent_max_output_tokens,
        vali_num=vali_num,
    )
    cprint("Evaluation Process Done!", color="green")

    # generate preference data (需要字符串形式的 epoch)
    generate_preference_data(bench_dic, data_set, str(
        epoch), task_type, zcp, limit, vali_num=vali_num)


def main():
    """CLI entrypoint：保留原有命令行用法。"""
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

    asyncio.run(
        run_evaluate(
            data_set=args.dataset,
            task_type=args.task,
            epoch=int(args.epoch),
            limit=args.limit,
            zcp=args.zcp,
            agent_max_output_tokens=args.agent_max_output_tokens,
        )
    )

    get_global_llm_manager().clear_all()
    logger.info("All LLM models have been cleared!")


if __name__ == "__main__":
    main()
