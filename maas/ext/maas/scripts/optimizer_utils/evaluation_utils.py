from maas.ext.maas.scripts.evaluator import Evaluator

class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    async def evaluate_graph_maas(self, optimizer, directory, data, initial=False, params: dict = None):
        evaluator = Evaluator(eval_path=directory, batch_size = optimizer.batch_size)

        # 获取score、avg_cost、total_cost和total_tokens
        score, avg_cost, total_cost, total_tokens = await evaluator.graph_evaluate( 
            optimizer.dataset,
            optimizer.graph,
            params,
            directory,
            is_test=False,
        )

        cur_round = optimizer.round
        # 传递真实的cost值和估算的token数量
        new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost=avg_cost, total_cost=total_cost, token=total_tokens)
        data.append(new_data)

        result_path = optimizer.data_utils.get_results_file_path(f"{optimizer.root_path}/train")
        optimizer.data_utils.save_results(result_path, data)

        return score
    
    async def evaluate_graph_test_maas(self, optimizer, directory, is_test=True, params: dict = None):
        evaluator = Evaluator(eval_path=directory, batch_size = optimizer.batch_size)
        
        # 返回score、avg_cost、total_cost和total_tokens
        return await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            params,
            directory,
            is_test=is_test,
        )
    