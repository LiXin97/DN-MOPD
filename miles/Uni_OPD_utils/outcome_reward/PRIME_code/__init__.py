import json
import traceback

from .utils import check_correctness as apps_check_correctness


def _normalize_text_cases(test_cases: dict[str, list]) -> dict[str, list]:
    # 格式适配：将 list 格式的 inputs/outputs 转换为 run_test 期望的字符串格式
    # 例如 inputs: [[100, 5], [83, 16]] -> ["100\n5", "83\n16"]
    #      outputs: [[95], [67]]         -> ["95", "67"]
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])
    if inputs and isinstance(inputs[0], list):
        test_cases["inputs"] = ["\n".join(json.dumps(x) for x in inp) for inp in inputs]
    if outputs and isinstance(outputs[0], list):
        test_cases["outputs"] = [json.dumps(out[0]) if len(out) == 1 else json.dumps(out) for out in outputs]
    return test_cases


def compute_score(
    completion: str,
    test_cases: dict[str, list] | str,
    continuous=False,
) -> tuple[float, list[dict[str, str]]]:
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    solution: str = completion.split("```python")[-1].split("```")[0]
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error:{e}")

        test_cases = _normalize_text_cases(test_cases)
        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        success: bool = False
        metadata_list = None
        try:
            res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x is True, res))
            metadata_list = metadata
            if success:
                return success, metadata_list
        except Exception:
            pass

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})

        if continuous:
            # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            # do not test all samples cuz some problems have enormous test cases
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=10, debug=False)
                try:
                    metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                except Exception:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)

                if test_case_id >= 9:
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1
            success = sum(map(lambda x: x is True, res_list)) / res_count
    except Exception:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    return success, metadata_list
