import os
import json
import copy
import argparse
from eval_utils import *

with open('./evaluation/all_failure_stage.json', 'r') as f:
    all_failure_stage = json.load(f)

def eval(model_name, model_url, dataset_folder, test_file, output_dir, llm_url, llm_model_name):
    stats_data = []
    results_data = {}

    with open(test_file, 'r') as f:
        annos_per_video = json.load(f)
    print(f"Loaded test QAs from {test_file}, {len(annos_per_video)} videos in total.")

    video_paths = []
    questions = []
    preds = []

    split_points = [0]
    for video_id, video_dict in annos_per_video.items():
        video_paths.extend([os.path.join(dataset_folder, video_dict['video'])] * len(video_dict['annos']))
        for question_type, qa in video_dict['annos'].items():
            if 'robofac' not in model_name.lower() :
                if question_type == 'High-level correction':
                    questions.append(qa[0]['value'] + " (Your answer should be a detailed textual suggestion of about two to three sentences.)")
                elif question_type == 'Low-level correction':
                    questions.append(qa[0]['value'] + " (Your answer should describe which direction (relative to the robot arm) and how much the robot arm should move to recover from the failure.)")
                elif question_type == 'Failure explanation':
                    questions.append(qa[0]['value'] + " (Your answer should be a detailed textual description of about two to three sentences.)")
                elif question_type == 'Task planning':
                    questions.append(qa[0]['value'] + " (Your answer should be a series of steps in the format of '1. ... 2. ... ...'.)")
                elif question_type == 'Task identification':
                    questions.append(qa[0]['value'] + " (Your answer should be a brief phrase that describes the task in the video.)")
            else:   
                questions.append(qa[0]['value'])
        split_points.append(split_points[-1] + len(video_dict['annos']))
    
    QA_num = len(questions)
    print(f"{QA_num} QAs in total.")
    
    # Evaluate the VLM model
    request_prompts = [question[8:] for question in questions]
    preds = asyncio.run(async_multi_request(request_prompts, frame_interval=30, video_paths=video_paths, model_name=model_name, url=model_url))
    with open('request_prompts.json', 'w') as f:
        json.dump(request_prompts, f, indent=4)
    with open('preds.json', 'w') as f:
        json.dump(preds, f, indent=4)
    print(f'{len(preds)} predictions generated')
    # split preds
    preds_per_video = []
    for i in range(len(split_points) - 1):
        preds_per_video.append(preds[split_points[i]:split_points[i+1]])
    assert len(preds_per_video) == len(annos_per_video), "The number of predictions is not equal to the number of videos."
    for i, video_id in enumerate(annos_per_video.keys()):
        pred_per_video = preds_per_video[i]
        annos_per_video[video_id]['preds'] = {}
        for j, question_type in enumerate(annos_per_video[video_id]['annos'].keys()):
            annos_per_video[video_id]['preds'][question_type] = pred_per_video[j]

    # Partition the test data
    annos_per_video_for_fuzzy_matching = copy.deepcopy(annos_per_video)
    annos_per_video_for_choice_question = {}
    to_be_deleted = []
    for video_id, video_dict in annos_per_video_for_fuzzy_matching.items():
        to_be_deleted_per_video = []
        for question_type in video_dict['annos'].keys():
            if question_type == 'Failure detection' or question_type == 'Failure identification' or question_type == 'Failure locating':
                to_be_deleted_per_video.append(question_type)
        if len(to_be_deleted_per_video) > 0:
            annos_per_video_for_choice_question[video_id] = {
                'video': video_dict['video'],
                'task': video_dict['task'],
                'annos': {},
                'preds': {}
            }
        for question_type in to_be_deleted_per_video:
            annos_per_video_for_choice_question[video_id]['annos'][question_type] = video_dict['annos'][question_type]
            annos_per_video_for_choice_question[video_id]['preds'][question_type] = video_dict['preds'][question_type]
            del video_dict['annos'][question_type]
            del video_dict['preds'][question_type]
        
        if len(video_dict['annos']) == 0:
            to_be_deleted.append(video_id)

    for video_id in to_be_deleted:
        del annos_per_video_for_fuzzy_matching[video_id]
    print(f'{len(annos_per_video_for_fuzzy_matching)} videos left for LLM fuzzy matching')
    print(f'{len(annos_per_video_for_choice_question)} videos left for choice question')

    # Evaluate by LLM
    questions = []
    refs = []
    preds = []
    split_points = [0]
    for video_id, video_dict in annos_per_video_for_fuzzy_matching.items():
        questions.extend([qa[0]['value'] for qa in video_dict['annos'].values()])
        refs.extend([qa[-1]['value'] for qa in video_dict['annos'].values()])
        preds.extend([video_dict['preds'][question_type] for question_type in video_dict['annos'].keys()])
        split_points.append(split_points[-1] + len(video_dict['annos']))

    eval_prompts = [make_eval_prompt(question, pred, ref) for question, pred, ref in zip(questions, preds, refs)]
    results = asyncio.run(async_multi_request(eval_prompts, model_name=llm_model_name, url=llm_url))
    results = [json.loads(result) for result in results]
    print(f'{len(results)} LLM evaluation results generated')


    # split results
    results_per_video = []
    for i in range(len(split_points) - 1):
        results_per_video.append(results[split_points[i]:split_points[i+1]])
    assert len(results_per_video) == len(annos_per_video_for_fuzzy_matching), f"length of results_per_video: {len(results_per_video)}, length of annos_per_video: {len(annos_per_video_for_fuzzy_matching)}"
    for i, video_id in enumerate(annos_per_video_for_fuzzy_matching.keys()):
        result_per_video = results_per_video[i]
        annos_per_video_for_fuzzy_matching[video_id]['results'] = {}
        for j, question_type in enumerate(annos_per_video_for_fuzzy_matching[video_id]['annos'].keys()):
            annos_per_video_for_fuzzy_matching[video_id]['results'][question_type] = result_per_video[j]

    # multi-choice question
    for video_id, video_dict in annos_per_video_for_choice_question.items():
        video_dict['results'] = {}
        ref_per_video = [qa[-1]['value'] for qa in video_dict['annos'].values()]
        pred_per_video = [video_dict['preds'][question_type] for question_type in video_dict['annos'].keys()]
        for question_type, ref, pred in zip(video_dict['annos'].keys(), ref_per_video, pred_per_video):
            if question_type == 'Failure detection':
                video_dict['results'][question_type] = 1 if pred.lower() in ref.lower() else 0
            elif question_type == 'Failure identification':
                video_dict['results'][question_type] = 1 if pred.lower() in ref.lower() else 0
            elif question_type == 'Failure locating':
                video_dict['results'][question_type] = 1 if pred.lower() in ref.lower() else 0
            else:
                raise ValueError(f"Invalid question type: {question_type}")

    # get results
    for video_id, video_dict in annos_per_video_for_fuzzy_matching.items():
        task = video_dict['task']
        if task not in results_data.keys():
            results_data[task] = {}
        for question_type in video_dict['annos'].keys():
            if question_type not in results_data[task].keys():
                results_data[task][question_type] = []
            results_data[task][question_type].append({
                "id": video_id,
                "video": video_dict['video'],
                "conversation": video_dict['annos'][question_type],
                "pred": video_dict['preds'][question_type],
                "result": video_dict['results'][question_type]
            })
    for video_id, video_dict in annos_per_video_for_choice_question.items():
        task = video_dict['task']
        if task not in results_data.keys():
            results_data[task] = {}
        for question_type in video_dict['annos'].keys():
            if question_type not in results_data[task].keys():
                results_data[task][question_type] = []
            results_data[task][question_type].append({
                "id": video_id,
                "video": video_dict['video'],
                "conversation": video_dict['annos'][question_type],
                "pred": video_dict['preds'][question_type],
                "result": video_dict['results'][question_type]
            })
    
    for task in results_data.keys():
        for question_type in results_data[task].keys():
            with open(os.path.join(output_dir, f'{task}_{question_type}_results.json'), 'w') as f:
                json.dump(results_data[task][question_type], f, indent=4)
            if question_type == 'Failure detection' or question_type == 'Failure identification' or question_type == 'Failure locating':
                score = 0
                for result in results_data[task][question_type]:
                    score += result['result']
                stats_data.append({
                    'task': task,
                    'question_type': question_type,
                    'score_overall': score / len(results_data[task][question_type])*100,
                    'num_qa': len(results_data[task][question_type])
                })
            else:
                score_correctness = 0
                score_relevance = 0
                score_completeness = 0
                for result in results_data[task][question_type]:
                    try:
                        score_correctness += result['result']['criteria']['correctness']['score']
                        score_relevance += result['result']['criteria']['relevance']['score']
                        score_completeness += result['result']['criteria']['completeness']['score']
                    except:
                        print(f"Error: {result['result']}")
                        raise ValueError(f"Invalid result: {result['result']}")
                num_qa = len(results_data[task][question_type])
                score_correctness = score_correctness / num_qa
                score_relevance = score_relevance / num_qa
                score_completeness = score_completeness / num_qa
                stats_data.append({
                    'task': task,
                    'question_type': question_type,
                    'score_correctness': score_correctness,
                    'score_relevance': score_relevance,
                    'score_completeness': score_completeness,
                    'score_overall': (score_correctness + score_relevance + score_completeness)/3*20,
                    'num_qa': num_qa
                })

    with open(os.path.join(output_dir, 'stats_data.json'), 'w') as f:
        json.dump(stats_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="./RoboFAC-7B", help="The model you want to evaluate.")
    parser.add_argument("--model_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--dataset_folder", type=str, default="./data/simulation_data", help="The data folder.")
    parser.add_argument("--test_dir", type=str, default="./data/test_qa_sim", help="The folder of the test set annotation files.")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--llm_model_name", type=str, default="gpt-4o", help="The name of the LLM model used for scoring the results.")
    parser.add_argument("--llm_url", type=str, default=None)
    args = parser.parse_args()
    
    for test_file in os.listdir(args.test_dir):
        output_dir = os.path.join(args.results_dir, args.model_name, test_file.split('.')[0])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        eval(args.model_name, args.model_url, args.dataset_folder, os.path.join(args.test_dir, test_file), output_dir, args.llm_url, args.llm_model_name)

    # Merge the results of each split
    results_merged = {}

    for folder in os.listdir(args.test_dir):
        video_list = []
        num_video = 0
        num_qa = 0
        for file in os.listdir(os.path.join(args.test_dir, folder)):
            if file.startswith('stats_data') or file.startswith('.'):
                continue
            with open(os.path.join(args.test_dir, folder, file), 'r') as f:
                results_split = json.load(f)
            for result in results_split:
                num_qa += 1
                video_list.append(result['video'])
        video_list = list(set(video_list))
        num_video = len(video_list)

        with open(os.path.join(args.test_dir, folder, 'stats_data.json'), 'r') as f:
            results_split = json.load(f)
        for result in results_split:
            key_name = result['task'] + '/' + result['question_type']
            if key_name not in results_merged:
                results_merged[key_name] = {}
                for key in result.keys():
                    if key != 'task' and key != 'question_type':
                        results_merged[key_name][key] = result[key]
            else:
                for key in result.keys():
                    if key != 'task' and key != 'question_type' and key != 'num_qa':
                        results_merged[key_name][key] = (result[key] * result['num_qa'] + results_merged[key_name][key] * results_merged[key_name]['num_qa']) / (result['num_qa'] + results_merged[key_name]['num_qa'])
                results_merged[key_name]['num_qa'] += result['num_qa']
            
    with open(os.path.join(args.test_dir, 'results_merged.json'), 'w') as f:
        json.dump(results_merged, f, indent=4)

    total_num_qa = 0
    for result_merged in results_merged.values():
        total_num_qa += result_merged['num_qa']
    print(f'total_num_qa: {total_num_qa}')