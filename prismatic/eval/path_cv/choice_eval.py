import argparse
import os
import json
from sklearn import metrics


subtype_questions = [
    "What subtype of cancer is this?",
    "What type of disease is this?",
]
question_lists = {
    'liver_subtype': subtype_questions
}

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for choice question', add_help=False)
    parser.add_argument('--gt', type=str, default="test.jsonl", help='path to groundtruth file', )
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    parser.add_argument('--dataset', type=str, default=None, help='key to find question list', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def evaluate(gt, pred, dataset):
    true_data = load_jsonl(gt)
    pred_data = load_jsonl(pred)

    true_list, pred_list = [], []
    for td, pd in zip(true_data, pred_data):
        assert td['question_id'] == pd['question_id'], f"{td['question_id']} != {pd['question_id']}"
        if dataset is None or any(td['text'].startswith(x) for x in question_lists[dataset]):#td['text'].startswith(question_lists[dataset]):
            true_list.append(td['answer'].lower()[0])
            pred_list.append(pd['text'].lower()[0])
    
    assert len(true_list) == len(pred_list)
    all_labels = sorted(set(true_list))
    print(f'True category number: {len(set(true_list))}, Pred category number: {len(set(pred_list))}')
    print(metrics.classification_report(true_list, pred_list, digits=4, zero_division=0, labels=all_labels, target_names=all_labels))
    print('Accuracy:', round(metrics.accuracy_score(true_list, pred_list), 4))
    print('Balanced ACC:', round(metrics.balanced_accuracy_score(true_list, pred_list), 4))


if __name__ == '__main__':
    args = parse_option()
    evaluate(args.gt, args.pred, args.dataset)
