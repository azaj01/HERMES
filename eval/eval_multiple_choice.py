import os
import json
import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_results(args):
    """Load results CSV from either --results_path or --save_dir/results.csv."""
    if args.results_path is not None:
        df = pd.read_csv(args.results_path)
        args.save_dir = os.path.dirname(args.results_path)
    else:
        df = pd.read_csv(os.path.join(args.save_dir, 'results.csv'))
    return df


# ---------------------------------------------------------------------------
# Average metric
# ---------------------------------------------------------------------------

def calc_average_metric(results, save_dir, metric):
    average_metric = sum(item[metric] for item in results) / len(results)
    print(f'#Samples: {len(results)}')
    print(f'Average {metric}: {average_metric:.2f}')
    print(f'save_dir: {save_dir}')


# ---------------------------------------------------------------------------
# OVOBench task-specific accuracy
# ---------------------------------------------------------------------------

PERCEPTION_TASKS = {
    'ACR': 'Action Recognition',
    'ATR': 'Attribute Recognition (AR)',
    'OJR': 'Object Recognition (OR)',
    'STU': 'Spatial Understanding (SU)',
    'OCR': 'OCR',
    'FPD': 'Future Prediction (FP)',
}

TRACING_TASKS = {
    'EPM': 'Episodic Memory',
    'HLD': 'Hallucination Detection',
    'ASI': 'Action Sequence Inference',
}


def calc_task_specific_accuracy(df, save_dir):
    """Calculate accuracy for each task and major category (OVOBench)."""
    all_tasks = {**PERCEPTION_TASKS, **TRACING_TASKS}

    print("\n" + "=" * 60)
    print("TASK-SPECIFIC ACCURACY (OVOBench)")
    print("=" * 60)

    task_accuracies = {}
    for code, name in all_tasks.items():
        task_data = df[df['task'] == code]
        if len(task_data) > 0:
            acc = task_data['qa_acc'].mean()
            task_accuracies[code] = acc
            print(f"{code} ({name}): {acc:.2f}% (n={len(task_data)})")

    print("\n" + "=" * 60)
    print("MAJOR CATEGORY ACCURACY")
    print("=" * 60)

    for category_name, task_map in [("Real-Time Visual Perception", PERCEPTION_TASKS),
                                     ("Backward Tracing", TRACING_TASKS)]:
        accs = [task_accuracies[c] for c in task_map if c in task_accuracies]
        if accs:
            cat_data = df[df['task'].isin(task_map.keys())]
            print(f"{category_name}: {sum(accs) / len(accs):.2f}% "
                  f"(n={len(cat_data)})")
            print(f"  - Includes: {', '.join(task_map.keys())}")

    print("=" * 60 + "\n")

    results_file = os.path.join(save_dir, 'task_breakdown.txt')
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TASK-SPECIFIC ACCURACY (OVOBench)\n")
        f.write("=" * 60 + "\n")
        for code, name in all_tasks.items():
            if code in task_accuracies:
                task_data = df[df['task'] == code]
                f.write(f"{code} ({name}): {task_accuracies[code]:.2f}% "
                        f"(n={len(task_data)})\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("MAJOR CATEGORY ACCURACY\n")
        f.write("=" * 60 + "\n")
        for category_name, task_map in [("Real-Time Visual Perception", PERCEPTION_TASKS),
                                         ("Backward Tracing", TRACING_TASKS)]:
            accs = [task_accuracies[c] for c in task_map if c in task_accuracies]
            if accs:
                cat_data = df[df['task'].isin(task_map.keys())]
                f.write(f"{category_name}: {sum(accs) / len(accs):.2f}% "
                        f"(n={len(cat_data)})\n")
                f.write(f"  - Includes: {', '.join(task_map.keys())}\n")
        f.write("=" * 60 + "\n")
    print(f"Task breakdown saved to: {results_file}")


# ---------------------------------------------------------------------------
# Prediction-choice error analysis
# ---------------------------------------------------------------------------

def analyze_pred_choice_errors(df, save_dir, debug=False):
    """Count and optionally print rows with invalid pred_choice values."""
    if 'pred_choice' not in df.columns:
        return
    valid_choices = set('ABCDEFGH')
    n_errors = 0
    for _, row in df.iterrows():
        pred_answer = row['pred_answer']
        pred_choice = row['pred_choice']
        if (isinstance(pred_answer, float)
                or len(str(pred_answer)) == 0
                or str(pred_choice)[0] not in valid_choices):
            n_errors += 1
            if debug:
                print(f'Video: {row["video_id"]}, Question: {row["question"]}, '
                      f'GT: {row["correct_choice"]}, Pred: {pred_choice}')
    print(f'%Errors: {n_errors / len(df) * 100:.2f}')

    results_file = os.path.join(save_dir, 'task_breakdown.txt')
    with open(results_file, 'a') as f:
        f.write(f'\n%Errors: {n_errors / len(df) * 100:.2f}')


# ---------------------------------------------------------------------------
# EgoSchema submission generation
# ---------------------------------------------------------------------------

def generate_egoschema_submission(df, save_dir):
    """Convert predictions to EgoSchema submission format (q_uid, answer)."""
    records = df.to_dict(orient='records')
    submission = []
    for r in records:
        choice = r.get('pred_choice', 'A')
        if choice not in ['A', 'B', 'C', 'D', 'E']:
            print(f"Invalid pred_choice: {choice}")
            choice = 'A'
        submission.append({
            'q_uid': r['video_id'],
            'answer': ord(choice) - ord('A'),
        })
    out_path = os.path.join(save_dir, 'submission.csv')
    pd.DataFrame(submission).to_csv(out_path, index=False)
    print(f'EgoSchema submission saved to: {out_path}')


# ---------------------------------------------------------------------------
# VideoMME duration-based evaluation
# ---------------------------------------------------------------------------

def eval_videomme_by_duration(df, anno_path, results_path):
    """Merge results with Video-MME metadata and report accuracy by duration."""
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno_data = json.load(f)
    duration_map = {item['video_id']: item['duration_category'] for item in anno_data}
    df['duration'] = df['video_id'].map(duration_map)
    merged = df

    duration_stats = merged.groupby('duration').agg(
        {'qa_acc': ['count', 'mean', 'sum']}
    ).round(2)
    duration_stats.columns = ['Total Questions', 'Average Accuracy (%)',
                              'Total Correct']
    duration_stats['Total Correct'] = \
        (duration_stats['Total Correct'] / 100).astype(int)

    total_questions = len(merged)
    total_correct = (merged['qa_acc'] == 100.0).sum()
    overall_accuracy = (total_correct / total_questions * 100
                        if total_questions > 0 else 0)

    output_file = results_path.replace('.csv', '.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        def write_line(text):
            print(text)
            f.write(text + '\n')

        write_line("=" * 60)
        write_line("Accuracy Statistics by Video Duration")
        write_line("=" * 60)
        write_line(str(duration_stats))
        write_line("=" * 60)

        write_line("\nOverall Statistics:")
        write_line(f"Total Questions: {total_questions}")
        write_line(f"Total Correct: {total_correct}")
        write_line(f"Overall Accuracy: {overall_accuracy:.2f}%")

        write_line("\nDetailed Statistics:")
        for duration in ['short', 'medium', 'long']:
            if duration in merged['duration'].values:
                subset = merged[merged['duration'] == duration]
                total = len(subset)
                correct = (subset['qa_acc'] == 100.0).sum()
                acc = correct / total * 100 if total > 0 else 0
                write_line(f"{duration.capitalize():8s}: {correct:4d}/{total:4d} "
                           f"correct, Accuracy: {acc:.2f}%")

    print(f"\nResults saved to: {output_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for multiple-choice benchmarks")

    sub = parser.add_subparsers(dest='command', help='Evaluation mode')

    # --- general: the default multi-metric evaluation ---
    p_gen = sub.add_parser('general',
                           help='Compute average metrics, OVOBench breakdown, '
                                'and error analysis')
    p_gen.add_argument('--save_dir', type=str)
    p_gen.add_argument('--results_path', type=str, default=None)
    p_gen.add_argument('--debug', action='store_true')

    # --- egoschema: generate submission CSV ---
    p_ego = sub.add_parser('egoschema',
                           help='Generate EgoSchema submission file')
    p_ego.add_argument('--save_dir', type=str)
    p_ego.add_argument('--results_path', type=str, default=None)

    # --- videomme: accuracy by video duration ---
    p_vmme = sub.add_parser('videomme',
                            help='Evaluate VideoMME results by duration')
    p_vmme.add_argument('--results_path', type=str, required=True)
    p_vmme.add_argument('--anno_path', type=str, required=True,
                        help='Path to videomme.json with duration_category')
    p_vmme.add_argument('--debug', action='store_true')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'general':
        df = load_results(args)
        results = df.to_dict(orient='records')
        calc_average_metric(results, args.save_dir, 'qa_acc')

        if 'task' in df.columns:
            calc_task_specific_accuracy(df, args.save_dir)

        analyze_pred_choice_errors(df, args.save_dir, debug=args.debug)

    elif args.command == 'egoschema':
        df = load_results(args)
        generate_egoschema_submission(df, args.save_dir)

    elif args.command == 'videomme':
        df = pd.read_csv(args.results_path)
        eval_videomme_by_duration(df, args.anno_path, args.results_path)


if __name__ == '__main__':
    main()
