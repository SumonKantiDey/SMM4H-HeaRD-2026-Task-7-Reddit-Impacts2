import pandas as pd
import ast
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from typing import List, NamedTuple, Dict
from collections import defaultdict

def _to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        return ast.literal_eval(x)  # safe for list-literals
    raise TypeError(f"Unsupported cell type: {type(x)}")


def evaluate_test_strict_ner(
    df: pd.DataFrame,
    gold_col: str = "test",
    pred_col: str = "prediction",
    print_report: bool = True,
):
    """
    Entity-level STRICT NER evaluation using BIO tags (seqeval).
    Strict = exact span + exact label.

    df columns:
      - gold_col: gold BIO tags per sentence (list or stringified list)
      - pred_col: predicted BIO tags per sentence (list or stringified list)
    """
    y_true = df[gold_col].apply(_to_list).tolist()
    y_pred = df[pred_col].apply(_to_list).tolist()

    # print(y_true)
    # sanity checks
    if len(y_true) != len(y_pred):
        raise ValueError(f"Row mismatch: gold rows={len(y_true)} pred rows={len(y_pred)}")

    for i, (g, p) in enumerate(zip(y_true, y_pred)):
        if len(g) != len(p):
            raise ValueError(
                f"Token length mismatch at row {i}: gold={len(g)} pred={len(p)}\n"
                f"gold: {g}\n"
                f"pred: {p}"
            )
    # print("y_true = ", y_true)
    metrics = {
        "precision_strict": precision_score(y_true, y_pred),
        "recall_strict": recall_score(y_true, y_pred),
        "f1_strict": f1_score(y_true, y_pred),
        "accuracy_strict": accuracy_score(y_true, y_pred),
    }
    # print("metrics = ", metrics)

    if print_report:
        print("=== STRICT ENTITY NER METRICS (seqeval) ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("\n=== PER-LABEL REPORT ===")
        print(classification_report(y_true, y_pred, digits=4))

    return metrics

# Define the Entity NamedTuple
class Entity(NamedTuple):
    e_type: str
    start_offset: int
    end_offset: int

# Convert BIO tags to entities
def bio_to_entities(bio_tags: List[str]) -> List[Entity]:
    """
    Convert a single BIO-tagged sequence to a list of Entity objects.
    Handles fragmented entities correctly.
    """
    entities = []
    start = None
    entity_type = None

    for i, tag in enumerate(bio_tags):
        if tag.startswith("B-"):
            if entity_type is not None:  # Close previous entity
                entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=i - 1))
            entity_type = tag[2:]  # Extract type after "B-"
            start = i
        elif tag.startswith("I-") and entity_type == tag[2:]:
            # Continuation of the same entity
            continue
        elif tag.startswith("I-") and entity_type != tag[2:]:
            # Fragmented entity, treat as a new one
            if entity_type is not None:
                entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=i - 1))
            entity_type = tag[2:]
            start = i
        elif tag == "O":
            if entity_type is not None:  # Close current entity
                entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=i - 1))
                entity_type = None
                start = None

    if entity_type is not None:  # Handle last entity
        entities.append(Entity(e_type=entity_type, start_offset=start, end_offset=len(bio_tags) - 1))

    return entities

# Calculate relaxed overlap
def relaxed_overlap(entity1: Entity, entity2: Entity) -> float:
    """
    Calculate token overlap between two entities.
    Returns the absolute number of overlapping tokens.
    """
    if entity1.e_type != entity2.e_type:
        return 0  # Different types, no overlap

    return max(0, min(entity1.end_offset, entity2.end_offset) - max(entity1.start_offset, entity2.start_offset) + 1)

# Calculate F1 score per entity using absolute overlaps
def calculate_f1_per_entity_covering_all(gold_labels: List[List[str]], pred_labels: List[List[str]]) -> dict:
    """
    Calculate precision, recall, and F1 score for each entity type using absolute token overlap.
    Ensures all predicted entities contribute to evaluation.
    """
    aggregated_results = defaultdict(lambda: {"TP_overlap": 0, "Total_True_Length": 0, "Total_Pred_Length": 0})

    for gold, pred in zip(gold_labels, pred_labels):
        # Convert BIO tags to entities
        true_entities = bio_to_entities(gold)
        pred_entities = bio_to_entities(pred)

        # Process each true entity
        matched_pred_indices = set()
        for true_entity in true_entities:
            # print(f"{true_entity=}")
            for i, pred_entity in enumerate(pred_entities):
                # print(f"{pred_entity=}")
                if i in matched_pred_indices:
                    continue  # Skip already matched predictions
                overlap = relaxed_overlap(true_entity, pred_entity)
                if overlap > 0:
                    aggregated_results[true_entity.e_type]["TP_overlap"] += overlap
                    matched_pred_indices.add(i)
            aggregated_results[true_entity.e_type]["Total_True_Length"] += (true_entity.end_offset - true_entity.start_offset + 1)

        # Count lengths of all predicted entities
        for pred_entity in pred_entities:
            aggregated_results[pred_entity.e_type]["Total_Pred_Length"] += (pred_entity.end_offset - pred_entity.start_offset + 1)

    # Calculate precision, recall, and F1 for each entity type
    final_results = {}
    overall_tp_overlap = 0
    overall_true_length = 0
    overall_pred_length = 0
    
    for entity_type, values in aggregated_results.items():
        precision = values["TP_overlap"] / values["Total_Pred_Length"] if values["Total_Pred_Length"] > 0 else 0.0
        recall = values["TP_overlap"] / values["Total_True_Length"] if values["Total_True_Length"] > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        overall_tp_overlap += values["TP_overlap"]
        overall_true_length += values["Total_True_Length"]
        overall_pred_length += values["Total_Pred_Length"]

        final_results[entity_type] = {
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-Score": round(f1, 3),
            "Coverage": f"{values['TP_overlap']}/{values['Total_True_Length']}"
        }
    # Calculate overall precision, recall, and F1 - Micro
    overall_precision = overall_tp_overlap / overall_pred_length if overall_pred_length > 0 else 0.0
    overall_recall = overall_tp_overlap / overall_true_length if overall_true_length > 0 else 0.0
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    final_results["Overall"] = {
        "Precision": round(overall_precision, 3),
        "Recall": round(overall_recall, 3),
        "F1-Score": round(overall_f1, 3),
        "Coverage": f"{overall_tp_overlap}/{overall_true_length}"
    }
    return final_results

if __name__ == "__main__": 
    df = pd.read_excel("file.xlsx")
    
    # Get strict NER metrics (Micro F1)
    metrics = evaluate_test_strict_ner(df, gold_col='ner_tags_str', pred_col='prediction', print_report=False)
    micro_f1 = metrics['f1_strict']

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    ner_tags_str = df['ner_tags_str'].apply(_to_list).tolist()
    prediction = df['prediction'].apply(_to_list).tolist()
    results_per_entity = calculate_f1_per_entity_covering_all(ner_tags_str, prediction)

    # Output results
    print("F1 Score Results Per Entity:")
    for entity, metrics in results_per_entity.items():
        print(f"Entity Type: {entity}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        print()
    
    # Extract overall relaxed F1
    overall_relax_f1 = results_per_entity['Overall']['F1-Score']

    print("=" * 50)
    print(f"Micro F1 (Strict):     {micro_f1:.4f}")
    print(f"Overall Relax F1:      {overall_relax_f1:.4f}")
    print("=" * 50)

    