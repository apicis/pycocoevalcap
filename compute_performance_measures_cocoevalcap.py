""" This script computes the performance measure values from the result files. """

import json
import argparse
import os
import csv
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, util
from pycocotools.coco import COCO
from .eval import COCOEvalCap


def compute_scores(reference_caption_list, proposed_caption_list, dst_path):
    # Map lists into dictionaries (coco eval format)
    new_proposed = []
    new_reference = {'images': [], 'annotations': []}
    k = 1
    for i in range(len(reference_caption_list)):
        new_proposed.append({'image_id': "{}".format(k), 'caption': proposed_caption_list[i]})
        new_reference['images'].append({'id': "{}".format(k)})
        new_reference['annotations'].append(
            {'image_id': "{}".format(k), 'id': "{}".format(k), 'caption': reference_caption_list[i]})
        k += 1

    SBERT_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device="cuda")

    # Compute cosine similarity score between SBERT embeddings
    cosine_score = compute_cosine_similarity(proposed_caption_list, reference_caption_list, SBERT_model)

    # Create temporary files used by coco
    results_file = dst_path.replace(".csv", "_pred_temp.json")
    with open(results_file, 'w') as json_file:
        json.dump(new_proposed, json_file)
    annotation_file = dst_path.replace(".csv", "_ref_temp.json")
    with open(annotation_file, 'w') as json_file:
        json.dump(new_reference, json_file)

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # Remove temporary files
    os.remove(results_file)
    os.remove(annotation_file)

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.4f}')
    print(f'Cosine similarity: {cosine_score.mean():.4f}')

    # save output evaluation scores
    results_df = {}
    for metric, score in coco_eval.eval.items():
        results_df[metric] = f'{score:.4f}'
    results_df["Cosine Sim"] = f'{cosine_score.mean():.4f}'
    with open(dst_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_df.keys())
        writer.writeheader()  # Write the dictionary keys as the header
        writer.writerow(results_df)  # Write the values on the next line
    print("Results saved!")


def compute_cosine_similarity(proposed_caption_list, reference_caption_list, model):
    # Encode the captions
    proposed_embeddings = model.encode(proposed_caption_list)
    reference_embeddings = model.encode(reference_caption_list)

    # Calculate cosine similarity matrix between proposed and reference, and get diagonal
    sentences_cosine_similarity = torch.diagonal(util.cos_sim(proposed_embeddings, reference_embeddings)).cpu().numpy()
    return sentences_cosine_similarity


def convert_captions(proposed_captions, reference_captions):
    proposed_caption_list, reference_caption_list = [], []
    for key in proposed_captions.keys():
        if key not in reference_captions:
            continue
        reference = reference_captions[key]
        proposed = proposed_captions[key]['pseudocaption']
        if proposed is None:
            print("None")
            continue
        proposed_caption_list.append(proposed)
        reference_caption_list.append(reference)
    return proposed_caption_list, reference_caption_list


def read_csv(csv_path):
    dataset = pd.read_csv(csv_path)
    dataset_cleand = dataset.dropna(subset=['reference_caption'])

    return dataset_cleand.filename.tolist(), dataset_cleand.bounding_box.tolist(), dataset_cleand.reference_caption.tolist(), dataset_cleand.proposed_caption.tolist()


def read_json(json_path):
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def read_csv_caption(csv_path):
    dataset = pd.read_csv(csv_path)
    new_d = {}
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        episode_id = row.episode_id
        object_id = row.object_id
        new_d[f'({episode_id}, {object_id})'] = row.caption
    return new_d


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path',
                        type=str,
                        default="...",
                        )
    parser.add_argument('--json_path',
                        type=str,
                        default=None,
                        )
    parser.add_argument('--dst_path',
                        type=str,
                        default="...",
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    csv_path = args.csv_path
    json_path = args.json_path
    dst_path = args.dst_path

    if json_path is None:
        _, _, reference_caption_list, proposed_caption_list = read_csv(csv_path)
    else:
        proposed_captions = read_json(json_path)
        reference_captions = read_csv_caption(csv_path)
        proposed_caption_list, reference_caption_list = convert_captions(proposed_captions, reference_captions)

    assert len(reference_caption_list) == len(proposed_caption_list), "Number of references != number of captions"
    compute_scores(reference_caption_list, proposed_caption_list, dst_path)
