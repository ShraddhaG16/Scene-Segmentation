import numpy as np
from sklearn.metrics import average_precision_score

------------------------------------------------------------------------------------------------------------------------------
def calc_avpr(g_dict, p_dict):
    """Average Precision (AP) for scene transitions
    Args:
        g_dict: Scene transition ground-truths.
        p_dict: Scene transition predictions.
        
    assert g_dict.keys() == p_dict.keys()

    ap_dict = dict()
    g = list()
    p = list()

    for imdb_id in g_dict.keys():
        ap_dict[imdb_id] = average_precision_score(g_dict[imdb_id], p_dict[imdb_id])
        g.append(g_dict[imdb_id])
        p.append(p_dict[imdb_id])

    mAP = sum(ap_dict.values()) / len(ap_dict)

    gt = np.concatenate(g)
    pr = np.concatenate(p)
    AP = average_precision_score(g, p)

    return AP, mAP, ap_dict

------------------------------------------------------------------------------------------------------------------------------

def calc_miou(g_dict, p_dict, shot_to_end_frame_dict, threshold=0.5):
    """Maximum IoU (Miou) for scene segmentation.
    Args:
        g_dict: Scene transition ground-truths.
        p_dict: Scene transition predictions.
        shot_to_end_frame_dict: End frame index for each shot.
        threshold: A threshold for filtering the predictions.

    def iou(x, y):
        s0, e0 = x
        s1, e1 = y
        smin, smax = (s0, s1) if s1 > s0 else (s1, s0)
        emin, emax = (e0, e1) if e1 > e0 else (e1, e0)
        return (emin - smax + 1) / (emax - smin + 1)

    def scene_frame_ranges(scene_transitions, shot_to_end_frame):
        end_shots = np.where(scene_transitions)[0]
        scenes = np.zeros((len(end_shots) + 1, 2), dtype=end_shots.dtype)
        scenes[:-1, 1] = shot_to_end_frame[end_shots]
        scenes[-1, 1] = shot_to_end_frame[len(scene_transitions)]
        scenes[1:, 0] = scenes[:-1, 1] + 1
        return scenes

    def miou(g_array, p_array, shot_to_end_frame):
        g_scenes = scene_frame_ranges(g_array, shot_to_end_frame)
        p_scenes = scene_frame_ranges(p_array >= threshold, shot_to_end_frame)
        assert g_scenes[-1, -1] == p_scenes[-1, -1]

        m = g_scenes.shape[0]
        n = p_scenes.shape[0]

        iou_table = np.zeros((m, n))

        j = 0
        for i in range(m):
            while p_scenes[j, 0] <= g_scenes[i, 1]:
                iou_table[i, j] = iou(g_scenes[i], p_scenes[j])
                if j < n - 1:
                    j += 1
                else:
                    break
            if p_scenes[j, 1] < g_scenes[i, 1] + 1:
                break
            if p_scenes[j, 0] > g_scenes[i, 1] + 1:
                j -= 1
        assert np.isnan(iou_table).sum() == 0
        assert iou_table.min() >= 0

        return (iou_table.max(axis=0).mean() + iou_table.max(axis=1).mean()) / 2

    assert g_dict.keys() == p_dict.keys()

    miou_dict = dict()

    for imdb_id in g_dict.keys():
        miou_dict[imdb_id] = miou(g_dict[imdb_id], p_dict[imdb_id], shot_to_end_frame_dict[imdb_id])
    mean_miou = sum(miou_dict.values()) / len(miou_dict)

    return mean_miou, miou_dict

------------------------------------------------------------------------------------------------------------------------------

def calc_prec_recall(g_dict, p_dict, threshold=0.5):
    """Precision, Recall and F1 for scene transitions at a given threshold.
    Args:
        g_dict: Scene transition ground-truths.
        p_dict: Scene transition predictions.
        threshold: A threshold to filter the predictions.


    def prec_recall(g_array, p_array):
        tp_fn = g_array == 1
        tp_fp = p_array >= threshold

        tps = (tp_fn & tp_fp).sum()

        precision = tps / tp_fp.sum()
        recall = tps / tp_fn.sum()

        return np.nan_to_num(precision), np.nan_to_num(recall)

    assert g_dict.keys() == p_dict.keys()

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()

    for imdb_id in gt_dict.keys():
        p, r = prec_recall(g_dict[imdb_id], p_dict[imdb_id])
        precision_dict[imdb_id] = p
        recall_dict[imdb_id] = r
        fscore_dict[imdb_id] = 2 * p * r / (p + r)

    n = len(gt_dict)
    mean_precision = sum(precision_dict.values()) / n
    mean_recall = sum(recall_dict.values()) / n
    mean_fscore = sum(fscore_dict.values()) / n

    return mean_precision, mean_recall, mean_fscore, precision_dict, recall_dict, fscore_dict


------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    

    import os
    import sys
    import glob
    import json
    import pickle

    data_dir = sys.argv[1]
    filenames = glob.glob(os.path.join(data_dir, "tt*.pkl"))

    print("Number of IMDB IDs:", len(filenames))

    g_dict = dict()
    p_dict = dict()
    shot_to_end_frame_dict = dict()

    for fn in filenames:
        x = pickle.load(open(fn, "rb"))

        g_dict[x["imdb_id"]] = x["scene_transition_boundary_ground_truth"]
        p_dict[x["imdb_id"]] = x["scene_transition_boundary_prediction"]
        shot_to_end_frame_dict[x["imdb_id"]] = x["shot_end_frame"]

    scores = dict()

    scores["AP"], scores["mAP"], _ = calc_avpr(g_dict, p_dict)
    scores["Miou"], _ = calc_miou(g_dict, p_dict, shot_to_end_frame_dict)
    scores["Precision"], scores["Recall"], scores["F1"], *_ = calc_prec_recall(g_dict, p_dict)

    print("Scores:", json.dumps(scores, indent=4))


------------------------------------------------------------------------------------------------------------------------------
