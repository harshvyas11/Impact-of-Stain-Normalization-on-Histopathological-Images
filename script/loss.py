import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment

def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    summation = tf.reduce_sum(y_true + y_pred)
    dice = (2. * intersection + smooth) / (summation + smooth)
    return dice

def dice_loss(y_true, y_pred):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)
  smooth = 1.0  # Adjust smooth value as needed
  intersection = tf.reduce_sum(y_true * y_pred)
  summation = tf.reduce_sum(y_true + y_pred)
  loss = 1 - (2. * intersection + smooth) / (summation + smooth)
  return loss

def get_fast_aji(true, pred):
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    pairwise_inter = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

    for true_id in true_id_list[1:]:
        true_id = int(true_id)  # Convert to integer
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            pred_id = int(pred_id)  # Convert to integer
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)

    if pairwise_iou.size == 0:
        return 0.0  # Return an appropriate value when there are no valid pairs

    if pairwise_iou.shape[0] == 0 or pairwise_iou.shape[1] == 0:
        return 0.0  # Ensure we have valid dimensions before proceeding

    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)

    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]

    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)
    paired_pred = list(paired_pred + 1)

    unpaired_true = np.array([int(idx) for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([int(idx) for idx in pred_id_list[1:] if idx not in paired_pred])

    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score


def get_fast_pq(true, pred, match_iou=0.5):
    paired_iou = []
    tp, fp, fn = 0, 0, 0

    pred_labels = np.unique(pred)
    pred_labels = pred_labels[pred_labels != 0]

    true_labels = np.unique(true)
    true_labels = true_labels[true_labels != 0]

    for pred_label in pred_labels:
        pred_mask = pred == pred_label
        max_iou = 0
        best_true_label = None
        for true_label in true_labels:
            true_mask = true == true_label
            iou = np.sum(pred_mask & true_mask) / np.sum(pred_mask | true_mask)
            if iou > max_iou:
                max_iou = iou
                best_true_label = true_label
        if max_iou >= match_iou:
            paired_iou.append(max_iou)
            tp += 1
            true_labels = true_labels[true_labels != best_true_label]
        else:
            fp += 1

    fn = len(true_labels)

    # Handle division by zero scenario
    if tp == 0 and (fp > 0 or fn > 0):
        dq = 0
    else:
        dq = tp / (tp + 0.5 * fp + 0.5 * fn +1.0e-6)

    sq = np.sum(paired_iou) / (tp + 1.0e-6) if tp > 0 else 0

    return dq * sq
