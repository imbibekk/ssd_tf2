import tensorflow as tf
from utils import box_utils

class MultiBoxLoss:
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.
        Basically, MultiBox loss combines classification loss
        and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def _l1_smooth_loss(self, y_pred, y_true):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss)
    
    def __call__(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = tf.shape(confidence)[2]
        loss = - tf.nn.log_softmax(confidence, axis=2)[:, :, 0]
        mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        confidence = confidence[mask]
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum')
        classification_loss = cross_entropy(labels[mask], tf.reshape(confidence, (-1, num_classes))) 
        
        pos_mask = labels > 0
        predicted_locations = tf.reshape(predicted_locations[pos_mask], (-1, 4))
        gt_locations = tf.reshape(gt_locations[pos_mask], (-1,4))
        loc_loss = self._l1_smooth_loss(predicted_locations, gt_locations)
        num_pos = tf.shape(gt_locations)[0]
        num_pos = tf.cast(num_pos, loc_loss.dtype) 
        return loc_loss / num_pos, classification_loss / num_pos

