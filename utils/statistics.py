import math
import time


class LossStatistics:
    """
    Accumulator for loss staistics. Modified from OpenNMT
    """

    def __init__(self, loss=0.0, rec_loss=0.0, kl=0.0, rec_M=0.0,
                 n_batch=0, forward_time=0.0, loss_compute_time=0.0, backward_time=0.0):
        assert type(loss) is float or type(loss) is int
        # assert type(n_tokens) is int
        self.loss = loss
        self.rec_loss = rec_loss
        self.kl = kl
        self.rec_M = rec_M
        if math.isnan(loss) or math.isnan(rec_loss) or math.isnan(kl):
            raise ValueError("Loss is NaN")
        # self.n_tokens = n_tokens
        self.n_batch = n_batch
        self.forward_time = forward_time
        self.loss_compute_time = loss_compute_time
        self.backward_time = backward_time

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `LossStatistics` object

        Args:
            stat: another statistic object
        """
        self.loss += stat.loss
        self.rec_loss += stat.rec_loss
        self.kl += stat.kl
        self.rec_M += stat.rec_M
        if math.isnan(stat.loss) or math.isnan(stat.rec_loss) or math.isnan(stat.kl) or math.isnan(stat.rec_M):
            raise ValueError("Loss is NaN")
        # self.n_tokens += stat.n_tokens
        self.n_batch += stat.n_batch
        self.forward_time += stat.forward_time
        self.loss_compute_time += stat.loss_compute_time
        self.backward_time += stat.backward_time


    def total_time(self):
        return self.forward_time, self.loss_compute_time, self.backward_time

    def clear(self):
        self.loss = 0.0
        self.rec_loss = 0.0
        self.kl = 0.0
        self.rec_M = 0.0
        # self.n_tokens = 0
        self.n_batch = 0
        self.forward_time = 0.0
        self.loss_compute_time = 0.0
        self.backward_time = 0.0

