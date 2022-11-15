class PairwiseHingeLoss(Loss):
    def __init__(self, weight=1., batch_axis=0, margin=1.0, **kwargs):
        super(PairwiseHingeLoss, self).__init__(weight, batch_axis, **kwargs)
        self.margin = margin

    def hybrid_forward(self, F, preds, labels):
        y_pos = F.slice(preds, begin=(0,), end=(None,), step=2)
        y_neg = F.slice(preds, begin=(1,), end=(None,), step=2)
        loss = F.relu(self.margin + y_neg - y_pos)
        ##loss = F.log(1+F.exp(-y_pos + y_neg))
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class PairwiseSampler(Sampler):
    def __init__(self, length, shuffle=False):
        assert length % 2 == 0, 'Pairwise dataset requires an even number of rows'
        self.length = length
        self.shuffle = shuffle

    def __iter__(self):
        even_indices = list(range(0, self.length, 2))
        if self.shuffle:
            random.shuffle(even_indices)
        indices = []
        for even_idx in even_indices:
            indices.append(even_idx)
            indices.append(even_idx + 1)
        return iter(indices)

    def __len__(self):
        return self.length