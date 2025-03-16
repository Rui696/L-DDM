import torch
import h5py

class MatRead(object):
    '''
    Load .mat file
    Output:
        - a_field: [nsamples, 1, resolution, resolution]
        - u_field: [nsamples, 1, resolution, resolution]
        - boundary: [nsamples, 4 * resolution - 4]
    '''
    def __init__(self, file_path):
        super(MatRead).__init__()
        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_data(self):
        return torch.tensor(self.data['a_field'][:], dtype=torch.float32).permute(2, 1, 0).unsqueeze(1), \
               torch.tensor(self.data['u_field'][:], dtype=torch.float32).permute(2, 1, 0).unsqueeze(1), \
               torch.tensor(self.data['ux_field'][:], dtype=torch.float32).permute(2, 1, 0).unsqueeze(1), \
               torch.cat((torch.tensor(self.data['u_T'][:-1], dtype=torch.float32),
                          torch.tensor(self.data['u_R'][1:], dtype=torch.float32),
                          torch.tensor(self.data['u_B'][1:], dtype=torch.float32),
                          torch.tensor(self.data['u_L'][:-1], dtype=torch.float32)), dim=0).permute(1, 0)


class MaxMinNormalizer(object):
    '''
    MaxMin normalizer, 0-1
    '''
    def __init__(self, x):
        super(MaxMinNormalizer, self).__init__()

        self.max = torch.max(x)
        self.min = torch.min(x)

    def encode(self, x):
        x = (x - self.min) / (self.max - self.min)
        return x

    def decode(self, x):
        x = (x * (self.max - self.min)) + self.min
        return x

    def cuda(self):
        self.max = self.max.cuda()
        self.min = self.min.cuda()

    def to(self, device):
        self.max = self.max.to(device)
        self.min = self.min.to(device)


class MaxMinNormalizerMultiple(object):
    '''
    MaxMin normalizer for multiple inputs
    '''
    def __init__(self, x):
        super(MaxMinNormalizerMultiple, self).__init__()
        self.max = -1000
        self.min = 1000
        for i in x:
            maxx = torch.max(i)
            minn = torch.min(i)
            self.max = max(self.max, maxx)
            self.min = min(self.min, minn)

    def encode(self, x):
        x = (x - self.min) / (self.max - self.min)
        return x

    def decode(self, x):
        x = (x * (self.max - self.min)) + self.min
        return x

    def cuda(self):
        self.max = self.max.cuda()
        self.min = self.min.cuda()

    def to(self, device):
        self.max = self.max.to(device)
        self.min = self.min.to(device)


class LpLoss(object):
    '''
    Lp loss function
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True, absolute=False):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[-1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1),
                                                          self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def forward(self, x, y):
        return self.abs(x, y) if self.absolute else self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)


