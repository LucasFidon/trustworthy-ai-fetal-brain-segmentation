#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch.nn


class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        target = target.long()
        num_batches = inp.size()[0]
        num_classes = inp.size()[1]

        inp = inp.view(num_batches, num_classes, -1)
        target = target.view(num_batches, -1)

        ce = super(CrossentropyND, self).forward(inp, target)
        # Average over all dim except the batch dimension (if needed)
        # If self.reduction != 'none', ce is already a scalar.
        while len(ce.shape) > 1:
            ce = ce.mean(dim=-1)

        return ce

