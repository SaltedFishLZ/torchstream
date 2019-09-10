from collections import OrderedDict

from torch import nn
from torchstream.models import TSN

from .backbones import resnet


class TSM(TSN):
    """
    Args:
        input_size (tuple): (T, H, W), shape of the input blob.
        channel == 3 only
    """
    def __init__(self, cls_num, input_size, base_model="resnet50",
                 dropout=0.8, partial_bn=True,
                 shift_enables=None, fold_div=8, shift_steps=1,
                 **kwargs):
        print("A new, general, TSM implementation")
        self.shift_enables = shift_enables
        self.fold_div = fold_div
        self.shift_steps = shift_steps
        super(TSM, self).__init__(cls_num=cls_num,
                                  input_size=input_size,
                                  base_model=base_model,
                                  dropout=dropout,
                                  partial_bn=partial_bn,
                                  **kwargs)

    def _prepare_base_model(self, base_model):
        """
        """
        if "resnet" in base_model:
            model_builder = getattr(resnet, base_model)
            self.base_model = model_builder(
                seg_num=self.input_size[0],
                shift_enables=self.shift_enables,
                fold_div= self.fold_div,
                shift_steps = self.shift_steps
            )
            # replace the classifier
            feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(OrderedDict([
                ("dropout", nn.Dropout(p=self.dropout)),
                ("fc", nn.Linear(feature_dim, self.cls_num))
                ]))
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def __repr__(self, idents=0):
        format_string = self.__class__.__name__
        format_string += "\n\tbase model:    {}"
        format_string += "\n\tclass number:  {}"
        format_string += "\n\t(T, H, W):     {}"
        format_string += "\n\tdropout ratio: {}"
        format_string += "\n\tshift enables: {}"
        format_string += "\n\tfold_div:      {}"
        format_string += "\n\tshift_steps:   {}"
        return format_string.format(self.base_model,
                                    self.cls_num,
                                    self.input_size,
                                    self.dropout,
                                    self.shift_enables,
                                    self.fold_div,
                                    self.shift_steps)


if __name__ == "__main__":
    net = TSM(cls_num=101, input_size=(8, 224, 224),
              shift_enables=[None,] * 3 + [True])
    print(net)
    # net = TSM(cls_num=101, input_size=(8, 224, 224), base_model="resnet18")
    # net = TSM(cls_num=101, input_size=(8, 224, 224), base_model="resnet34")
