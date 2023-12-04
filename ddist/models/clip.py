import torch.nn as nn
import torchvision.transforms.v2 as T

class Clip0Shot(nn.Module):

    def __init__(self, labels, vision_model_type, model_pretrain_save):
        import open_clip
        super(Clip0Shot, self).__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(vision_model_type, pretrained=model_pretrain_save)
        self.resize = T.Resize((224, 224))

        tokenizer = open_clip.get_tokenizer(vision_model_type)
        label_tokens = tokenizer(['photo of a ' + label for label in labels])
        self.label_features = self.model.encode_text(label_tokens).detach()
        self.label_features /= self.label_features.norm(dim=-1, keepdim=True)
        self.label_features = nn.Parameter(self.label_features)

    def forward(self, x):
        image_features = self.model.encode_image(self.resize(x))
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.label_features.T
        return logits

# can add diff versions of the vision model here (like they've done for resnet)
def clip_vitb32(labels, model_pretrain_save='laion2b_s34b_b79k'):
    return Clip0Shot(labels, 'ViT-B-32', model_pretrain_save)
