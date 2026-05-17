import torch
import torch.nn.functional as F


def cosine_distill_loss(student_feat, teacher_feat):
    student_feat = F.normalize(student_feat, dim=-1)
    teacher_feat = F.normalize(teacher_feat, dim=-1)
    return 1.0 - F.cosine_similarity(student_feat, teacher_feat, dim=-1).mean()


def clip_contrastive_loss(logits):
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0


def similarity_distill_loss(student_logits, teacher_logits, temperature: float = 2.0):
    temp = temperature
    loss_i2t = F.kl_div(
        F.log_softmax(student_logits / temp, dim=1),
        F.softmax(teacher_logits / temp, dim=1),
        reduction="batchmean",
    ) * (temp * temp)

    loss_t2i = F.kl_div(
        F.log_softmax(student_logits.t() / temp, dim=1),
        F.softmax(teacher_logits.t() / temp, dim=1),
        reduction="batchmean",
    ) * (temp * temp)

    return (loss_i2t + loss_t2i) / 2.0

