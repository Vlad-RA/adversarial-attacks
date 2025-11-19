import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import numpy as np
import pandas as pd

# -------------------------------------------------------
# Вспомогательная функция для безопасной загрузки изображений
# -------------------------------------------------------

def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Не удалось скачать {url}: HTTP {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        raise RuntimeError(
            f"URL {url} вернул не картинку, а тип '{content_type}'. "
            f"Открой ссылку в браузере и посмотри, что там на самом деле."
        )

    try:
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except UnidentifiedImageError:
        raise RuntimeError(
            f"PIL не смог распознать файл как изображение по URL {url}. "
            "Скорее всего, контент повреждён или это не JPEG/PNG."
        )

    return img


torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

image_urls = [
    "https://ids.si.edu/ids/deliveryService?id=NZP-20140217-001ES&max_w=800",
    "https://ids.si.edu/ids/deliveryService?id=NZP-20140817-6602RG-000003&max_w=800",
    "https://ids.si.edu/ids/deliveryService?id=NZP-20100323-121MM&max_w=800",
    "https://ids.si.edu/ids/deliveryService?id=NZP-20090210-668MM&max_w=800",
    "https://ids.si.edu/ids/deliveryService?id=NZP-20080117-316MM&max_w=800",
    "https://ids.si.edu/ids/deliveryService?id=NZP-20100819-091MM&max_w=800",
    "https://ids.si.edu/ids/deliveryService?id=NZP-20070122-443MM&max_w=800",
    "https://ids.si.edu/ids/deliveryService?id=NZP-20091222-416MM&max_w=800",
    "https://ids.si.edu/ids/deliveryService?id=NZP-20181031-148SB&max_w=800",
    "https://ids.si.edu/ids/deliveryService?id=NZP-20070723-148MM&max_w=800",
]

pil_images: List[Image.Image] = []
X_list: List[torch.Tensor] = []

for idx, url in enumerate(image_urls):
    print(f"Loading image {idx}: {url}")
    img = load_image_from_url(url)
    pil_images.append(img)
    x_i = transform(img).unsqueeze(0).to(device)
    X_list.append(x_i)

X = torch.cat(X_list, dim=0)
num_images = X.size(0)
print(f"Всего изображений: {num_images}")


def grad_to_heatmap(grad_tensor: torch.Tensor) -> torch.Tensor:
    grad_mag = grad_tensor.abs().sum(dim=1).squeeze(0)
    grad_norm = grad_mag / (grad_mag.max() + 1e-8)
    return grad_norm


def compute_heatmap_for_input(model: torch.nn.Module, x_in: torch.Tensor, y_target=None):
    x_var = x_in.clone().detach().to(device).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    out = model(x_var)
    if y_target is None:
        y_t = out.argmax(dim=1)
    else:
        y_t = y_target
    loss = F.cross_entropy(out, y_t)
    loss.backward()
    grad = x_var.grad.detach()
    return grad_to_heatmap(grad)


def distortion_metrics(x_orig: torch.Tensor, x_adv: torch.Tensor):
    with torch.no_grad():
        delta = x_adv - x_orig
        linf = delta.abs().view(delta.size(0), -1).max(dim=1).values
        l2 = delta.view(delta.size(0), -1).norm(p=2, dim=1)
        l1 = delta.view(delta.size(0), -1).norm(p=1, dim=1)
        mse = (delta ** 2).view(delta.size(0), -1).mean(dim=1)
        psnr = -10.0 * torch.log10(mse + 1e-12)
    return linf.cpu().numpy(), l2.cpu().numpy(), l1.cpu().numpy(), psnr.cpu().numpy()


def project_onto_l2_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
    flat = delta.view(delta.size(0), -1)
    norms = torch.norm(flat, p=2, dim=1, keepdim=True)
    factor = torch.clamp(eps / (norms + 1e-12), max=1.0)
    projected = flat * factor
    return projected.view_as(delta)

def fgsm_attack(model, x_in, y, eps: float, targeted: bool = False, y_target: torch.Tensor | None = None):
    if eps == 0.0:
        return x_in.clone().detach()
    x_adv = x_in.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    out = model(x_adv)
    if targeted:
        assert y_target is not None
        loss = -F.cross_entropy(out, y_target)
    else:
        loss = F.cross_entropy(out, y)
    loss.backward()
    grad = x_adv.grad.sign()
    with torch.no_grad():
        direction = -grad if targeted else grad
        x_adv = torch.clamp(x_adv + eps * direction, 0, 1)
    return x_adv.detach()


def bim_attack(model, x_in, y, eps: float, steps: int = 10, targeted: bool = False, y_target: torch.Tensor | None = None):
    if eps == 0.0:
        return x_in.clone().detach()

    x_adv = x_in.clone().detach()
    alpha = eps / steps

    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(x_adv)
        if targeted:
            assert y_target is not None
            loss = -F.cross_entropy(out, y_target)
        else:
            loss = F.cross_entropy(out, y)
        loss.backward()
        grad = x_adv.grad.sign()
        with torch.no_grad():
            direction = -grad if targeted else grad
            x_adv = x_adv + alpha * direction
            eta = torch.clamp(x_adv - x_in, min=-eps, max=eps)
            x_adv = torch.clamp(x_in + eta, 0, 1)
    return x_adv.detach()


def pgd_attack(model, x_in, y, eps: float, steps: int = 10, alpha_ratio: float = 0.25, targeted: bool = False, y_target: torch.Tensor | None = None, random_start: bool = True):
    if eps == 0.0:
        return x_in.clone().detach()

    if random_start:
        x_adv = x_in + torch.empty_like(x_in).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0, 1)
    else:
        x_adv = x_in.clone().detach()

    alpha = eps * alpha_ratio

    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(x_adv)
        if targeted:
            assert y_target is not None
            loss = -F.cross_entropy(out, y_target)
        else:
            loss = F.cross_entropy(out, y)
        loss.backward()
        grad = x_adv.grad.sign()
        with torch.no_grad():
            direction = -grad if targeted else grad
            x_adv = x_adv + alpha * direction
            eta = torch.clamp(x_adv - x_in, min=-eps, max=eps)
            x_adv = torch.clamp(x_in + eta, 0, 1)
    return x_adv.detach()


def mi_fgsm_attack(model, x_in, y, eps: float, steps: int = 10, mu: float = 1.0):
    if eps == 0.0:
        return x_in.clone().detach()

    x_adv = x_in.clone().detach()
    g = torch.zeros_like(x_adv)
    alpha = eps / steps

    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(x_adv)
        loss = F.cross_entropy(out, y)
        loss.backward()
        grad = x_adv.grad

        grad_mean = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-12
        grad_normed = grad / grad_mean
        g = mu * g + grad_normed

        with torch.no_grad():
            x_adv = x_adv + alpha * g.sign()
            eta = torch.clamp(x_adv - x_in, min=-eps, max=eps)
            x_adv = torch.clamp(x_in + eta, 0, 1)

    return x_adv.detach()


def pgd_l2_attack(model, x_in, y, eps: float, steps: int = 20, alpha_ratio: float = 0.25):
    if eps == 0.0:
        return x_in.clone().detach()

    delta = torch.randn_like(x_in)
    delta = project_onto_l2_ball(delta, eps)
    x_adv = torch.clamp(x_in + delta, 0, 1)
    alpha = eps * alpha_ratio

    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(x_adv)
        loss = F.cross_entropy(out, y)
        loss.backward()
        grad = x_adv.grad
        grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True)
        grad_direction = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-12)
        with torch.no_grad():
            x_adv = x_adv + alpha * grad_direction
            delta = x_adv - x_in
            delta = project_onto_l2_ball(delta, eps)
            x_adv = torch.clamp(x_in + delta, 0, 1)
    return x_adv.detach()


def cw_l2_attack(model, x_in, y, c: float = 1e-3, steps: int = 200, lr: float = 0.01, kappa: float = 0.0):
    x_in = x_in.clone().detach()
    w = torch.atanh((x_in * 2 - 1) * 0.999999).detach()
    w.requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        x_adv = torch.tanh(w) * 0.5 + 0.5
        logits = model(x_adv)
        y_one_hot = F.one_hot(y, num_classes=logits.size(1)).float()
        real = torch.sum(y_one_hot * logits, dim=1)
        other = torch.max((1 - y_one_hot) * logits - y_one_hot * 1e4, dim=1).values
        f = torch.clamp(real - other, min=-kappa)
        l2 = ((x_adv - x_in) ** 2).view(x_in.size(0), -1).sum(dim=1)
        loss = l2 + c * f
        loss = loss.mean()
        loss.backward()
        optimizer.step()

    x_adv = torch.tanh(w) * 0.5 + 0.5
    return x_adv.detach()


def deepfool_attack(model, x_in, max_iter: int = 50, overshoot: float = 0.02, num_classes: int = 10):
    assert x_in.size(0) == 1, "DeepFool реализован для батча размера 1"
    x_adv = x_in.clone().detach().requires_grad_(True)
    with torch.no_grad():
        initial_label = int(model(x_adv).argmax(dim=1).item())

    for _ in range(max_iter):
        outputs = model(x_adv)
        logits = outputs[0]
        _, indices = torch.topk(logits, k=min(num_classes, logits.numel()))
        grad_orig = torch.autograd.grad(logits[initial_label], x_adv, retain_graph=True)[0]
        min_pert = None
        w_best = None
        for cls in indices:
            cls_id = int(cls.item())
            if cls_id == initial_label:
                continue
            grad_cls = torch.autograd.grad(logits[cls_id], x_adv, retain_graph=True)[0]
            w_k = grad_cls - grad_orig
            f_k = logits[cls_id] - logits[initial_label]
            w_norm = torch.norm(w_k.view(1, -1), p=2)
            pert_k = torch.abs(f_k) / (w_norm + 1e-12)
            if min_pert is None or pert_k < min_pert:
                min_pert = pert_k
                w_best = w_k
        if w_best is None:
            break
        r_i = (min_pert + 1e-4) * w_best / (torch.norm(w_best.view(1, -1), p=2) + 1e-12)
        with torch.no_grad():
            x_adv = x_adv.detach() + (1 + overshoot) * r_i
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv.requires_grad_(True)
        new_label = int(model(x_adv).argmax(dim=1).item())
        if new_label != initial_label:
            break
    return x_adv.detach()

def apply_gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    padding = kernel_size // 2
    coords = torch.arange(kernel_size, device=x.device) - padding
    grid_y, grid_x = torch.meshgrid(coords, coords)
    kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, kernel, padding=padding, groups=x.size(1))


def apply_median_filter(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    x_pad = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    unfolded = x_pad.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    median = unfolded.contiguous().reshape(x.size(0), x.size(1), x.size(2), x.size(3), -1)
    median = median.median(dim=-1).values
    return median


def reduce_bit_depth(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    levels = 2 ** bits - 1
    return torch.round(x * levels) / levels


def jpeg_compress(x: torch.Tensor, quality: int = 50) -> torch.Tensor:
    clamp = torch.clamp(x.detach().cpu(), 0, 1)
    compressed = []
    to_pil = T.ToPILImage()
    to_tensor = T.ToTensor()
    for sample in clamp:
        img = to_pil(sample)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        rec = Image.open(buf).convert("RGB")
        compressed.append(to_tensor(rec))
    stacked = torch.stack(compressed, dim=0)
    return stacked.to(x.device)


def identity_defense(x: torch.Tensor) -> torch.Tensor:
    return x


defense_configs: List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
    ("none", identity_defense),
    ("gaussian_blur", lambda x: apply_gaussian_blur(x, kernel_size=5, sigma=1.0)),
    ("median_filter", lambda x: apply_median_filter(x, kernel_size=3)),
    ("bit_depth_4bit", lambda x: reduce_bit_depth(x, bits=4)),
    ("jpeg_50", lambda x: jpeg_compress(x, quality=50)),
]

IMAGENET_CATEGORIES = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


def load_madry_resnet50_linf():
    try:
        model = torch.hub.load("facebookresearch/robustness", "resnet50", pretrained=True)
        print("Загружена робастная ResNet50 (Madry, Linf)")
        return model
    except Exception as exc:
        print(f"Не удалось загрузить робастную модель Linf: {exc}. Используем ResNet50 V2")
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)


def load_madry_resnet50_l2():
    try:
        model = torch.hub.load(
            "facebookresearch/robustness",
            "resnet50",
            pretrained=True,
            threat_model="L2",
        )
        print("Загружена робастная ResNet50 (Madry, L2)")
        return model
    except Exception as exc:
        print(f"Не удалось загрузить робастную модель L2: {exc}. Используем DenseNet121")
        return models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)


models_to_run = [
    {
        "name": "ResNet50",
        "builder": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        "categories": models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"],
    },
    {
        "name": "VGG16",
        "builder": lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        "categories": models.VGG16_Weights.IMAGENET1K_V1.meta["categories"],
    },
    {
        "name": "DenseNet121",
        "builder": lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
        "categories": models.DenseNet121_Weights.IMAGENET1K_V1.meta["categories"],
    },
    {
        "name": "Robust ResNet50 Linf",
        "builder": load_madry_resnet50_linf,
        "categories": IMAGENET_CATEGORIES,
    },
    {
        "name": "Robust ResNet50 L2",
        "builder": load_madry_resnet50_l2,
        "categories": IMAGENET_CATEGORIES,
    },
]

attack_names = [
    "FGSM",
    "BIM",
    "PGD",
    "MI-FGSM",
    "PGD_L2",
    "CW_L2",
    "DeepFool",
    "AutoAttackLite",
]
ensemble_base_attacks = ["FGSM", "BIM", "PGD", "MI-FGSM", "PGD_L2", "CW_L2", "DeepFool"]

def evaluate_attack(
    *,
    model: torch.nn.Module,
    model_name: str,
    image_id: int,
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    attack_name: str,
    mode: str,
    epsilon: float,
    orig_label_id: int,
    orig_label_name: str,
    classes_txt: List[str],
    defense_records: Dict[str, Dict[str, Dict]],
    results: List[Dict],
):
    linf, l2, l1, psnr = distortion_metrics(x_orig, x_adv)
    linf_val = float(linf[0])
    l2_val = float(l2[0])
    l1_val = float(l1[0])
    psnr_val = float(psnr[0])

    for defense_name, defense_fn in defense_configs:
        with torch.no_grad():
            x_eval = defense_fn(x_adv)
            out = model(x_eval)
            probs = F.softmax(out, dim=1)[0]
        pred_id = int(probs.argmax().item())
        record = {
            "model": model_name,
            "image_id": image_id,
            "attack": attack_name,
            "source_attack": attack_name,
            "mode": mode,
            "epsilon": epsilon,
            "orig_label_id": orig_label_id,
            "orig_label_name": orig_label_name,
            "adv_pred_id": pred_id,
            "adv_pred_name": classes_txt[pred_id],
            "prob_orig_class": float(probs[orig_label_id].item()),
            "prob_top1": float(probs.max().item()),
            "linf": linf_val,
            "l2": l2_val,
            "l1": l1_val,
            "psnr": psnr_val,
            "defense": defense_name,
        }
        results.append(record)
        if mode == "untargeted":
            defense_records[defense_name][attack_name] = record

def run_all():
    dfs_for_excel: Dict[str, pd.DataFrame] = {}
    eps_range = np.linspace(0.0, 0.02, 80)

    for model_cfg in models_to_run:
        model_name = model_cfg["name"]
        print("\n" + "=" * 90)
        print(f"МОДЕЛЬ: {model_name}")
        print("=" * 90)
        model = model_cfg["builder"]().to(device).eval()
        classes_txt = model_cfg.get("categories", IMAGENET_CATEGORIES)

        labels: List[int] = []
        orig_probs: List[float] = []

        for i in range(num_images):
            xi = X[i:i+1]
            with torch.no_grad():
                out_i = model(xi)
                probs_i = F.softmax(out_i, dim=1)[0]
            label_i = int(probs_i.argmax().item())
            prob_i = float(probs_i[label_i].item())
            labels.append(label_i)
            orig_probs.append(prob_i)
            print(f"Image {i}: predicted {classes_txt[label_i]} (p={prob_i:.3f})")

        Y = torch.tensor(labels, device=device)
        x_target = X[0:1]
        y_target_true = Y[0:1]
        target_label_id = labels[0]
        target_label = classes_txt[target_label_id]
        target_prob = orig_probs[0]
        print(f"\nTarget image original for {model_name}: {target_label} (p = {target_prob:.4f})")

        with torch.no_grad():
            out_target = model(x_target)
            probs_target = F.softmax(out_target, dim=1)[0]
            top2 = torch.topk(probs_target, k=2)
            target_ids = top2.indices.cpu().numpy().tolist()
        if target_ids[0] == target_label_id and len(target_ids) > 1:
            targeted_class_id = target_ids[1]
        else:
            targeted_class_id = target_ids[0]
        y_target_cls = torch.tensor([targeted_class_id], device=device)
        print(f"Targeted class for {model_name}: {classes_txt[targeted_class_id]}")

        results: List[Dict] = []

        for eps in eps_range:
            eps_f = float(eps)
            for i in range(num_images):
                xi = X[i:i+1]
                yi = Y[i:i+1]
                orig_id = labels[i]
                orig_name = classes_txt[orig_id]

                defense_records: Dict[str, Dict[str, Dict]] = {name: {} for name, _ in defense_configs}

                x_adv_fgsm = fgsm_attack(model, xi, yi, eps_f, targeted=False)
                evaluate_attack(
                    model=model,
                    model_name=model_name,
                    image_id=i,
                    x_orig=xi,
                    x_adv=x_adv_fgsm,
                    attack_name="FGSM",
                    mode="untargeted",
                    epsilon=eps_f,
                    orig_label_id=orig_id,
                    orig_label_name=orig_name,
                    classes_txt=classes_txt,
                    defense_records=defense_records,
                    results=results,
                )

                x_adv_fgsm_t = fgsm_attack(model, xi, yi, eps_f, targeted=True, y_target=y_target_cls)
                evaluate_attack(
                    model=model,
                    model_name=model_name,
                    image_id=i,
                    x_orig=xi,
                    x_adv=x_adv_fgsm_t,
                    attack_name="FGSM",
                    mode="targeted",
                    epsilon=eps_f,
                    orig_label_id=orig_id,
                    orig_label_name=orig_name,
                    classes_txt=classes_txt,
                    defense_records=defense_records,
                    results=results,
                )

                x_adv_bim = bim_attack(model, xi, yi, eps_f, steps=10, targeted=False)
                evaluate_attack(
                    model=model,
                    model_name=model_name,
                    image_id=i,
                    x_orig=xi,
                    x_adv=x_adv_bim,
                    attack_name="BIM",
                    mode="untargeted",
                    epsilon=eps_f,
                    orig_label_id=orig_id,
                    orig_label_name=orig_name,
                    classes_txt=classes_txt,
                    defense_records=defense_records,
                    results=results,
                )

                x_adv_pgd = pgd_attack(model, xi, yi, eps_f, steps=10, alpha_ratio=0.25, targeted=False)
                evaluate_attack(
                    model=model,
                    model_name=model_name,
                    image_id=i,
                    x_orig=xi,
                    x_adv=x_adv_pgd,
                    attack_name="PGD",
                    mode="untargeted",
                    epsilon=eps_f,
                    orig_label_id=orig_id,
                    orig_label_name=orig_name,
                    classes_txt=classes_txt,
                    defense_records=defense_records,
                    results=results,
                )

                x_adv_mi = mi_fgsm_attack(model, xi, yi, eps_f, steps=10, mu=1.0)
                evaluate_attack(
                    model=model,
                    model_name=model_name,
                    image_id=i,
                    x_orig=xi,
                    x_adv=x_adv_mi,
                    attack_name="MI-FGSM",
                    mode="untargeted",
                    epsilon=eps_f,
                    orig_label_id=orig_id,
                    orig_label_name=orig_name,
                    classes_txt=classes_txt,
                    defense_records=defense_records,
                    results=results,
                )

                x_adv_pgd_l2 = pgd_l2_attack(model, xi, yi, eps_f, steps=15, alpha_ratio=0.3)
                evaluate_attack(
                    model=model,
                    model_name=model_name,
                    image_id=i,
                    x_orig=xi,
                    x_adv=x_adv_pgd_l2,
                    attack_name="PGD_L2",
                    mode="untargeted",
                    epsilon=eps_f,
                    orig_label_id=orig_id,
                    orig_label_name=orig_name,
                    classes_txt=classes_txt,
                    defense_records=defense_records,
                    results=results,
                )

                c_weight = max(eps_f, 1e-4)
                x_adv_cw = cw_l2_attack(model, xi, yi, c=c_weight, steps=60, lr=0.01)
                evaluate_attack(
                    model=model,
                    model_name=model_name,
                    image_id=i,
                    x_orig=xi,
                    x_adv=x_adv_cw,
                    attack_name="CW_L2",
                    mode="untargeted",
                    epsilon=eps_f,
                    orig_label_id=orig_id,
                    orig_label_name=orig_name,
                    classes_txt=classes_txt,
                    defense_records=defense_records,
                    results=results,
                )

                overshoot = max(0.005, eps_f + 1e-4)
                x_adv_deepfool = deepfool_attack(model, xi, max_iter=30, overshoot=overshoot, num_classes=25)
                evaluate_attack(
                    model=model,
                    model_name=model_name,
                    image_id=i,
                    x_orig=xi,
                    x_adv=x_adv_deepfool,
                    attack_name="DeepFool",
                    mode="untargeted",
                    epsilon=eps_f,
                    orig_label_id=orig_id,
                    orig_label_name=orig_name,
                    classes_txt=classes_txt,
                    defense_records=defense_records,
                    results=results,
                )

                for defense_name, attack_map in defense_records.items():
                    candidates = [attack_map[name] for name in ensemble_base_attacks if name in attack_map]
                    if not candidates:
                        continue
                    worst = min(candidates, key=lambda rec: rec["prob_orig_class"])
                    auto_record = dict(worst)
                    auto_record["attack"] = "AutoAttackLite"
                    auto_record["source_attack"] = worst["attack"]
                    results.append(auto_record)

        df_model = pd.DataFrame(results)
        dfs_for_excel[model_name] = df_model

        df_baseline = df_model[(df_model["mode"] == "untargeted") & (df_model["defense"] == "none")]

        group_prob = (
            df_baseline
            .groupby(["attack", "epsilon"])["prob_orig_class"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(9, 5))
        for attack_name in attack_names:
            sub = group_prob[group_prob["attack"] == attack_name]
            if len(sub) == 0:
                continue
            plt.plot(sub["epsilon"], sub["prob_orig_class"], label=attack_name)

        plt.xlabel("epsilon (сила атаки)")
        plt.ylabel("Средняя уверенность в исходном классе")
        plt.title(
            f"{model_name}: средняя уверенность модели (untargeted)\n"
            f"от силы атаки (L_inf, L2, AutoAttackLite)"
        )
        plt.grid(True)
        plt.legend()
        plt.show()

        rob_group = (
            df_baseline
            .assign(is_correct=lambda d: d["adv_pred_id"] == d["orig_label_id"])
            .groupby(["attack", "epsilon"])["is_correct"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(9, 5))
        for attack_name in attack_names:
            sub = rob_group[rob_group["attack"] == attack_name]
            if len(sub) == 0:
                continue
            plt.plot(sub["epsilon"], sub["is_correct"], label=attack_name)

        plt.xlabel("epsilon (сила атаки)")
        plt.ylabel("Robust accuracy (доля без смены класса)")
        plt.title(f"{model_name}: robust accuracy (defense = none)")
        plt.grid(True)
        plt.legend()
        plt.show()

        defense_focus_attack = "PGD"
        defense_prob = (
            df_model[
                (df_model["mode"] == "untargeted")
                & (df_model["attack"] == defense_focus_attack)
            ]
            .groupby(["defense", "epsilon"])["prob_orig_class"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(9, 5))
        for defense_name in defense_prob["defense"].unique():
            sub = defense_prob[defense_prob["defense"] == defense_name]
            plt.plot(sub["epsilon"], sub["prob_orig_class"], label=defense_name)
        plt.xlabel("epsilon (PGD, L_inf)")
        plt.ylabel("Средняя уверенность (PGD)")
        plt.title(f"{model_name}: влияние защит на PGD-атаку")
        plt.grid(True)
        plt.legend()
        plt.show()

        auto_defense = (
            df_model[(df_model["mode"] == "untargeted") & (df_model["attack"] == "AutoAttackLite")]
            .assign(is_correct=lambda d: d["adv_pred_id"] == d["orig_label_id"])
            .groupby(["defense", "epsilon"])["is_correct"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(9, 5))
        for defense_name in auto_defense["defense"].unique():
            sub = auto_defense[auto_defense["defense"] == defense_name]
            plt.plot(sub["epsilon"], sub["is_correct"], label=defense_name)
        plt.xlabel("epsilon (AutoAttackLite)")
        plt.ylabel("Robust accuracy")
        plt.title(f"{model_name}: защита против AutoAttackLite")
        plt.grid(True)
        plt.legend()
        plt.show()

        eps_star_records = []
        for attack_name in attack_names:
            df_attack = df_baseline[df_baseline["attack"] == attack_name]
            for i in range(num_images):
                df_ai = df_attack[df_attack["image_id"] == i].sort_values("epsilon")
                changed = df_ai[df_ai["adv_pred_id"] != df_ai["orig_label_id"]]
                if len(changed) == 0:
                    eps_star = float(eps_range[-1])
                else:
                    eps_star = float(changed["epsilon"].iloc[0])
                eps_star_records.append(
                    {
                        "model": model_name,
                        "attack": attack_name,
                        "image_id": i,
                        "eps_star": eps_star,
                    }
                )

        eps_star_df = pd.DataFrame(eps_star_records)

        plt.figure(figsize=(9, 5))
        for attack_name in attack_names:
            sub = eps_star_df[eps_star_df["attack"] == attack_name]
            if len(sub) == 0:
                continue
            plt.hist(sub["eps_star"], bins=15, alpha=0.4, label=attack_name)

        plt.xlabel("eps* (минимальный epsilon смены класса)")
        plt.ylabel("Количество изображений")
        plt.title(f"{model_name}: распределение eps* (defense = none)")
        plt.legend()
        plt.show()

        df_target = df_baseline[df_baseline["image_id"] == 0]

        def first_change_eps(df_attack: pd.DataFrame) -> float:
            df_sorted = df_attack.sort_values("epsilon")
            changed = df_sorted[df_sorted["adv_pred_id"] != df_sorted["orig_label_id"]]
            if len(changed) == 0:
                return float(eps_range[-1])
            return float(changed["epsilon"].iloc[0])

        eps_fgsm_final = first_change_eps(df_target[df_target["attack"] == "FGSM"])
        eps_bim_final = first_change_eps(df_target[df_target["attack"] == "BIM"])
        eps_pgd_final = first_change_eps(df_target[df_target["attack"] == "PGD"])
        eps_mi_final = first_change_eps(df_target[df_target["attack"] == "MI-FGSM"])

        x_adv_fgsm_final = fgsm_attack(model, x_target, y_target_true, eps_fgsm_final)
        x_adv_bim_final = bim_attack(model, x_target, y_target_true, eps_bim_final, steps=10)
        x_adv_pgd_final = pgd_attack(model, x_target, y_target_true, eps_pgd_final, steps=10, alpha_ratio=0.25)
        x_adv_mi_final = mi_fgsm_attack(model, x_target, y_target_true, eps_mi_final, steps=10, mu=1.0)

        with torch.no_grad():
            out_fgsm_final = model(x_adv_fgsm_final)
            probs_fgsm_final = F.softmax(out_fgsm_final, dim=1)[0]
            fgsm_label_id = int(probs_fgsm_final.argmax().item())
            fgsm_label = classes_txt[fgsm_label_id]
            fgsm_prob = float(probs_fgsm_final[fgsm_label_id].item())

            out_bim_final = model(x_adv_bim_final)
            probs_bim_final = F.softmax(out_bim_final, dim=1)[0]
            bim_label_id = int(probs_bim_final.argmax().item())
            bim_label = classes_txt[bim_label_id]
            bim_prob = float(probs_bim_final[bim_label_id].item())

            out_pgd_final = model(x_adv_pgd_final)
            probs_pgd_final = F.softmax(out_pgd_final, dim=1)[0]
            pgd_label_id = int(probs_pgd_final.argmax().item())
            pgd_label = classes_txt[pgd_label_id]
            pgd_prob = float(probs_pgd_final[pgd_label_id].item())

            out_mi_final = model(x_adv_mi_final)
            probs_mi_final = F.softmax(out_mi_final, dim=1)[0]
            mi_label_id = int(probs_mi_final.argmax().item())
            mi_label = classes_txt[mi_label_id]
            mi_prob = float(probs_mi_final[mi_label_id].item())

        print(f"\n[{model_name}] Target image final predictions (untargeted):")
        print(f"FGSM final:    {fgsm_label} (p = {fgsm_prob:.4f}) at eps = {eps_fgsm_final:.4f}")
        print(f"BIM  final:    {bim_label} (p = {bim_prob:.4f}) at eps = {eps_bim_final:.4f}")
        print(f"PGD  final:    {pgd_label} (p = {pgd_prob:.4f}) at eps = {eps_pgd_final:.4f}")
        print(f"MI-FGSM final: {mi_label} (p = {mi_prob:.4f}) at eps = {eps_mi_final:.4f}")

        heatmap_fgsm = compute_heatmap_for_input(model, x_adv_fgsm_final)
        heatmap_bim = compute_heatmap_for_input(model, x_adv_bim_final)
        heatmap_pgd = compute_heatmap_for_input(model, x_adv_pgd_final)
        heatmap_mi = compute_heatmap_for_input(model, x_adv_mi_final)

        rows, cols = 4, 3
        plt.figure(figsize=(14, 12))

        plt.subplot(rows, cols, 1)
        plt.imshow(pil_images[0])
        plt.title(f"{model_name} / FGSM: Original\n{target_label}")
        plt.axis("off")

        plt.subplot(rows, cols, 2)
        plt.imshow(x_adv_fgsm_final.squeeze(0).detach().cpu().permute(1, 2, 0))
        plt.title(f"{model_name} / FGSM: Adversarial\n{fgsm_label}\neps={eps_fgsm_final:.4f}")
        plt.axis("off")

        plt.subplot(rows, cols, 3)
        plt.imshow(heatmap_fgsm.cpu(), cmap="inferno")
        plt.title("FGSM: Gradient Heatmap")
        plt.axis("off")

        plt.subplot(rows, cols, 4)
        plt.imshow(pil_images[0])
        plt.title(f"{model_name} / BIM: Original\n{target_label}")
        plt.axis("off")

        plt.subplot(rows, cols, 5)
        plt.imshow(x_adv_bim_final.squeeze(0).detach().cpu().permute(1, 2, 0))
        plt.title(f"{model_name} / BIM: Adversarial\n{bim_label}\neps={eps_bim_final:.4f}")
        plt.axis("off")

        plt.subplot(rows, cols, 6)
        plt.imshow(heatmap_bim.cpu(), cmap="inferno")
        plt.title("BIM: Gradient Heatmap")
        plt.axis("off")

        plt.subplot(rows, cols, 7)
        plt.imshow(pil_images[0])
        plt.title(f"{model_name} / PGD: Original\n{target_label}")
        plt.axis("off")

        plt.subplot(rows, cols, 8)
        plt.imshow(x_adv_pgd_final.squeeze(0).detach().cpu().permute(1, 2, 0))
        plt.title(f"{model_name} / PGD: Adversarial\n{pgd_label}\neps={eps_pgd_final:.4f}")
        plt.axis("off")

        plt.subplot(rows, cols, 9)
        plt.imshow(heatmap_pgd.cpu(), cmap="inferno")
        plt.title("PGD: Gradient Heatmap")
        plt.axis("off")

        plt.subplot(rows, cols, 10)
        plt.imshow(pil_images[0])
        plt.title(f"{model_name} / MI-FGSM: Original\n{target_label}")
        plt.axis("off")

        plt.subplot(rows, cols, 11)
        plt.imshow(x_adv_mi_final.squeeze(0).detach().cpu().permute(1, 2, 0))
        plt.title(f"{model_name} / MI-FGSM: Adversarial\n{mi_label}\neps={eps_mi_final:.4f}")
        plt.axis("off")

        plt.subplot(rows, cols, 12)
        plt.imshow(heatmap_mi.cpu(), cmap="inferno")
        plt.title("MI-FGSM: Gradient Heatmap")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    output_path = "adversarial_results_all_models.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        for model_name, df_model in dfs_for_excel.items():
            sheet_name = model_name[:31]
            df_model.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nВсе результаты сохранены в файле: {output_path}")


if __name__ == "__main__":
    run_all()
