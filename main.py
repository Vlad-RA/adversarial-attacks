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


# -------------------------------------------------------
# 0. Детерминированность (по возможности)
# -------------------------------------------------------

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# -------------------------------------------------------
# 1. Device
# -------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# -------------------------------------------------------
# 2. Загрузка набора изображений (общая для всех моделей)
#    Первое изображение считаем target-изображением для наглядных визуализаций
# -------------------------------------------------------

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

image_urls = [
    # target-изображение (первое в списке)
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

pil_images = []
X_list = []

for idx, url in enumerate(image_urls):
    print(f"Loading image {idx}: {url}")
    img = load_image_from_url(url)
    pil_images.append(img)

    x_i = transform(img).unsqueeze(0).to(device)
    X_list.append(x_i)

X = torch.cat(X_list, dim=0)  # [N,3,224,224]
num_images = X.size(0)
print(f"Всего изображений: {num_images}")


# -------------------------------------------------------
# 3. Вспомогательные функции: градиент, heatmap и меры искажений
# -------------------------------------------------------

def grad_to_heatmap(grad_tensor: torch.Tensor) -> torch.Tensor:
    """Преобразуем градиент [1,3,H,W] в нормированный heatmap [H,W]."""
    grad_mag = grad_tensor.abs().sum(dim=1).squeeze(0)  # [H,W]
    grad_norm = grad_mag / (grad_mag.max() + 1e-8)
    return grad_norm


def compute_heatmap_for_input(model, x_in, y_target=None):
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
    """L_inf, L2, L1 и PSNR для оценки силы и видимости искажений.

    x_orig, x_adv: [1,3,H,W] или [N,3,H,W] в диапазоне [0,1].
    """
    with torch.no_grad():
        delta = x_adv - x_orig
        # L_inf по батчу
        linf = delta.abs().view(delta.size(0), -1).max(dim=1).values
        # L2
        l2 = delta.view(delta.size(0), -1).norm(p=2, dim=1)
        # L1
        l1 = delta.view(delta.size(0), -1).norm(p=1, dim=1)
        # PSNR
        mse = (delta ** 2).view(delta.size(0), -1).mean(dim=1)
        psnr = -10.0 * torch.log10(mse + 1e-12)
    return linf.cpu().numpy(), l2.cpu().numpy(), l1.cpu().numpy(), psnr.cpu().numpy()


# -------------------------------------------------------
# 4. Реализация атак (4 популярных L_inf метода)
# -------------------------------------------------------

def fgsm_attack(model, x_in, y, eps: float, targeted: bool = False, y_target: torch.Tensor | None = None):
    """FGSM (L_inf). Если targeted=True, используем целевой класс y_target."""
    if eps == 0.0:
        return x_in.clone().detach()
    x_adv = x_in.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    out = model(x_adv)
    if targeted:
        assert y_target is not None
        loss = -F.cross_entropy(out, y_target)  # минус, чтобы увеличить P(y_target)
    else:
        loss = F.cross_entropy(out, y)
    loss.backward()
    grad = x_adv.grad.sign()
    with torch.no_grad():
        x_adv = torch.clamp(x_adv + eps * grad, 0, 1)
    return x_adv.detach()


def bim_attack(model, x_in, y, eps: float, steps: int = 10, targeted: bool = False, y_target: torch.Tensor | None = None):
    """BIM / I-FGSM (iterative FGSM, L_inf). Поддерживает targeted режим."""
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
            x_adv = x_adv + alpha * grad
            eta = torch.clamp(x_adv - x_in, min=-eps, max=eps)
            x_adv = torch.clamp(x_in + eta, 0, 1)
    return x_adv.detach()


def pgd_attack(model, x_in, y, eps: float, steps: int = 10, alpha_ratio: float = 0.25, targeted: bool = False, y_target: torch.Tensor | None = None, random_start: bool = True):
    """PGD (Madry) с L_inf-ограничением. Поддерживает targeted режим."""
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
            x_adv = x_adv + alpha * grad
            eta = torch.clamp(x_adv - x_in, min=-eps, max=eps)
            x_adv = torch.clamp(x_in + eta, 0, 1)
    return x_adv.detach()


def mi_fgsm_attack(model, x_in, y, eps: float, steps: int = 10, mu: float = 1.0, targeted: bool = False, y_target: torch.Tensor | None = None):
    """MI-FGSM (Momentum Iterative FGSM, L_inf). Поддерживает targeted режим."""
    if eps == 0.0:
        return x_in.clone().detach()

    x_adv = x_in.clone().detach()
    g = torch.zeros_like(x_adv)
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
        grad = x_adv.grad

        grad_mean = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-12
        grad_normed = grad / grad_mean
        g = mu * g + grad_normed

        with torch.no_grad():
            x_adv = x_adv + alpha * g.sign()
            eta = torch.clamp(x_adv - x_in, min=-eps, max=eps)
            x_adv = torch.clamp(x_in + eta, 0, 1)

    return x_adv.detach()


# -------------------------------------------------------
# 5. Список моделей для экспериментов (3 популярные архитектуры)
# -------------------------------------------------------

models_to_run = [
    ("ResNet50",   models.resnet50,   models.ResNet50_Weights.IMAGENET1K_V1),
    ("VGG16",      models.vgg16,      models.VGG16_Weights.IMAGENET1K_V1),
    ("DenseNet121", models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
]

attack_names = ["FGSM", "BIM", "PGD", "MI-FGSM"]

# сюда будем складывать таблицы для последующей записи в один файл

dfs_for_excel = {}


# -------------------------------------------------------
# 6. Основной цикл по моделям
# -------------------------------------------------------

eps_range = np.linspace(0.0, 0.02, 80)  # мелкий шаг по epsilon

for model_name, ctor, weights_enum in models_to_run:
    print("\n" + "=" * 90)
    print(f"МОДЕЛЬ: {model_name}")
    print("=" * 90)

    weights = weights_enum
    classes_txt = weights.meta["categories"]

    model = ctor(weights=weights).to(device).eval()

    # -------- 6.1. Предсказания модели на исходных изображениях --------
    labels = []
    orig_probs = []

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

    # target-изображение — первое в наборе
    x_target = X[0:1]
    y_target_true = Y[0:1]
    target_label_id = labels[0]
    target_label = classes_txt[target_label_id]
    target_prob = orig_probs[0]
    print(f"\nTarget image original for {model_name}: {target_label} (p = {target_prob:.4f})")

    # выберем фиксированный целевой класс для таргетированных атак:
    # возьмём Top-2 класс для target-изображения
    with torch.no_grad():
        out_target = model(x_target)
        probs_target = F.softmax(out_target, dim=1)[0]
        top2 = torch.topk(probs_target, k=2)
        target_ids = top2.indices.cpu().numpy().tolist()
    # если top-0 == исходный класс, возьмём top-1, иначе top-0
    if target_ids[0] == target_label_id and len(target_ids) > 1:
        targeted_class_id = target_ids[1]
    else:
        targeted_class_id = target_ids[0]
    y_target_cls = torch.tensor([targeted_class_id], device=device)
    print(f"Targeted class for {model_name}: {classes_txt[targeted_class_id]}")

    # -------- 6.2. Сканирование по eps и сбор таблицы результатов --------
    results = []

    for eps in eps_range:
        eps_f = float(eps)

        for i in range(num_images):
            xi = X[i:i+1]
            yi = Y[i:i+1]
            orig_id = labels[i]
            orig_name = classes_txt[orig_id]

            # Untargeted FGSM
            x_adv_fgsm = fgsm_attack(model, xi, yi, eps_f, targeted=False)
            with torch.no_grad():
                out_fgsm = model(x_adv_fgsm)
                probs_fgsm = F.softmax(out_fgsm, dim=1)[0]
            pred_fgsm = int(probs_fgsm.argmax().item())
            linf, l2, l1, psnr = distortion_metrics(xi, x_adv_fgsm)

            results.append({
                "model": model_name,
                "attack": "FGSM",
                "mode": "untargeted",
                "image_id": i,
                "epsilon": eps_f,
                "orig_label_id": orig_id,
                "orig_label_name": orig_name,
                "adv_pred_id": pred_fgsm,
                "adv_pred_name": classes_txt[pred_fgsm],
                "prob_orig_class": float(probs_fgsm[orig_id].item()),
                "prob_top1": float(probs_fgsm.max().item()),
                "linf": float(linf[0]),
                "l2": float(l2[0]),
                "l1": float(l1[0]),
                "psnr": float(psnr[0]),
            })

            # Targeted FGSM (к общему df тоже добавляем, но mode="targeted")
            x_adv_fgsm_t = fgsm_attack(model, xi, yi, eps_f, targeted=True, y_target=y_target_cls)
            with torch.no_grad():
                out_fgsm_t = model(x_adv_fgsm_t)
                probs_fgsm_t = F.softmax(out_fgsm_t, dim=1)[0]
            pred_fgsm_t = int(probs_fgsm_t.argmax().item())
            linf_t, l2_t, l1_t, psnr_t = distortion_metrics(xi, x_adv_fgsm_t)

            results.append({
                "model": model_name,
                "attack": "FGSM",
                "mode": "targeted",
                "image_id": i,
                "epsilon": eps_f,
                "orig_label_id": orig_id,
                "orig_label_name": orig_name,
                "adv_pred_id": pred_fgsm_t,
                "adv_pred_name": classes_txt[pred_fgsm_t],
                "prob_orig_class": float(probs_fgsm_t[orig_id].item()),
                "prob_top1": float(probs_fgsm_t.max().item()),
                "linf": float(linf_t[0]),
                "l2": float(l2_t[0]),
                "l1": float(l1_t[0]),
                "psnr": float(psnr_t[0]),
            })

            # BIM / I-FGSM (untargeted)
            x_adv_bim = bim_attack(model, xi, yi, eps_f, steps=10, targeted=False)
            with torch.no_grad():
                out_bim = model(x_adv_bim)
                probs_bim = F.softmax(out_bim, dim=1)[0]
            pred_bim = int(probs_bim.argmax().item())
            linf, l2, l1, psnr = distortion_metrics(xi, x_adv_bim)

            results.append({
                "model": model_name,
                "attack": "BIM",
                "mode": "untargeted",
                "image_id": i,
                "epsilon": eps_f,
                "orig_label_id": orig_id,
                "orig_label_name": orig_name,
                "adv_pred_id": pred_bim,
                "adv_pred_name": classes_txt[pred_bim],
                "prob_orig_class": float(probs_bim[orig_id].item()),
                "prob_top1": float(probs_bim.max().item()),
                "linf": float(linf[0]),
                "l2": float(l2[0]),
                "l1": float(l1[0]),
                "psnr": float(psnr[0]),
            })

            # PGD (untargeted)
            x_adv_pgd = pgd_attack(model, xi, yi, eps_f, steps=10, alpha_ratio=0.25, targeted=False)
            with torch.no_grad():
                out_pgd = model(x_adv_pgd)
                probs_pgd = F.softmax(out_pgd, dim=1)[0]
            pred_pgd = int(probs_pgd.argmax().item())
            linf, l2, l1, psnr = distortion_metrics(xi, x_adv_pgd)

            results.append({
                "model": model_name,
                "attack": "PGD",
                "mode": "untargeted",
                "image_id": i,
                "epsilon": eps_f,
                "orig_label_id": orig_id,
                "orig_label_name": orig_name,
                "adv_pred_id": pred_pgd,
                "adv_pred_name": classes_txt[pred_pgd],
                "prob_orig_class": float(probs_pgd[orig_id].item()),
                "prob_top1": float(probs_pgd.max().item()),
                "linf": float(linf[0]),
                "l2": float(l2[0]),
                "l1": float(l1[0]),
                "psnr": float(psnr[0]),
            })

            # MI-FGSM (untargeted)
            x_adv_mi = mi_fgsm_attack(model, xi, yi, eps_f, steps=10, mu=1.0, targeted=False)
            with torch.no_grad():
                out_mi = model(x_adv_mi)
                probs_mi = F.softmax(out_mi, dim=1)[0]
            pred_mi = int(probs_mi.argmax().item())
            linf, l2, l1, psnr = distortion_metrics(xi, x_adv_mi)

            results.append({
                "model": model_name,
                "attack": "MI-FGSM",
                "mode": "untargeted",
                "image_id": i,
                "epsilon": eps_f,
                "orig_label_id": orig_id,
                "orig_label_name": orig_name,
                "adv_pred_id": pred_mi,
                "adv_pred_name": classes_txt[pred_mi],
                "prob_orig_class": float(probs_mi[orig_id].item()),
                "prob_top1": float(probs_mi.max().item()),
                "linf": float(linf[0]),
                "l2": float(l2[0]),
                "l1": float(l1[0]),
                "psnr": float(psnr[0]),
            })

    df_model = pd.DataFrame(results)
    dfs_for_excel[model_name] = df_model

    # -------- 6.3. График средней уверенности в исходном классе (untargeted) --------
    group_prob = (
        df_model[df_model["mode"] == "untargeted"]
        .groupby(["attack", "epsilon"])["prob_orig_class"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    for attack_name in attack_names:
        sub = group_prob[group_prob["attack"] == attack_name]
        plt.plot(sub["epsilon"], sub["prob_orig_class"], label=attack_name)

    plt.xlabel("epsilon (сила атаки)")
    plt.ylabel("Средняя уверенность в исходном классе")
    plt.title(
        f"{model_name}: средняя уверенность модели (untargeted)\n"
        f"от силы атаки (4 метода, {num_images} изображений)"
    )
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------- 6.4. Robust accuracy: доля изображений без смены класса (untargeted) --------
    rob_group = (
        df_model[df_model["mode"] == "untargeted"]
        .assign(is_correct=lambda d: d["adv_pred_id"] == d["orig_label_id"])
        .groupby(["attack", "epsilon"])["is_correct"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    for attack_name in attack_names:
        sub = rob_group[rob_group["attack"] == attack_name]
        plt.plot(sub["epsilon"], sub["is_correct"], label=attack_name)

    plt.xlabel("epsilon (сила атаки)")
    plt.ylabel("Robust accuracy (доля без смены класса)")
    plt.title(
        f"{model_name}: robust accuracy (untargeted) в зависимости от epsilon\n"
        f"(4 метода, {num_images} изображений)"
    )
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------- 6.5. Распределение eps* по датасету (минимальный eps смены класса, untargeted) --------
    eps_star_records = []

    for attack_name in attack_names:
        df_attack = df_model[(df_model["attack"] == attack_name) & (df_model["mode"] == "untargeted")]
        for i in range(num_images):
            df_ai = df_attack[df_attack["image_id"] == i].sort_values("epsilon")
            changed = df_ai[df_ai["adv_pred_id"] != df_ai["orig_label_id"]]
            if len(changed) == 0:
                eps_star = float(eps_range[-1])
            else:
                eps_star = float(changed["epsilon"].iloc[0])
            eps_star_records.append({
                "model": model_name,
                "attack": attack_name,
                "image_id": i,
                "eps_star": eps_star,
            })

    eps_star_df = pd.DataFrame(eps_star_records)

    plt.figure(figsize=(8, 5))
    for attack_name in attack_names:
        sub = eps_star_df[eps_star_df["attack"] == attack_name]
        plt.hist(sub["eps_star"], bins=15, alpha=0.5, label=attack_name)

    plt.xlabel("eps* (минимальный epsilon смены класса)")
    plt.ylabel("Количество изображений")
    plt.title(f"{model_name}: распределение eps* по датасету (untargeted, 4 атаки)")
    plt.legend()
    plt.show()

    # -------- 6.6. eps* для target-изображения по каждой атаке (untargeted) --------
    df_target = df_model[(df_model["image_id"] == 0) & (df_model["mode"] == "untargeted")]

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

    print(f"\n[{model_name}] target-class change eps (untargeted):")
    print(f"  FGSM:    {eps_fgsm_final}")
    print(f"  BIM:     {eps_bim_final}")
    print(f"  PGD:     {eps_pgd_final}")
    print(f"  MI-FGSM: {eps_mi_final}")

    # -------- 6.7. Финальные adversarial-примеры для target-изображения (untargeted) --------
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

    # -------- 6.8. Heatmaps для target-изображения по каждой атаке --------
    heatmap_fgsm = compute_heatmap_for_input(model, x_adv_fgsm_final)
    heatmap_bim = compute_heatmap_for_input(model, x_adv_bim_final)
    heatmap_pgd = compute_heatmap_for_input(model, x_adv_pgd_final)
    heatmap_mi = compute_heatmap_for_input(model, x_adv_mi_final)

    # -------- 6.9. Визуализация: 4 строки (атаки) × 3 столбца --------
    rows = 4
    cols = 3
    plt.figure(figsize=(14, 12))

    # FGSM
    plt.subplot(rows, cols, 1)
    plt.imshow(pil_images[0])
    plt.title(f"{model_name} / FGSM: Original\\n{target_label}")
    plt.axis("off")

    plt.subplot(rows, cols, 2)
    plt.imshow(x_adv_fgsm_final.squeeze(0).detach().cpu().permute(1, 2, 0))
    plt.title(f"{model_name} / FGSM: Adversarial\\n{fgsm_label}\\neps={eps_fgsm_final:.4f}")
    plt.axis("off")

    plt.subplot(rows, cols, 3)
    plt.imshow(heatmap_fgsm.cpu(), cmap="inferno")
    plt.title("FGSM: Gradient Heatmap")
    plt.axis("off")

    # BIM
    plt.subplot(rows, cols, 4)
    plt.imshow(pil_images[0])
    plt.title(f"{model_name} / BIM: Original\\n{target_label}")
    plt.axis("off")

    plt.subplot(rows, cols, 5)
    plt.imshow(x_adv_bim_final.squeeze(0).detach().cpu().permute(1, 2, 0))
    plt.title(f"{model_name} / BIM: Adversarial\\n{bim_label}\\neps={eps_bim_final:.4f}")
    plt.axis("off")

    plt.subplot(rows, cols, 6)
    plt.imshow(heatmap_bim.cpu(), cmap="inferno")
    plt.title("BIM: Gradient Heatmap")
    plt.axis("off")

    # PGD
    plt.subplot(rows, cols, 7)
    plt.imshow(pil_images[0])
    plt.title(f"{model_name} / PGD: Original\\n{target_label}")
    plt.axis("off")

    plt.subplot(rows, cols, 8)
    plt.imshow(x_adv_pgd_final.squeeze(0).detach().cpu().permute(1, 2, 0))
    plt.title(f"{model_name} / PGD: Adversarial\\n{pgd_label}\\neps={eps_pgd_final:.4f}")
    plt.axis("off")

    plt.subplot(rows, cols, 9)
    plt.imshow(heatmap_pgd.cpu(), cmap="inferno")
    plt.title("PGD: Gradient Heatmap")
    plt.axis("off")

    # MI-FGSM
    plt.subplot(rows, cols, 10)
    plt.imshow(pil_images[0])
    plt.title(f"{model_name} / MI-FGSM: Original\\n{target_label}")
    plt.axis("off")

    plt.subplot(rows, cols, 11)
    plt.imshow(x_adv_mi_final.squeeze(0).detach().cpu().permute(1, 2, 0))
    plt.title(f"{model_name} / MI-FGSM: Adversarial\\n{mi_label}\\neps={eps_mi_final:.4f}")
    plt.axis("off")

    plt.subplot(rows, cols, 12)
    plt.imshow(heatmap_mi.cpu(), cmap="inferno")
    plt.title("MI-FGSM: Gradient Heatmap")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# 7. Сохранение всех таблиц в один Excel-файл (3 листа — по модели)
# -------------------------------------------------------

output_path = "adversarial_results_all_models.xlsx"
with pd.ExcelWriter(output_path) as writer:
    for model_name, df_model in dfs_for_excel.items():
        sheet_name = model_name[:31]  # ограничение Excel на длину имени листа
        df_model.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\nВсе результаты сохранены в файле: {output_path}")
