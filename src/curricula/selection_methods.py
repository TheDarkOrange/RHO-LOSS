import torch
import numpy as np

import numbers
import warnings

from torch.nn import functional as F
import torch.nn as nn
from src.curricula.utils_bald import get_bald, enable_dropout, compute_entropy, compute_conditional_entropy
from src.curricula.utils_selection_methods import top_x_indices, top_x_distant_indices, create_logging_dict

import wandb

import time

from sklearn.cluster import KMeans


def _compute_irreducible_loss(
    data=None,
    target=None,
    global_index=None,
    irreducible_loss_generator=None,
    target_device=None,
):
    if type(irreducible_loss_generator) is torch.Tensor:
        # send the whole tensor over
        if (
            target_device is not None
            and irreducible_loss_generator.device != target_device
        ):
            irreducible_loss_generator = irreducible_loss_generator.to(
                device=target_device
            )

        irreducible_loss = irreducible_loss_generator[global_index]
    else:
        irreducible_loss = F.cross_entropy(
            irreducible_loss_generator(data), target, reduction="none"
        )

    return irreducible_loss


class reducible_loss_dropout_selection:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        with torch.no_grad():
            # 1) compute model loss
            model_net = proxy_model if proxy_model is not None else large_model
            logits = model_net(data)
            model_loss = F.cross_entropy(logits, target, reduction="none")

            # 2) estimate irreducible loss
            if 'irreducible_uncertainties' in irreducible_loss_generator:
                # use MC Dropout
                idx = global_index.to(data.device)
                il_mean = irreducible_loss_generator["irreducible_losses"].to(data.device)[idx]
                il_var = irreducible_loss_generator["irreducible_uncertainties"].to(data.device)[idx]
                irreducible_loss = il_mean# + 1.0 * il_var.sqrt()
            else:
                print('irreducible_uncertainties' in irreducible_loss_generator.keys())
                # static tensor or dict of precomputed losses
                if isinstance(irreducible_loss_generator, dict):
                    ir_tensor = irreducible_loss_generator["irreducible_losses"]
                else:
                    ir_tensor = irreducible_loss_generator

                # ensure tensor on correct device
                ir_tensor = ir_tensor.to(data.device)

                # convert indices to long and clamp to valid range
                idx = global_index.long().to(data.device)
                max_idx = ir_tensor.size(0) - 1
                if idx.min() < 0 or idx.max() > max_idx:
                    warnings.warn(
                        f"global_index contains out-of-bounds values "
                        f"[{int(idx.min())}, {int(idx.max())}] for irreducible-loss tensor of length {ir_tensor.size(0)}; clamping to [0, {max_idx}]."
                    )
                    idx = idx.clamp(0, max_idx)

                irreducible_loss = ir_tensor[idx]

            # 3) compute reducible loss and select top samples
            reducible_loss = model_loss - irreducible_loss
            selected_minibatch, not_selected_minibatch = top_x_indices(
                reducible_loss,
                selected_batch_size,
                largest=True
            )
            selected_irreducible_loss = irreducible_loss[selected_minibatch]

        # 4) compute proportion of batch reducible loss in top 20%
        batch_size = data.size(0)
        sorted_rloss, _ = torch.sort(reducible_loss, descending=True)
        sorted_rloss[sorted_rloss < 0] = 0.0
        percentage_top_20 = (
            sorted_rloss[: int(0.2 * batch_size)].sum() / sorted_rloss.sum()
        ).item()

        # 5) prepare logging metrics
        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "model_loss": model_loss,
        }
        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )
        metrics_to_log["proportion_top_20_reducible_loss"] = percentage_top_20
        metrics_to_log["detailed_only_keys"] = ["selected_id", "not_selected_id"]

        # 6) log elapsed time
        wandb.log(
            {"elapsed_time": time.time() - wandb.config.start_time},
            commit=False
        )

        return selected_minibatch, metrics_to_log, selected_irreducible_loss


class reducible_loss_plus_conditional_entropy_selection:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data,
        target,
        global_index,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        num_mc=10,
        num_classes=10,
        *args,
        **kwargs,
    ):
        predictions = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        log_probs = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        with torch.no_grad():
            if proxy_model is not None:
                predictions[0] = proxy_model(data)
            else:
                predictions[0] = large_model(data)
            model_loss = F.cross_entropy(
                    predictions[0], target, reduction="none"
                )
            log_probs[0] = F.log_softmax(predictions[0], dim=-1)

            for i in range(num_mc):
                predictions[i] = predictions[0]
                log_probs[i] = log_probs[0]
            
            conditional_entropy = compute_conditional_entropy(
                log_probs.transpose(0, 1))

            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator,
                model_loss.device,
            )

            reducible_loss = model_loss - irreducible_loss + conditional_entropy

            selected_minibatch, not_selected_minibatch = top_x_indices(
                reducible_loss, selected_batch_size, largest=True
            )
            selected_irreducible_loss = irreducible_loss[selected_minibatch]

        # compute what proportion of the reducible loss is made up of the points with the highest 20% reducible losses
        batch_size = len(
            data
        )  # get big batch size, to compute what proportion of the batch model loss is made up by the points in the batch with the highest loss
        reducible_loss_sorted, _ = torch.sort(reducible_loss, descending=True)
        reducible_loss_sorted[
            reducible_loss_sorted < 0
        ] = 0.0  # ignore negative reducible losses
        percentage_top_20 = torch.sum(
            reducible_loss_sorted[: int(0.2 * batch_size)]
        ) / torch.sum(reducible_loss_sorted)

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "model_loss": model_loss,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log[
            "proportion_of_total_batch_reducible_loss_corresponding_to_the_top_20%_points_with_highest_reducible_loss"
        ] = float(percentage_top_20)

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        wandb.log({'elapsed_time': time.time() - wandb.config.start_time}, commit=False)

        return selected_minibatch, metrics_to_log, selected_irreducible_loss


class reducible_loss_selection_rand:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        """
        Selects samples using reducible loss, optionally based on proxy model
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module
             Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
             nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
            compare_to_large_model_selection: if True, when proxy is specified, also calculate points using the large model,
                                                and include statistics of the selected points
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """
        with torch.no_grad():
            if proxy_model is not None:
                model_loss = F.cross_entropy(
                    proxy_model(data), target, reduction="none"
                )
            else:
                model_loss = F.cross_entropy(
                    large_model(data), target, reduction="none"
                )

            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator,
                model_loss.device,
            )

            reducible_loss = torch.exp(3*(model_loss - irreducible_loss))

            selected_minibatch = torch.multinomial(reducible_loss, num_samples=selected_batch_size, replacement=False)
            not_selected_minibatch = torch.from_numpy(np.setdiff1d(torch.arange(data.size(0)).cpu().numpy(),selected_minibatch.cpu().numpy())).to(data.device)

            selected_irreducible_loss = irreducible_loss[selected_minibatch]

        # compute what proportion of the reducible loss is made up of the points with the highest 20% reducible losses
        batch_size = len(
            data
        )  # get big batch size, to compute what proportion of the batch model loss is made up by the points in the batch with the highest loss
        reducible_loss_sorted, _ = torch.sort(reducible_loss, descending=True)
        reducible_loss_sorted[
            reducible_loss_sorted < 0
        ] = 0.0  # ignore negative reducible losses
        percentage_top_20 = torch.sum(
            reducible_loss_sorted[: int(0.2 * batch_size)]
        ) / torch.sum(reducible_loss_sorted)

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "model_loss": model_loss,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log[
            "proportion_of_total_batch_reducible_loss_corresponding_to_the_top_20%_points_with_highest_reducible_loss"
        ] = float(percentage_top_20)

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        wandb.log({'elapsed_time': time.time() - wandb.config.start_time}, commit=False)

        return selected_minibatch, metrics_to_log, selected_irreducible_loss


class reducible_loss_selection7:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        """
        Selects samples using reducible loss combined with a diversity component.
        In addition to computing the standard reducible loss (model_loss - irreducible_loss),
        we compute a gradient embedding for each sample (using a simple difference between one-hot targets
        and softmax outputs) and use k-means clustering to select a diverse subset among the top candidates.
        
        Args:
            selected_batch_size: int, number of samples to be selected.
            data: tensor, input data for the current batch.
            target: tensor, ground truth labels.
            global_index: tensor, global indices of the datapoints.
            large_model: nn.Module, the main model.
            irreducible_loss_generator: Tensor or nn.Module used to compute irreducible loss.
            proxy_model: nn.Module, an optional proxy model for computing loss and gradient embeddings.
            *args, **kwargs: additional parameters.
        Returns:
            selected_minibatch: tensor, local indices of the selected samples (w.r.t. the current batch).
            metrics_to_log: dictionary, various computed metrics.
            selected_irreducible_loss: tensor, irreducible loss values for the selected samples.
        """
        # Compute model loss (using proxy_model if provided, else large_model)
        with torch.no_grad():
            if proxy_model is not None:
                logits = proxy_model(data)
            else:
                logits = large_model(data)
            model_loss = F.cross_entropy(logits, target, reduction="none")
        
            # Compute irreducible loss using a helper function (assumed to be defined elsewhere)
            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator,
                model_loss.device,
            )
        
            # Compute reducible loss as the difference
            reducible_loss = model_loss - irreducible_loss

        # ---- Diversity component: Compute gradient embeddings ----
        # We'll compute a simple gradient embedding as: one_hot(target) - softmax(logits)
        # This requires gradients; we can perform a forward pass with grad enabled.
        with torch.enable_grad():
            data_grad = data.clone().detach().requires_grad_(True)
            if proxy_model is not None:
                logits_grad = proxy_model(data_grad)
            else:
                logits_grad = large_model(data_grad)
            probs = F.softmax(logits_grad, dim=1)
            one_hot = F.one_hot(target, num_classes=logits_grad.size(1)).float()
            # Gradient embedding proxy: difference between one-hot and probabilities.
            grad_embedding = one_hot - probs  # shape: (batch_size, num_classes)
            # Detach for clustering (and convert to numpy)
            grad_embedding_np = grad_embedding.detach().cpu().numpy()

        # ---- Candidate selection based on reducible loss ranking ----
        # Sort all indices by reducible_loss in descending order.
        sorted_indices = torch.argsort(reducible_loss, descending=True)
        # Define a candidate pool size (e.g., top 2x the desired batch or at least 10)
        candidate_pool_size = min(max(selected_batch_size * 2, 10), len(data))
        candidate_indices = sorted_indices[:candidate_pool_size].cpu().numpy()
        
        # Get candidate gradient embeddings from the pool
        candidate_embeddings = grad_embedding_np[candidate_indices]  # shape: (candidate_pool_size, embedding_dim)

        # ---- Diversity selection via clustering ----
        # Use k-means clustering on the candidate gradient embeddings.
        # Set number of clusters equal to the desired selected_batch_size.
        kmeans = KMeans(n_clusters=selected_batch_size, random_state=0)
        cluster_labels = kmeans.fit_predict(candidate_embeddings)
        cluster_centers = kmeans.cluster_centers_

        # For each cluster, select the candidate whose embedding is closest to the cluster center.
        selected_candidate_indices = []
        for cluster in range(selected_batch_size):
            # indices within candidate pool belonging to this cluster
            cluster_mask = (cluster_labels == cluster)
            if np.sum(cluster_mask) == 0:
                continue  # safety check
            cluster_candidate_indices = np.where(cluster_mask)[0]
            embeddings_cluster = candidate_embeddings[cluster_candidate_indices]
            center = cluster_centers[cluster]
            # Compute Euclidean distances to the cluster center
            distances = np.linalg.norm(embeddings_cluster - center, axis=1)
            best_idx_in_cluster = cluster_candidate_indices[np.argmin(distances)]
            # Map back to original index in the batch
            selected_candidate_indices.append(candidate_indices[best_idx_in_cluster])
        
        # In case we get fewer than desired due to edge cases, pad with top reducible loss indices.
        if len(selected_candidate_indices) < selected_batch_size:
            additional = sorted_indices.cpu().numpy()[:selected_batch_size - len(selected_candidate_indices)]
            selected_candidate_indices = list(selected_candidate_indices) + list(additional)
            selected_candidate_indices = np.array(selected_candidate_indices)[:selected_batch_size]
        else:
            selected_candidate_indices = np.array(selected_candidate_indices)

        # Convert to torch tensor (local indices with respect to current minibatch)
        selected_minibatch = torch.tensor(selected_candidate_indices, device=data.device)

        # Compute selected irreducible loss for logging
        selected_irreducible_loss = irreducible_loss[selected_minibatch]

        # ---- Logging additional metrics ----
        batch_size = len(data)
        reducible_loss_sorted, _ = torch.sort(reducible_loss, descending=True)
        reducible_loss_sorted[reducible_loss_sorted < 0] = 0.0  # ignore negatives
        percentage_top_20 = torch.sum(reducible_loss_sorted[: int(0.2 * batch_size)]) / torch.sum(reducible_loss_sorted)

        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "model_loss": model_loss,
        }
        metrics_to_log = create_logging_dict(variables_to_log, selected_minibatch, None)
        metrics_to_log["proportion_of_total_batch_reducible_loss_corresponding_to_the_top_20%_points_with_highest_reducible_loss"] = float(percentage_top_20)
        metrics_to_log["detailed_only_keys"] = ["selected_id", "not_selected_id"]

        wandb.log({'elapsed_time': time.time() - wandb.config.start_time}, commit=False)

        return selected_minibatch, metrics_to_log, selected_irreducible_loss


class reducible_loss_selection6:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        current_epoch=0,
        *args,
        **kwargs,
    ):
        """
        Selects samples using reducible loss, optionally based on proxy model
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module
             Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
             nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
            compare_to_large_model_selection: if True, when proxy is specified, also calculate points using the large model,
                                                and include statistics of the selected points
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """
        with torch.no_grad():
            if proxy_model is not None:
                model_loss = F.cross_entropy(
                    proxy_model(data), target, reduction="none"
                )
            else:
                model_loss = F.cross_entropy(
                    large_model(data), target, reduction="none"
                )

            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator,
                model_loss.device,
            )

            reducible_loss = model_loss - (2 / (1 + np.exp(-current_epoch*0.1))) * irreducible_loss

            selected_minibatch, not_selected_minibatch = top_x_indices(
                reducible_loss, selected_batch_size, largest=True
            )
            selected_irreducible_loss = irreducible_loss[selected_minibatch]

        # compute what proportion of the reducible loss is made up of the points with the highest 20% reducible losses
        batch_size = len(
            data
        )  # get big batch size, to compute what proportion of the batch model loss is made up by the points in the batch with the highest loss
        reducible_loss_sorted, _ = torch.sort(reducible_loss, descending=True)
        reducible_loss_sorted[
            reducible_loss_sorted < 0
        ] = 0.0  # ignore negative reducible losses
        percentage_top_20 = torch.sum(
            reducible_loss_sorted[: int(0.2 * batch_size)]
        ) / torch.sum(reducible_loss_sorted)

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "model_loss": model_loss,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log[
            "proportion_of_total_batch_reducible_loss_corresponding_to_the_top_20%_points_with_highest_reducible_loss"
        ] = float(percentage_top_20)

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        wandb.log({'elapsed_time': time.time() - wandb.config.start_time}, commit=False)

        return selected_minibatch, metrics_to_log, selected_irreducible_loss


class reducible_loss_selection5:
    bald = False

    def __init__(self, switch=50):
        self.switch = switch
        self.mixup = reducible_loss_selection_mixup()
        self.no_mixup = reducible_loss_selection()

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        current_epoch=None,
        *args,
        **kwargs,
    ):
        if current_epoch >= self.switch:
            result = self.mixup(
                selected_batch_size=selected_batch_size,
                data=data,
                target=target,
                global_index=global_index,
                large_model=large_model,
                irreducible_loss_generator=irreducible_loss_generator,
                proxy_model=proxy_model,
            )
        else:
            result = self.no_mixup(
                selected_batch_size=selected_batch_size,
                data=data,
                target=target,
                global_index=global_index,
                large_model=large_model,
                irreducible_loss_generator=irreducible_loss_generator,
                proxy_model=proxy_model,
            )

        return result
    

class reducible_loss_selection_mixup:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        with torch.no_grad():
            if proxy_model is not None:
                model_loss = F.cross_entropy(
                    proxy_model(data), target, reduction="none"
                )
            else:
                model_loss = F.cross_entropy(
                    large_model(data), target, reduction="none"
                )

            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator,
                model_loss.device,
            )

            reducible_loss = model_loss - irreducible_loss

            # Sort indices by reducible loss (descending order)
            sorted_indices = torch.argsort(reducible_loss, descending=True)

            # Get top `selected_batch_size` and next `selected_batch_size`
            selected_minibatch = sorted_indices[:selected_batch_size]
            not_selected_minibatch = sorted_indices[selected_batch_size:]
            next_selected_minibatch = sorted_indices[selected_batch_size:2 * selected_batch_size]

            selected_irreducible_loss = irreducible_loss[selected_minibatch]

            # MixUp: Compute mixup factor (lambda) from softmax-normalized reducible loss
            lambda_values = 1/(1 + torch.exp(reducible_loss[next_selected_minibatch])*torch.exp(-reducible_loss[selected_minibatch]))

            # Apply MixUp interpolation
            mixed_data = lambda_values.view(-1, 1, 1, 1) * data[selected_minibatch] + \
                        (1 - lambda_values.view(-1, 1, 1, 1)) * data[next_selected_minibatch]

            mixed_target = lambda_values * target[selected_minibatch] + \
                        (1 - lambda_values) * target[next_selected_minibatch]

# compute what proportion of the reducible loss is made up of the points with the highest 20% reducible losses
        batch_size = len(
            data
        )  # get big batch size, to compute what proportion of the batch model loss is made up by the points in the batch with the highest loss
        reducible_loss_sorted, _ = torch.sort(reducible_loss, descending=True)
        reducible_loss_sorted[
            reducible_loss_sorted < 0
        ] = 0.0  # ignore negative reducible losses
        percentage_top_20 = torch.sum(
            reducible_loss_sorted[: int(0.2 * batch_size)]
        ) / torch.sum(reducible_loss_sorted)

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "model_loss": model_loss,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log[
            "proportion_of_total_batch_reducible_loss_corresponding_to_the_top_20%_points_with_highest_reducible_loss"
        ] = float(percentage_top_20)

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        wandb.log({'elapsed_time': time.time() - wandb.config.start_time}, commit=False)

        return selected_minibatch, metrics_to_log, selected_irreducible_loss
    

class threshold_reducible_loss_selection:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        """
        Selects samples using reducible loss, optionally based on proxy model
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module
             Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
             nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
            compare_to_large_model_selection: if True, when proxy is specified, also calculate points using the large model,
                                                and include statistics of the selected points
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """
        with torch.no_grad():
            if proxy_model is not None:
                model_loss = F.cross_entropy(
                    proxy_model(data), target, reduction="none"
                )
            else:
                model_loss = F.cross_entropy(
                    large_model(data), target, reduction="none"
                )

            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator,
                model_loss.device,
            )

            reducible_loss = model_loss - irreducible_loss

            if torch.sum(reducible_loss > 0.0) > 10:
                selected_minibatch = torch.where(reducible_loss > 0.0)[0]
                not_selected_minibatch = torch.where(reducible_loss <= 0.0)[0]
            else:
                selected_minibatch, not_selected_minibatch = top_x_indices(
                reducible_loss, 10, largest=True)
            selected_irreducible_loss = irreducible_loss[selected_minibatch]

        # compute what proportion of the reducible loss is made up of the points with the highest 20% reducible losses
        batch_size = len(
            data
        )  # get big batch size, to compute what proportion of the batch model loss is made up by the points in the batch with the highest loss
        reducible_loss_sorted, _ = torch.sort(reducible_loss, descending=True)
        reducible_loss_sorted[
            reducible_loss_sorted < 0
        ] = 0.0  # ignore negative reducible losses
        percentage_top_20 = torch.sum(
            reducible_loss_sorted[: int(0.2 * batch_size)]
        ) / torch.sum(reducible_loss_sorted)

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "model_loss": model_loss,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log[
            "proportion_of_total_batch_reducible_loss_corresponding_to_the_top_20%_points_with_highest_reducible_loss"
        ] = float(percentage_top_20)

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        wandb.log({'elapsed_time': time.time() - wandb.config.start_time}, commit=False)

        return selected_minibatch, metrics_to_log, selected_irreducible_loss


class sublabels_reducible_loss_selection:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        sub_label_mapping=None,
        num_sub_labels=None,
        *args,
        **kwargs,
    ):
        with torch.no_grad():
            if proxy_model is not None:
                model = proxy_model
            else:
                model = large_model

            model_loss = F.cross_entropy(
                model(data), target, reduction="none"
            )

            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator,
                model_loss.device,
            )

            reducible_loss = model_loss - irreducible_loss

            # Ensure one sample per sub-label in the selected minibatch
            selected_minibatch = []
            used_sub_labels = set()
            for idx in torch.argsort(reducible_loss, descending=True):
                if len(selected_minibatch) >= selected_batch_size:
                    break
                sub_label = sub_label_mapping.get(
                    global_index[idx].item(), None)
                if sub_label is None:
                    raise KeyError(
                        f"Global index {global_index[idx].item()} not found in sub_label_mapping.")
                if sub_label not in used_sub_labels:
                    selected_minibatch.append(idx)
                    used_sub_labels.add(sub_label)

            selected_minibatch = torch.tensor(
                selected_minibatch, dtype=torch.long)
            not_selected_minibatch = torch.tensor([
                idx for idx in range(len(global_index)) if idx not in selected_minibatch
            ], dtype=torch.long)

            selected_irreducible_loss = irreducible_loss[selected_minibatch]

        # compute what proportion of the reducible loss is made up of the points with the highest 20% reducible losses
        batch_size = len(data)
        reducible_loss_sorted, _ = torch.sort(reducible_loss, descending=True)
        reducible_loss_sorted[reducible_loss_sorted < 0] = 0.0
        percentage_top_20 = torch.sum(
            reducible_loss_sorted[: int(0.2 * batch_size)]
        ) / torch.sum(reducible_loss_sorted)

        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "model_loss": model_loss,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log[
            "proportion_of_total_batch_reducible_loss_corresponding_to_the_top_20%_points_with_highest_reducible_loss"
        ] = float(percentage_top_20)

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]

        wandb.log({'elapsed_time': time.time() - wandb.config.start_time}, commit=False)

        return selected_minibatch, metrics_to_log, selected_irreducible_loss


class distance_gated_reducible_loss_selection:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        embedding_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        with torch.no_grad():
            if proxy_model is not None:
                model = proxy_model
            else:
                model = large_model

            model_loss = F.cross_entropy(
                model(data), target, reduction="none"
            )

            # Extract embedding
            embedding = embedding_generator.get_embedding(data)

            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator,
                model_loss.device,
            )

            reducible_loss = model_loss - irreducible_loss

            selected_minibatch, not_selected_minibatch = top_x_distant_indices(
                reducible_loss, selected_batch_size, largest=True, embedding=embedding, dis=10
            )

            selected_irreducible_loss = irreducible_loss[selected_minibatch]

        # compute what proportion of the reducible loss is made up of the points with the highest 20% reducible losses
        batch_size = len(
            data
        )  # get big batch size, to compute what proportion of the batch model loss is made up by the points in the batch with the highest loss
        reducible_loss_sorted, _ = torch.sort(reducible_loss, descending=True)
        reducible_loss_sorted[
            reducible_loss_sorted < 0
        ] = 0.0  # ignore negative reducible losses
        percentage_top_20 = torch.sum(
            reducible_loss_sorted[: int(0.2 * batch_size)]
        ) / torch.sum(reducible_loss_sorted)

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "model_loss": model_loss,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log[
            "proportion_of_total_batch_reducible_loss_corresponding_to_the_top_20%_points_with_highest_reducible_loss"
        ] = float(percentage_top_20)

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]

        wandb.log({'elapsed_time': time.time() - wandb.config.start_time}, commit=False)

        return selected_minibatch, metrics_to_log, selected_irreducible_loss


class uniform_selection:
    bald = False

    def __init__(self, tracking=False):
        self.tracking = tracking

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        """
        Selects samples with uniform probability
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample [Not Required]
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset [not required]
            large_model: nn.Module, PyTorch Model of the large model [Not Required]
            irreducible_loss_generator: Tensor or nn.Module [Not Required]
                Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
                nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model [Not Required]
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples, w.r.t. to the current minibatch.
            metrics_to_log: dictionary, of metrics to log
        """
        selected_minibatch = torch.randperm(len(data))[:selected_batch_size]
        metrics_to_log = {"detailed_only_keys": []}
        irreducible_loss = None

        if self.tracking:
            with torch.inference_mode():
                logits = (
                    proxy_model(
                        data) if proxy_model is not None else large_model(data)
                )
                model_loss = F.cross_entropy(logits, target, reduction="none")

                _, num_classes = logits.shape
                probs = F.softmax(logits, dim=1)
                one_hot_targets = F.one_hot(target, num_classes=num_classes)
                g_i_norm_ub = torch.norm(probs - one_hot_targets, dim=-1)

                irreducible_loss = _compute_irreducible_loss(
                    data,
                    target,
                    global_index,
                    irreducible_loss_generator,
                    model_loss.device,
                )

                reducible_loss = model_loss - irreducible_loss

                batch_size = len(
                    data
                )  # get big batch size, to compute what proportion of the batch model loss is made up by the points in the batch with the highest loss

                def compute_percentage_top_20(tensor, batch_size):
                    tensor[tensor < 0.0] = 0.0  # filter out zeros
                    sorted_tensor, _ = torch.sort(tensor, descending=True)
                    percentage_top_20 = torch.sum(
                        sorted_tensor[: int(0.2 * batch_size)]
                    ) / torch.sum(sorted_tensor)

                    return percentage_top_20

                for name, tensor in zip(
                    ["loss", "reducible_loss", "grad_norm"],
                    [model_loss, reducible_loss, g_i_norm_ub],
                ):
                    metrics_to_log[
                        f"proportion_of_total_batch_loss_corresponding_to_the_top_20%_points_with_highest_{name}"
                    ] = float(compute_percentage_top_20(tensor, batch_size))

        wandb.log({'elapsed_time': time.time() - wandb.config.start_time}, commit=False)

        return selected_minibatch, metrics_to_log, irreducible_loss


class reducible_loss_selection:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        """
        Selects samples using reducible loss, optionally based on proxy model
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module
             Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
             nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
            compare_to_large_model_selection: if True, when proxy is specified, also calculate points using the large model,
                                                and include statistics of the selected points
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """
        with torch.no_grad():
            if proxy_model is not None:
                model_loss = F.cross_entropy(
                    proxy_model(data), target, reduction="none"
                )
            else:
                model_loss = F.cross_entropy(
                    large_model(data), target, reduction="none"
                )

            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator,
                model_loss.device,
            )

            reducible_loss = model_loss - irreducible_loss

            selected_minibatch, not_selected_minibatch = top_x_indices(
                reducible_loss, selected_batch_size, largest=True
            )
            selected_irreducible_loss = irreducible_loss[selected_minibatch]

        # compute what proportion of the reducible loss is made up of the points with the highest 20% reducible losses
        batch_size = len(
            data
        )  # get big batch size, to compute what proportion of the batch model loss is made up by the points in the batch with the highest loss
        reducible_loss_sorted, _ = torch.sort(reducible_loss, descending=True)
        reducible_loss_sorted[
            reducible_loss_sorted < 0
        ] = 0.0  # ignore negative reducible losses
        percentage_top_20 = torch.sum(
            reducible_loss_sorted[: int(0.2 * batch_size)]
        ) / torch.sum(reducible_loss_sorted)

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "model_loss": model_loss,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log[
            "proportion_of_total_batch_reducible_loss_corresponding_to_the_top_20%_points_with_highest_reducible_loss"
        ] = float(percentage_top_20)

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        wandb.log({'elapsed_time': time.time() - wandb.config.start_time}, commit=False)

        return selected_minibatch, metrics_to_log, selected_irreducible_loss


class reducible_loss_selection_full_model_comp:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data,
        target,
        global_index,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        """
        Selects samples using reducible loss, optionally based on proxy model
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module
             Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
             nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
            compare_to_large_model_selection: if True, when proxy is specified, also calculate points using the large model,
                                                and include statistics of the selected points
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """

        assert proxy_model is not None

        with torch.no_grad():
            proxy_loss = F.cross_entropy(
                proxy_model(data), target, reduction="none")
            target_loss = F.cross_entropy(
                large_model(data), target, reduction="none")

            # compute irreducible loss

            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator,
                target_loss.device,
            )

            proxy_reducible_loss = proxy_loss - irreducible_loss
            proxy_selected_minibatch, proxy_not_selected_minibatch = top_x_indices(
                proxy_reducible_loss, selected_batch_size, largest=True
            )
            proxy_selected_irreducible_loss = irreducible_loss[proxy_selected_minibatch]

            target_reducible_loss = target_loss - irreducible_loss
            target_selected_minibatch, target_not_selected_minibatch = top_x_indices(
                target_reducible_loss, selected_batch_size, largest=True
            )

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "proxy_model_loss": proxy_loss,
            "target_model_loss": target_loss,
        }

        # log metrics of the batch selected by the proxy model,
        proxy_metrics_to_log = create_logging_dict(
            variables_to_log, proxy_selected_minibatch, proxy_not_selected_minibatch
        )
        # attach "proxy_" to the start of each key
        for key in proxy_metrics_to_log.keys():
            proxy_metrics_to_log["proxy_" +
                                 key] = proxy_metrics_to_log.pop(key)

        # log metrics of the batch selected by the target model,
        target_metrics_to_log = create_logging_dict(
            variables_to_log, target_selected_minibatch, target_not_selected_minibatch
        )
        # attach "target_" to the start of each key
        for key in target_metrics_to_log.keys():
            target_metrics_to_log["target_" +
                                  key] = target_metrics_to_log.pop(key)

        # merge the two dicts
        metrics_to_log = {
            **proxy_metrics_to_log,
            **target_metrics_to_log,
        }  # not using .update() to avoid in-pace modification

        proxy_target_overlap = float(
            np.in1d(
                metrics_to_log["target_selected_id"],
                metrics_to_log["proxy_selected_id"],
            )
            / selected_batch_size
        )
        metrics_to_log[proxy_target_overlap] = proxy_target_overlap

        metrics_to_log["detailed_only_keys"] = [
            "proxy_selected_id",
            "target_selected_id",
            "proxy_not_selected_id",
            "target_not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        return proxy_selected_minibatch, metrics_to_log, proxy_selected_irreducible_loss


class ce_loss_selection:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data,
        target,
        global_index,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        """
        Selects samples using CE loss, optionally with proxy model
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset [not required]
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module [Not Required]
                Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
                nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """
        with torch.no_grad():
            if proxy_model is not None:
                model_loss = F.cross_entropy(
                    proxy_model(data), target, reduction="none"
                )
            else:
                model_loss = F.cross_entropy(
                    large_model(data), target, reduction="none"
                )

            selected_minibatch, not_selected_minibatch = top_x_indices(
                model_loss, selected_batch_size, largest=True
            )

        # compute what proportion of the loss is made up of the points with the highest 20% losses
        batch_size = len(
            data
        )  # get big batch size, to compute what proportion of the batch model loss is made up by the points in the batch with the highest loss
        model_loss_sorted, _ = torch.sort(model_loss, descending=True)
        percentage_top_20 = torch.sum(
            model_loss_sorted[: int(0.2 * batch_size)]
        ) / torch.sum(model_loss_sorted)

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "model_loss": model_loss,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log[
            "proportion_of_total_batch_loss_corresponding_to_the_top_20%_points_with_highest_loss"
        ] = float(percentage_top_20)

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        irreducible_loss = None
        return selected_minibatch, metrics_to_log, irreducible_loss


class irreducible_loss_selection:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data,
        target,
        global_index,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        """
        Selects samples using irreducible loss
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset
            large_model: nn.Module, PyTorch Model of the large model [Not Required]
            irreducible_loss_generator: Tensor or nn.Module
                Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
                nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model [Not Required]
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """
        with torch.no_grad():
            # compute irreducible loss
            irreducible_loss = _compute_irreducible_loss(
                data, target, global_index, irreducible_loss_generator
            )

            selected_minibatch, not_selected_minibatch = top_x_indices(
                irreducible_loss, selected_batch_size, largest=False
            )
            selected_irreducible_loss = irreducible_loss[selected_minibatch]

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        return selected_minibatch, metrics_to_log, selected_irreducible_loss


class gradnorm_ub_selection:
    bald = False

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        *args,
        **kwargs,
    ):
        """
        Selects samples using an upper bound on the gradient norm, optionally on a proxy model. Note that the gradient
        norm is approximated using an upper bound.

        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module
             Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
             nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
            compare_to_large_model_selection: if True, when proxy is specified, also calculate points using the large model,
                                                and include statistics of the selected points
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """
        with torch.inference_mode():
            logits = proxy_model(
                data) if proxy_model is not None else large_model(data)
            _, num_classes = logits.shape
            probs = F.softmax(logits, dim=1)
            one_hot_targets = F.one_hot(target, num_classes=num_classes)
            g_i_norm_ub = torch.norm(probs - one_hot_targets, dim=-1)

            selected_minibatch, not_selected_minibatch = top_x_indices(
                g_i_norm_ub, selected_batch_size, largest=True
            )

        # compute what proportion of the gradnorm is made up of the points with the highest 20% gradnorm
        batch_size = len(
            data
        )  # get big batch size, to compute what proportion of the batch model loss is made up by the points in the batch with the highest loss
        g_i_norm_sorted, _ = torch.sort(g_i_norm_ub, descending=True)
        percentage_top_20 = torch.sum(
            g_i_norm_sorted[: int(0.2 * batch_size)]
        ) / torch.sum(g_i_norm_sorted)

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {"id": global_index, "g_i_norm_ub": g_i_norm_ub}

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log[
            "proportion_of_total_gradient_norm_corresponding_to_the_top_20%_points_with_highest_gradient_norm"
        ] = float(percentage_top_20)

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        return selected_minibatch, metrics_to_log, None


class boltzmann_reducible_loss_selection:
    bald = False

    def __init__(self, temperature_schedule=None):
        """
        Initialize boltzmann reducible loss selection.

        Args:
            temperature_schedule: defaults to function returning 0.25. If temperature_schedule is a number, the temperature
                                    is fixed to be that number. Otherwise, temperature schedule should be a function which
                                    takes in the current epoch and returns the temperature for that epoch.
        """
        if temperature_schedule is None:
            self.temperature_schedule = lambda epoch: 0.25
        elif isinstance(temperature_schedule, numbers.Number):
            self.temperature_schedule = lambda epoch: float(
                temperature_schedule)
        else:
            self.temperature_schedule = temperature_schedule

    def __call__(
        self,
        selected_batch_size,
        data,
        target,
        global_index,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        current_epoch=None,
        *args,
        **kwargs,
    ):
        """
        Selects samples using reducible loss, optionally based on proxy model
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module
             Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
             nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
            current_epoch: takes current epoch, used to schedule temperature.
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """

        if current_epoch is None:
            warnings.warn(
                "boltzmann reducible loss selection has not been provided the current epoch; epoch reverting to one"
            )
            current_epoch = 1

        full_batch_size = len(data)

        with torch.inference_mode():
            # compute CE loss
            if proxy_model is not None:
                model_loss = F.cross_entropy(
                    proxy_model(data), target, reduction="none"
                )
            else:
                model_loss = F.cross_entropy(
                    large_model(data), target, reduction="none"
                )

            # compute irreducible loss
            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator,
                model_loss.device,
            )

            reducible_loss = model_loss - irreducible_loss

            temperature = self.temperature_schedule(current_epoch)
            probabilities = F.softmax(reducible_loss / temperature)

            entropy = (
                torch.sum(probabilities * torch.log(1.0 /
                          probabilities)).cpu().numpy()
            )

            indices = torch.arange(
                0, full_batch_size, device=reducible_loss.device)
            selected_minibatch = torch.multinomial(
                probabilities, selected_batch_size, replacement=True
            )

            # compute selected (unique) and not selected point
            combined = torch.cat((indices, selected_minibatch))
            uniques, counts = combined.unique(return_counts=True)
            not_selected_minibatch = uniques[counts == 1]

            selected_irreducible_loss = irreducible_loss[selected_minibatch]

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "irreducible_loss": irreducible_loss,
            "model_loss": model_loss,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        # add entropy to logging
        metrics_to_log["sampling_dist_entropy"] = entropy

        return selected_minibatch, metrics_to_log, selected_irreducible_loss


class reducible_loss_to_uniform_after_X_epochs_selection:
    bald = False

    def __init__(self, switch_to_uniform_epoch=25):
        self.switch_to_uniform_epoch = switch_to_uniform_epoch
        self.reducible_loss_selection = reducible_loss_selection
        self.uniform_selection = uniform_selection

    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        current_epoch=None,
        *args,
        **kwargs,
    ):
        if current_epoch >= self.switch_to_uniform_epoch:
            self.uniform_selection(
                selected_batch_size,
                data=data,
                target=target,
                global_index=global_index,
                large_model=large_model,
                irreducible_loss_generator=irreducible_loss_generator,
                proxy_model=proxy_model,
            )

        else:
            self.reducible_loss_selection(
                selected_batch_size,
                data=data,
                target=target,
                global_index=global_index,
                large_model=large_model,
                irreducible_loss_generator=irreducible_loss_generator,
                proxy_model=proxy_model,
            )


# Active Learning Baselines
class bald_selection:
    bald = True

    def __call__(
        self,
        selected_batch_size,
        data,
        target,
        global_index,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        num_mc=10,
        num_classes=10,
        *args,
        **kwargs,
    ):
        """
        Selects samples with using BALD
        Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máté Lengyel. Bayesian active learning for classification and preference learning. arXiv preprint arXiv:1112.5745, 2011.
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset [not required]
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module [Not Required]
                Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
                nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model [Not Required]
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """
        enable_dropout(large_model)
        predictions = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        log_probs = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        with torch.no_grad():
            for i in range(num_mc):
                predictions[i] = large_model(data)
                log_probs[i] = F.log_softmax(predictions[i], dim=-1)
            bald = get_bald(log_probs.transpose(0, 1))
            selected_minibatch, not_selected_minibatch = top_x_indices(
                bald, selected_batch_size, largest=True
            )

        # Define the metrics that you want to log. Will log the metrics for
        # selected and not-selected points separately.
        variables_to_log = {
            "id": global_index,
            "bald": bald,
        }

        metrics_to_log = create_logging_dict(
            variables_to_log, selected_minibatch, not_selected_minibatch
        )

        metrics_to_log["detailed_only_keys"] = [
            "selected_id",
            "not_selected_id",
        ]  # I (Jan) did not understand why this key/value pair was here in the first place, so I left it here for now.

        irreducible_loss = None
        return selected_minibatch, metrics_to_log, irreducible_loss


class entropy_selection:
    bald = True

    def __call__(
        self,
        selected_batch_size,
        data,
        target,
        global_index,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        num_mc=10,
        num_classes=10,
        *args,
        **kwargs,
    ):
        """
        Selects samples using output entropy
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset [not required]
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module [Not Required]
                Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
                nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """
        enable_dropout(large_model)
        predictions = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        log_probs = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        with torch.no_grad():
            for i in range(num_mc):
                predictions[i] = large_model(data)
                log_probs[i] = F.log_softmax(predictions[i], dim=-1)
            entropy = compute_entropy(log_probs.transpose(0, 1))
            selected_minibatch, not_selected_minibatch = top_x_indices(
                entropy, selected_batch_size, largest=True
            )

        metrics_to_log = {"detailed_only_keys": []}

        return selected_minibatch, metrics_to_log, None


class conditional_entropy_selection:
    bald = True

    def __call__(
            self,
            selected_batch_size,
            data,
            target,
            global_index,
            large_model=None,
            irreducible_loss_generator=None,
            proxy_model=None,
            num_mc=10,
            num_classes=10,
            *args,
            **kwargs,
    ):
        """
        Selects samples using conditional entropy
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset [not required]
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module [Not Required]
                Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
                nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """
        enable_dropout(large_model)
        predictions = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        log_probs = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        with torch.no_grad():
            for i in range(num_mc):
                predictions[i] = large_model(data)
                log_probs[i] = F.log_softmax(predictions[i], dim=-1)
            conditional_entropy = compute_conditional_entropy(
                log_probs.transpose(0, 1))
            selected_minibatch, not_selected_minibatch = top_x_indices(
                conditional_entropy, selected_batch_size, largest=True
            )

        metrics_to_log = {"detailed_only_keys": []}

        return selected_minibatch, metrics_to_log, None


class loss_minus_conditional_entropy_selection:
    bald = True

    def __call__(
        self,
        selected_batch_size,
        data,
        target,
        global_index,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        num_mc=10,
        num_classes=10,
        *args,
        **kwargs,
    ):
        """
        Selects samples using loss - conditional entropy entropy
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset [not required]
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module [Not Required]
                Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
                nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """

        enable_dropout(large_model)
        predictions = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        loss = torch.zeros(
            (num_mc, len(data)), device=data.device
        )

        log_probs = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        with torch.no_grad():
            for i in range(num_mc):
                predictions[i] = large_model(data)
                log_probs[i] = F.log_softmax(predictions[i], dim=-1)

                loss[i] = F.cross_entropy(
                    predictions[i], target, reduction="none")

            conditional_entropy = compute_conditional_entropy(
                log_probs.transpose(0, 1))
            mean_loss = loss.mean(axis=0)
            selected_minibatch, not_selected_minibatch = top_x_indices(
                mean_loss - conditional_entropy, selected_batch_size, largest=True
            )

        metrics_to_log = {"detailed_only_keys": []}

        irreducible_loss = None
        return selected_minibatch, metrics_to_log, None


class loss_plus_conditional_entropy_selection:
    bald = True

    def __call__(
            self,
            selected_batch_size,
            data,
            target,
            global_index,
            large_model=None,
            irreducible_loss_generator=None,
            proxy_model=None,
            num_mc=10,
            num_classes=10,
            *args,
            **kwargs,
    ):
        """
        Selects samples using loss + conditional entropy
        Args:
            selected_batch_size: int, number of samples to be selected
            data: tensor, data of sample
            target: tensor, label/target of sample
            global_index: tensor, the (global) indices of the datapoints in <data>, w.r.t. to the whole dataset [not required]
            large_model: nn.Module, PyTorch Model of the large model
            irreducible_loss_generator: Tensor or nn.Module [Not Required]
                Tensor: with irreducible losses for train set, ordered by <global_index> (see datamodules)
                nn.Module: irreducible loss model
            proxy_model: nn.Module, PyTorch Model of the model that acts as a proxy for the large model
        Returns:
            selected_minibatch: tensor, of (local) indices of the selected samples (wrt to the current minibatch)
            metrics_to_log: dictionary, of metrics to log
        """

        enable_dropout(large_model)
        predictions = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        loss = torch.zeros(
            (num_mc, len(data)), device=data.device
        )

        log_probs = torch.zeros(
            (num_mc, len(data), num_classes), device=data.device
        )

        with torch.no_grad():
            for i in range(num_mc):
                predictions[i] = large_model(data)
                log_probs[i] = F.log_softmax(predictions[i], dim=-1)

                loss[i] = F.cross_entropy(
                    predictions[i], target, reduction="none")

            conditional_entropy = compute_conditional_entropy(
                log_probs[i].transpose(0, 1))
            mean_loss = loss.mean(axis=0)
            selected_minibatch, not_selected_minibatch = top_x_indices(
                mean_loss - conditional_entropy, selected_batch_size, largest=True
            )

        metrics_to_log = {"detailed_only_keys": []}

        irreducible_loss = None
        return selected_minibatch, metrics_to_log, None
