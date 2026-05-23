"""Model wrappers for loading probe networks and extracting activations."""

import os
from typing import List

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from utils import dataset_utils


class ModelWrapper:

    # Reference: https://github.com/jayelm/compexp/blob/master/vision/loader/model_loader.py
    def hook(self, hook_fn, feature_names):
        """
        Register a hook to a model.

        Args:
            model (torch.nn.Module): model
            hook_fn (function): hook function
            feature_names (list): list of feature names

        Returns:
            list: list of handles
        """
        handles = []
        for name in feature_names:
            if isinstance(name, list):
                # Iteratively retrive the module
                hook_model = self.model
                for n in name:
                    hook_model = hook_model._modules.get(n)
            else:
                hook_model = self.model._modules.get(name)
            if hook_model is None:
                raise ValueError(f"Couldn't find feature {name}")
            handles.append(hook_model.register_forward_hook(hook_fn))
        return handles

    @torch.no_grad()
    def compute_activations(self, layers: List):
        """Retrieve the activations of a given layer feeding the model with the
        images in the loader.

        Args:
            layers (list): list of layers

        Returns:
            torch.Tensor: activations
        """
        device = next(self.model.parameters()).device
        temp_activations = []

        def hook_feature(module, inp, output):
            temp_activations.append(output.data.cpu())

        handles = self.hook(hook_feature, layers)

        activations = [[] for _ in range(len(layers))]
        for data in tqdm(self.data_loader, desc="Computing activations"):
            # Transformations
            images = self.iter_data(data)

            if len(images.shape) == 4:
                images = images.squeeze(0)

            if self.transformation is not None:
                images = self.transformation(images)
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

            # Move to GPU
            images = images.to(device)

            # Forward pass
            _ = self.model(images)

            # collect data
            for index_layer in range(len(layers)):
                activations[index_layer].append(temp_activations[index_layer])

            # Empty the temp list
            del temp_activations[:]
            temp_activations = []

        for handle in handles:
            handle.remove()
        for layer in range(len(layers)):
            activations[layer] = torch.unsqueeze(torch.cat(activations[layer]), dim=0)
        activations = torch.cat(activations, dim=0)
        return activations

    def get_layer_activations(self, layer, dir_activations):
        """Checks if the activations are already computed and saved, otherwise
        computes them and saves them.

        Args:
            model (torch.nn.Module): model
            layer (str): layer name
            units (list): list of units whose activations are to be computed

        Returns:
            activations (list): list of activations for each unit
        """
        if self.data_loader is None:
            raise ValueError(
                "Data loader not set. Please set the data loader first (run wrapper_item.set_loader(args))."
            )
        layer_file = f"{dir_activations}/{layer}.npy"
        total_activations = []
        if os.path.exists(layer_file):
            total_activations = np.load(layer_file)
        else:
            print(f"Computing activations for layer {layer}")

            activations = self.compute_activations([layer])
            # since the function checks one layer at a time
            activations = activations[0].numpy()
            if not os.path.exists(dir_activations):
                os.makedirs(dir_activations)
            np.save(f"{layer_file}", activations)
            total_activations = activations
        total_activations = torch.from_numpy(total_activations)
        return total_activations


class Place365Model(ModelWrapper):
    def __init__(self, model_name, weights, device):
        super().__init__()
        self.input_size = 227 if "alexnet" in model_name else 224
        self.model = self.load_checkpoint(model_name, weights, device)

    def set_loader(self, dataset_name, cfg=None):
        """Sets the data loader for the model.
        Args:
            dataset_name (str): name of the dataset to set the data loader for. It can be either a dataset registered in the detectron2 DatasetCatalog or 'broden'.
            cfg (Config): configuration object containing the dataset information. It is needed only if the dataset is 'broden'.
        """
        wrapper = (
            dataset_utils.BrodenWrapper(cfg)
            if dataset_name == "broden"
            else dataset_utils.DetectronWrapper(dataset_name)
        )
        self.data_loader = wrapper.data_loader
        self.iter_data = wrapper.data_iter_fn
        self.transformation = dataset_utils.get_probing_transformations(
            self.input_size, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

    def load_checkpoint(
        self,
        model_name,
        weights,
        device,
    ):
        """
        Loads the model from the checkpoint.
        Args:
            model_name (str): name of the model to load. It can be either 'alexnet' or 'densenet161'.
            weights (str): path to the checkpoint to load the model from.
            device (str): device to load the model on.
        Returns:
            model (torch.nn.Module): loaded model.
        """

        print(f"Loading model:{model_name}\n\tfrom {weights}")

        model_fn = torchvision.models.__dict__[model_name]

        checkpoint = torch.load(weights, map_location=device)
        model = model_fn(num_classes=365)
        # the data parallel layer will add 'module' before each
        # layer name
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model


class DenseNetPlace365(Place365Model):

    def load_checkpoint(
        self,
        model_name,
        weights,
        device,
    ):
        """
        Load the model from the settings.
        Args:
            config (settings.Settings): current config of the settings
            device (torch.device): device to use

        Returns:
            torch.nn.Module: model
        """

        def rep(k):
            for i in range(6):
                k = k.replace(f"norm.{i}", f"norm{i}")
                k = k.replace(f"relu.{i}", f"relu{i}")
                k = k.replace(f"conv.{i}", f"conv{i}")
            return k

        print(f"Loading model:{model_name}\n\tfrom {weights}")

        model_fn = torchvision.models.__dict__[model_name]

        checkpoint = torch.load(weights, map_location=device)
        model = model_fn(num_classes=365)
        # Fix old densenet pytorch names.
        state_dict = checkpoint.state_dict()
        state_dict = {rep(k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()
        print(f"{model_name} loaded in {device}. Modality: Evaluation")
        return model
