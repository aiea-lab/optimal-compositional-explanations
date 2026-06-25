"""Central experiment configuration object and path policy definitions.

Encapsulates dataset, model, heuristic, and output directory settings.
"""

class Settings:
    """
    Class that stores all the settings used in each run.
    """

    def __init__(
        self,
        *,
        model,
        num_clusters,
        beam_limit,
        heuristic,
        length,
        layer,
        device,
        step_size=80,
        seed=0,
        fast_impl=False,
        dataset="places365",
        root_models="data/model/",
        root_datasets="data/dataset/",
        root_segmentations="data/cache/segmentations/",
        root_activations="data/cache/activations/",
        root_results="data/results/",
        root_optimal_info="data/cache/optimal_info",
    ):
        self.__model = model
        self.__dataset = dataset
        self.__num_clusters = num_clusters
        self.__beam_limit = beam_limit
        self.__heuristic = heuristic
        self.__root_datasets = root_datasets
        self.__root_segmentations = root_segmentations
        self.__root_activations = root_activations
        self.__root_results = root_results
        self.__root_models = root_models
        self.__root_optimal_info = root_optimal_info
        self.__device = device
        self.__step_size = step_size
        self.__layer = layer
        self.__length = length
        self.__seed = seed
        self.__fast_impl = fast_impl

    ###### Getters ######

    def get_root_activations(self):
        return self.__root_activations

    def get_root_datasets(self):
        return self.__root_datasets

    def get_dataset_name(self):
        return self.__dataset

    def get_model(self):
        return self.__model

    def get_num_clusters(self):
        return self.__num_clusters

    def get_beam_limit(self):
        return self.__beam_limit

    def get_heuristic(self):
        return self.__heuristic

    def get_length(self):
        return self.__length

    def get_layer(self):
        return self.__layer

    def get_device(self):
        return self.__device

    def get_step_size(self):
        return self.__step_size

    def get_seed(self):
        return self.__seed

    def fast_impl_is_enabled(self):
        return self.__fast_impl
    
    ###### Setters ######
    def set_heuristic(self, heuristic):
        self.__heuristic = heuristic

    ##### Getters combined with some logic #####

    def get_annotations_dir(self):
        segmentations_dir = f"{self.__root_segmentations}/{self.__dataset}/masks"
        return segmentations_dir

    def get_info_dir(self):
        info_dir = f"{self.__root_segmentations}/{self.__dataset}/info"
        return info_dir

    def get_result_dir(self):
        if self.get_heuristic() == "optimal":
            root = f"{self.__root_results}/optimal"
        elif self.get_heuristic() == "beam_optimal":
            root = f"{self.__root_results}/beam_optimal"
        elif self.get_heuristic() == "none":
            root = f"{self.__root_results}/no_heuristic"
        elif self.get_heuristic() == "mmesh":
            root = f"{self.__root_results}/mmesh"
        else:
            raise ValueError(f"Unknown heuristic {self.get_heuristic()}")
        if self.get_length() == 1:
            # vanilla netdissect explanation doesn't depend on the heuristic, so we can ignore it for the results dir and doesn't depend on beam limits
            results_dir = f"{self.__root_results}/{self.__dataset}/{self.__model}/{self.get_layer()}/length_{self.get_length()}/clusters_{self.get_num_clusters()}/no_limit/mask_shape_{self.get_mask_shape()[0]}x{self.get_mask_shape()[1]}"
        elif self.get_heuristic() == "optimal":
            results_dir = f"{root}/{self.__dataset}/{self.__model}/{self.get_layer()}/length_{self.get_length()}/clusters_{self.get_num_clusters()}/no_limit/mask_shape_{self.get_mask_shape()[0]}x{self.get_mask_shape()[1]}"
        else:
            results_dir = f"{root}/{self.__dataset}/{self.__model}/{self.get_layer()}/length_{self.get_length()}/clusters_{self.get_num_clusters()}/beam_limit_{self.get_beam_limit()}/mask_shape_{self.get_mask_shape()[0]}x{self.get_mask_shape()[1]}"
        return results_dir

    def get_mask_shape(self):
        """
        Returns the shape of the mask.
        In this paper we use masks of shape (112, 112) for all the datasets.
        """
        return (112, 112)

    def get_img_size(self):
        """
        Returns the size of the images.
        """
        if self.__model != "alexnet":
            return 224
        else:
            return 227

    def get_weights(self):
        """
        Returns the pretrained weights of the model.
        """
        if self.__model == "densenet161":
            model_file_name = "whole_densenet161_places365_python36.pth.tar"
        else:
            model_file_name = f"{self.__model}_places365.pth.tar"
        return self.__root_models + "/zoo/" + model_file_name
