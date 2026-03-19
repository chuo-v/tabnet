from typing import List, Any, Dict
import torch
import numpy as np
from scipy.special import softmax
from pytorch_tabnet import tab_network
from pytorch_tabnet.utils import (
    SparsePredictDataset,
    PredictDataset,
    filter_weights,
    create_group_matrix,
    create_explain_matrix,
)
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from torch.utils.data import DataLoader
import scipy


class TabNetClassifier(TabModel):
    def __init__(
        self,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        cat_idxs: List[int] = [],
        cat_dims: List[int] = [],
        cat_emb_dim: int = 1,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        momentum: float = 0.02,
        lambda_sparse: float = 1e-3,
        seed: int = 0,
        clip_value: int = 1,
        verbose: int = 1,
        optimizer_fn: Any = torch.optim.Adam,
        optimizer_params: Dict = dict(lr=2e-2),
        scheduler_fn: Any = None,
        scheduler_params: Dict = dict(),
        mask_type: str = "sparsemax",
        input_dim: int = None,
        output_dim: int = None,
        device_name: str = "auto",
        n_shared_decoder: int = 1,
        n_indep_decoder: int = 1,
        grouped_features: List[List[int]] = [],
        use_kan: bool = False,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        **kwargs
    ):
        super(TabNetClassifier, self).__init__(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            momentum=momentum,
            lambda_sparse=lambda_sparse,
            seed=seed,
            clip_value=clip_value,
            verbose=verbose,
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            scheduler_fn=scheduler_fn,
            scheduler_params=scheduler_params,
            mask_type=mask_type,
            input_dim=input_dim,
            output_dim=output_dim,
            device_name=device_name,
            n_shared_decoder=n_shared_decoder,
            n_indep_decoder=n_indep_decoder,
            grouped_features=grouped_features,
            **kwargs
        )
        # store KAN parameters
        self.use_kan = use_kan
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order

    def __post_init__(self):
        super(TabNetClassifier, self).__post_init__()
        self._task = 'classification'
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = 'accuracy'

    def weight_updater(self, weights):
        """
        Updates weights dictionary according to target_mapper.

        Parameters
        ----------
        weights : bool or dict
            Given weights for balancing training.

        Returns
        -------
        bool or dict
            Same bool if weights are bool, updated dict otherwise.

        """
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    def prepare_target(self, y):
        return np.vectorize(self.target_mapper.get)(y)

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.long())

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = ('auc' if self.output_dim == 2 else 'accuracy')
        self.classes_ = train_labels
        self.target_mapper = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {
            str(index): class_label for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights = self.weight_updater(weights)

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        y_score = softmax(y_score, axis=1)
        return y_true, y_score

    def predict_func(self, outputs):
        outputs = np.argmax(outputs, axis=1)
        return np.vectorize(self.preds_mapper.get)(outputs.astype(str))

    def predict_proba(self, X):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        res : np.ndarray

        """
        self.network.eval()

        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            output, M_loss = self.network(data)
            predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return res

    def _set_network(self):
        """
        Instantiates the PyTorch neural network.
        Override is for passing KAN parameters and setting up matrices.
        """
        torch.manual_seed(self.seed)

        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)

        self.network = tab_network.TabNet(
            self.input_dim,
            self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dim,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
            group_attention_matrix=self.group_matrix.to(self.device),
            use_kan=self.use_kan,
            kan_grid_size=self.kan_grid_size,
            kan_spline_order=self.kan_spline_order
        ).to(self.device)

        self.reducing_matrix = create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )


class TabNetRegressor(TabModel):
    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()
        self._task = 'regression'
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = 'mse'

    def prepare_target(self, y):
        return y

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights
    ):
        if len(y_train.shape) != 2:
            msg = "Targets should be 2D : (n_samples, n_regression) " + \
                  f"but y_train.shape={y_train.shape} given.\n" + \
                  "Use reshape(-1, 1) for single regression."
            raise ValueError(msg)
        self.output_dim = y_train.shape[1]
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score
