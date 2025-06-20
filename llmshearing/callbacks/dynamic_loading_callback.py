import os
import time
from typing import Any, Dict, List, Tuple

import torch
from composer import Callback, Logger, State
from composer.loggers import Logger
from composer.utils import dist
from torch.nn import functional as F

def _project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """Duchi et al. (2008) projection onto the probability simplex."""
    # sort descending
    vs, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(vs, dim=0) - 1          # cumulative sum minus 1
    rho = torch.nonzero(vs - cssv / torch.arange(1, v.numel() + 1, device=v.device) > 0,
                        as_tuple=False).max()
    tau = cssv[rho] / (rho + 1).float()
    return torch.clamp(v - tau, min=0.0)


class DynamicLoadingCallback(Callback):
    """ 
        Callback for dynamic loading of data from different domains. The key components include 1) calculate the new proportion after each evaluation step; 2) update proportion in the dataset objective; 3) save the used domain ids after each epoch for resuming training from a previous checkpoint to make sure that used samples are not used again.
    """
    def __init__(self, 
                 target_loss: List[float] = [], 
                 proportion: List[float] = [],
                 set_names: List[str] = [],
                 update_type: str = "", 
                 ) -> None:
        self.set_names = set_names
        self.n_domains = len(set_names)
        self.update_type = update_type 
        self.target_loss = target_loss
        self.proportion = proportion
        self.initial_proportion = proportion 
        self.count = -1
        self.used_domain_ids = [[] for _ in range(self.n_domains)]
        print("Target loss:", self.target_loss)
            
    def update_proportion(self, current_prop, current_lambdas, losses) -> Tuple[List[float], List[float]]:
        """ Update the proportion of each domain """
        # Get current learning rate from the first parameter group
        # current_lr = state.optimizers[0].param_groups[0]['lr']
        
        diff = torch.tensor(losses) - torch.tensor(self.target_loss)
        eta = 1.
        c = 1e-4 # following Doremi (Xie et al., 2023)
        extrapolation_factor = 1
        new_lambdas, updated_alpha = torch.tensor(current_lambdas), torch.tensor(current_prop)
        if self.update_type == "doremi": # update with exponential descent
            updated_alpha = torch.log(updated_alpha + 1e-6) + eta * diff 
            updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
            updated_domain_weights = (1-c) * updated_alpha + c / self.n_domains # convex combination with uniform distribution
        elif self.update_type == "bandit": 
            updated_alpha = updated_alpha + eta * diff 
            updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
            updated_domain_weights = (1-c) * updated_alpha + c / self.n_domains # convex combination with uniform distribution
        elif self.update_type == "pd-kl":
            new_lambdas = torch.log(new_lambdas + 1e-6) + eta * diff 
            new_lambdas = torch.nn.functional.softmax(new_lambdas, dim=0)
            updated_domain_weights = \
                new_lambdas + extrapolation_factor * (new_lambdas - torch.tensor(current_lambdas)) # extrapolation
            updated_domain_weights = (1-c) * updated_domain_weights + c / self.n_domains            
        elif self.update_type == "pd-chi-square":
            eta = 0.15
            new_lambdas = new_lambdas + eta * diff * torch.tensor(self.initial_proportion)
            new_lambdas = _project_to_simplex(new_lambdas)
            updated_domain_weights = \
                new_lambdas + extrapolation_factor * (new_lambdas - torch.tensor(current_lambdas)) # extrapolation
            updated_domain_weights = (1-c) * updated_domain_weights + c / self.n_domains
        elif self.update_type == "constant": # constant proportion
            return current_prop, current_lambdas            

        updated_domain_weights = updated_domain_weights.numpy().astype('float64')
        updated_domain_weights = updated_domain_weights / updated_domain_weights.sum()
        return updated_domain_weights.tolist(), new_lambdas.tolist()
    
    def after_train_batch(self, state: State, logger: Logger) -> None:
        """ Print out the number of used samples in each domain after each training batch, and log the updated proportion of each domain """
        idx = state.batch["idx"]
        sets = state.batch["set"]
        all_idx = torch.cat(dist.all_gather(idx))  # type: ignore
        all_sets = torch.cat(dist.all_gather(sets)) # type: ignore
        dist.barrier() 
        
        for i in range(self.n_domains):
            mask = all_sets == i
            domain_idx = all_idx[mask]
            self.used_domain_ids[i].extend(domain_idx.cpu().tolist())
            # for debugging
            # print(f"domain {i} used {mask.sum().item()} new samples")

        prop = state.train_dataloader.dataset.proportion
        for domain in self.set_names:
            logger.log_metrics({f'metrics/train/{domain}_weight': round(prop[self.set_names.index(domain)], 4)})
        
    def eval_end(self, state: State, logger: Logger) -> None:
        """ Update the proportion of each domain after each evaluation and update the dataset """
        current_prop = state.train_dataloader.dataset.proportion
        current_lambdas = state.train_dataloader.dataset.lambdas
        losses = []
        for domain in self.set_names:
            losses.append(state.eval_metrics["eval"][f"{domain}_LanguageCrossEntropy"].compute().item())
        new_proportion, new_lambdas = self.update_proportion(current_prop, current_lambdas, losses)
        state.train_dataloader.dataset.update_proportion(new_proportion, new_lambdas)
    
    def state_dict(self) -> Dict[str, Any]:
        """ Save the used domain ids after each epoch, for resuming training from a previous checkpoint to make sure that used samples are not used again """
        return {"used_domain_ids": self.used_domain_ids}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """ Load the used domain ids """
        self.used_domain_ids = state_dict["used_domain_ids"]
     