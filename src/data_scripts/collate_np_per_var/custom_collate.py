# collate_np.py (modified FastCollate)
import sys
sys.dont_write_bytecode = True
import os
import time as clock
import logging
import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger()

from .dataset import DownscalingDataset
#====================================================================
class FastCollate:
    """
    New behaviour:
      - Accepts input_transforms: dict(var -> transform_obj) or None
      - Accepts target_transforms: dict(var -> transform_obj) or None
      - If the batch items are dict-based (new dataset), stacks per-variable into (B,C,H,W)
      - Applies per-variable transforms in a loop over variables (vectorized across batch).
      - Falls back to old behaviour if batch entries are numpy arrays.
    """
    def __init__(
            self,
            input_transforms=None,
            target_transforms=None,
            time_range=None,
            input_variable_order=None,
            target_variable_order=None
        ):
        # expected: input_transforms is either None or dict: {var: transform_obj}
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms
        self.time_range = time_range
        # allow user override of variable order; otherwise deduce from input_transforms or the first batch
        self.input_variable_order = list(input_variable_order) if input_variable_order is not None else None
        self.target_variable_order = list(target_variable_order) if target_variable_order is not None else None

    def _apply_transform_safe(self, xfm, arr):
        """
        Try several common call patterns for saved transforms:
          - xfm.transforms(np_array)
          - xfm.transform(np_array)
          - xfm(np_array)
        Accepts arr shape (B,H,W) or (B,1,H,W) or (B,C,H,W). Returns numpy arr.
        """
        if xfm is None:
            return arr
        try:
            if hasattr(xfm, "transforms") and callable(xfm.transforms):
                out = xfm.transforms(arr)
            elif hasattr(xfm, "transform") and callable(xfm.transform):
                out = xfm.transform(arr)
            elif callable(xfm):
                out = xfm(arr)
            else:
                raise AttributeError("No callable transform found")
        except Exception as e:
            # try adding a channel dimension if transform expects (B,C,H,W)
            try:
                arr_c = arr[:, None, ...] if arr.ndim == 3 else arr
                if hasattr(xfm, "transforms") and callable(xfm.transforms):
                    out = xfm.transforms(arr_c)
                elif hasattr(xfm, "transform") and callable(xfm.transform):
                    out = xfm.transform(arr_c)
                else:
                    out = xfm(arr_c)
            except Exception as e2:
                logger.exception("Transform application failed: %s ; fallback to identity. (%s / %s)", e, e2, type(xfm))
                return arr
        # ensure numpy
        if torch.is_tensor(out):
            out = out.cpu().numpy()
        else:
            out = np.asarray(out)
        # if returned shape (B,1,H,W) squeeze channel
        if out.ndim == 4 and out.shape[1] == 1:
            out = np.squeeze(out, axis=1)
        return out.astype(np.float32)

    def __call__(self, batch):
        """
        batch: list of (cond, targ, time)
           cond/targ can be either:
             - dict(var->np.array)  (new behavior)
             - stacked numpy arrays (B,C,H,W) style returned by older dataset (legacy)
        Returns:
           conds (torch.Tensor), targs (torch.Tensor), times (np.array)
        """
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        start_time = clock.time()

        # detect dict-mode
        first_cond = batch[0][0]
        if isinstance(first_cond, dict):
            # determine variable order
            if self.input_variable_order is not None:
                input_vars = list(self.input_variable_order)
            elif self.input_transforms is not None:
                input_vars = list(self.input_transforms.keys())
            else:
                input_vars = list(first_cond.keys())

            if self.target_variable_order is not None:
                target_vars = list(self.target_variable_order)
            elif self.target_transforms is not None:
                target_vars = list(self.target_transforms.keys())
            else:
                target_vars = list(batch[0][1].keys())

            B = len(batch)
            C_in = len(input_vars)
            C_out = len(target_vars)

            # get H,W from first variable sample
            sample_arr = np.asarray(first_cond[input_vars[0]])
            if sample_arr.ndim == 2:
                H, W = sample_arr.shape
                # preallocate
                conds = np.empty((B, C_in, H, W), dtype=np.float32)
                targs = np.empty((B, C_out, H, W), dtype=np.float32)
            elif sample_arr.ndim == 3:
                # sample_arr maybe (ch, H, W). treat channels as extra dim per-variable.
                ch = sample_arr.shape[0]
                H, W = sample_arr.shape[1], sample_arr.shape[2]
                conds = np.empty((B, C_in * ch, H, W), dtype=np.float32)
                # we will flatten channel dim into variable-channel axis. User should prefer single-channel vars.
            else:
                raise RuntimeError(f"Unexpected per-variable array ndim={sample_arr.ndim}")

            # fill conds: vectorized per-variable (one stack per variable)
            for i, v in enumerate(input_vars):
                var_stack = np.stack([np.asarray(b[0][v]) for b in batch], axis=0).astype(np.float32)  # (B, H, W) or (B, ch, H, W)
                # if returned with channel dim (B,ch,H,W), try to squeeze/reshape into conds
                if var_stack.ndim == 4:
                    # flatten channel into variable dimension: like C_in*ch
                    ch_dim = var_stack.shape[1]
                    conds[:, i*ch_dim:(i+1)*ch_dim, :, :] = var_stack
                else:
                    conds[:, i, :, :] = var_stack

            # fill targs
            for j, v in enumerate(target_vars):
                tvar_stack = np.stack([np.asarray(b[1][v]) for b in batch], axis=0).astype(np.float32)
                if tvar_stack.ndim == 4:
                    ch_dim = tvar_stack.shape[1]
                    targs[:, j*ch_dim:(j+1)*ch_dim, :, :] = tvar_stack
                else:
                    targs[:, j, :, :] = tvar_stack

            end_time = clock.time()
            #logger.debug(" >> >> INSIDE FastCollate: stacked batch (rank %d pid %d) concat time: %s", rank, os.getpid(), str(round(end_time - start_time, 7)))

            # apply per-variable transforms (vectorized across batch)
            # For input transforms: iterate over input_vars and transform conds[:,i,...]
            start_time = clock.time()
            if self.input_transforms is not None:
                # handle case where variables may have multiple channels per var (rare)
                for i, v in enumerate(input_vars):
                    xfm = self.input_transforms.get(v)
                    # determine slice indices: if each var is single-channel (common)
                    # We assume single-channel per variable; if multi-channel, the transform should accept (B,ch,H,W).
                    var_slice = conds[:, i, ...] if conds.ndim == 4 else conds[:, i, ...]
                    transformed = self._apply_transform_safe(xfm, var_slice)
                    # transformed should be (B,H,W) or (B,1,H,W) or (B,ch,H,W)
                    if transformed.ndim == 3:
                        conds[:, i, :, :] = transformed
                    elif transformed.ndim == 4 and transformed.shape[1] == 1:
                        conds[:, i, :, :] = np.squeeze(transformed, axis=1)
                    else:
                        # if transformed has multiple channels, try to place them
                        if transformed.ndim == 4:
                            ch = transformed.shape[1]
                            conds[:, i:i+ch, :, :] = transformed
                        else:
                            raise RuntimeError("Unexpected transformed shape for input var %s: %s" % (v, str(transformed.shape)))
            end_time = clock.time()
            #logger.info(" >> >> INSIDE FastCollate: input transforms applied (rank %d pid %d) time: %s", rank, os.getpid(), str(round(end_time - start_time, 7)))

            # apply target transforms
            start_time = clock.time()
            if self.target_transforms is not None:
                for j, v in enumerate(target_vars):
                    xfm = self.target_transforms.get(v)
                    t_slice = targs[:, j, ...]
                    transformed_t = self._apply_transform_safe(xfm, t_slice)
                    if transformed_t.ndim == 3:
                        targs[:, j, :, :] = transformed_t
                    elif transformed_t.ndim == 4 and transformed_t.shape[1] == 1:
                        targs[:, j, :, :] = np.squeeze(transformed_t, axis=1)
                    else:
                        if transformed_t.ndim == 4:
                            ch = transformed_t.shape[1]
                            targs[:, j:j+ch, :, :] = transformed_t
                        else:
                            raise RuntimeError("Unexpected transformed shape for target var %s: %s" % (v, str(transformed_t.shape)))
            end_time = clock.time()
            #logger.info(" >> >> INSIDE FastCollate: target transforms applied (rank %d pid %d) time: %s", rank, os.getpid(), str(round(end_time - start_time, 7)))

            # add time channels if requested
            if self.time_range is not None:
                times = np.array([b[2] for b in batch])
                cond_time_torch = DownscalingDataset.time_to_tensor(times, conds.shape, self.time_range)
                if cond_time_torch is not None:
                    cond_time_np = cond_time_torch.numpy()  # (B,3,H,W)
                    conds = np.concatenate([conds, cond_time_np], axis=1)

            # final conversion to torch
            conds_t = torch.from_numpy(conds)
            targs_t = torch.from_numpy(targs)
            times = np.array([b[2] for b in batch])
            return conds_t, targs_t, times
