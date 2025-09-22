# # transforms_np.py
# import sys
# sys.dont_write_bytecode = True
# import os
# import time
# import gc
# import logging
# import pickle
# from datetime import timedelta
# from typing import Dict, Any, Iterable, List

# import numpy as np

# logger = logging.getLogger(__name__)

# from .data_scripts.data_utils import get_variables, open_zarr
# #====================================================================
# # -----------------------
# # save / load transforms
# # -----------------------
# def save_transform(xfm, path: str):
#     """Save transform as a pickle file."""
#     with open(path, "wb") as f:
#         logger.info(f"Storing transform: {path}")
#         pickle.dump(xfm, f, pickle.HIGHEST_PROTOCOL)


# def load_transform(path: str):
#     with open(path, "rb") as f:
#         logger.info(f"Using stored transform: {path}")
#         xfm = pickle.load(f)
#     return xfm

# # -----------------------
# # helpers for numpy-array friendly transforms
# # -----------------------
# def _is_numpy_array(x):
#     return isinstance(x, np.ndarray)

# def _array_channel_info(arr: np.ndarray, variables: Iterable[str]):
#     """
#     Interpret arr as either single-sample (C, H, W, ...) or batch (B, C, H, W, ...).
#     Returns: arr_cf (B, C, ...), had_batch (bool), variables_list
#     """
#     if not _is_numpy_array(arr):
#         raise TypeError("arr must be numpy.ndarray")
#     variables_list = list(variables)
#     C = len(variables_list)

#     # Case 1: batch (B, C, ...)
#     if arr.ndim >= 3 and arr.shape[1] == C:
#         return arr, True, variables_list
#     # Case 2: single sample (C, ...)
#     if arr.ndim >= 2 and arr.shape[0] == C:
#         return arr[np.newaxis, ...], False, variables_list

#     # Last-resort heuristics: if arr shape first dim equals C for higher dims
#     if arr.shape[0] == C:
#         return arr[np.newaxis, ...], False, variables_list

#     raise ValueError(f"Could not interpret numpy array shape {arr.shape} as channel-first for {C} variables")

# def _stack_param_dict_to_array(param_dict: Dict[str, Any], variables: List[str]) -> np.ndarray:
#     elems = []
#     for v in variables:
#         p = param_dict[v]
#         elems.append(np.asarray(p))
#     return np.stack(elems, axis=0)

# def _param_broadcast_for_arr(param_stack: np.ndarray, arr_cf: np.ndarray) -> np.ndarray:
#     """
#     Given param_stack shape (C, ...spatial...) or (C,), broadcast to arr_cf shape (B,C, ...spatial...).
#     """
#     spatial_ndim = arr_cf.ndim - 2  # dims after B and C
#     C = param_stack.shape[0]

#     if param_stack.ndim == 1:
#         target_shape = (1, C) + (1,) * spatial_ndim
#         reshaped = param_stack.reshape(target_shape)
#         return np.broadcast_to(reshaped, arr_cf.shape)
#     else:
#         # param_stack shape (C, d1, d2, ...)
#         if param_stack.ndim - 1 != spatial_ndim:
#             # try to align by adding trailing singleton dims
#             pad = spatial_ndim - (param_stack.ndim - 1)
#             if pad < 0:
#                 raise ValueError("Parameter spatial dims bigger than array spatial dims")
#             reshaped = param_stack.reshape((1,) + param_stack.shape + (1,) * pad)
#             return np.broadcast_to(reshaped, arr_cf.shape)
#         reshaped = param_stack.reshape((1,) + param_stack.shape)
#         return np.broadcast_to(reshaped, arr_cf.shape)


# #------------------------
# # find and build transforms (unchanged)
# #------------------------
# def _build_transform(filename, variables, active_dataset_name, model_src_dataset_name, transform_keys, builder):
#     logging.info(" >> >> INSIDE transforms_np _build transform: Fitting transform ...")
#     xfm = builder(variables, transform_keys)
#     model_src_ds = open_zarr(model_src_dataset_name, filename)
#     active_ds = open_zarr(active_dataset_name, filename)
#     try:
#         model_src_np = _ensure_numpy_dict(model_src_ds, variables)
#         active_np = _ensure_numpy_dict(active_ds, variables)
#         xfm.fit(active_np, model_src_np)
#     finally:
#         _close_dataset_if_possible(model_src_ds)
#         _close_dataset_if_possible(active_ds)
#         del model_src_np, active_np, model_src_ds, active_ds
#         gc.collect()
#     return xfm

# def _find_or_create_transforms(
#     filename,
#     active_dataset_name,
#     model_src_dataset_name,
#     transform_dir,
#     input_transform_key,
#     target_transform_key,
#     evaluation,
# ):
#     from datetime import timedelta
#     variables, target_variables = get_variables(model_src_dataset_name)

#     if transform_dir is None:
#         input_transform = _build_transform(
#             filename,
#             variables,
#             active_dataset_name,
#             model_src_dataset_name,
#             input_transform_key,
#             build_input_transform,
#         )
#         if evaluation:
#             raise RuntimeError("Target transform should only be fitted during training")
#         target_transform = _build_transform(
#             filename,
#             target_variables,
#             active_dataset_name,
#             model_src_dataset_name,
#             target_transform_key,
#             build_target_transform,
#         )
#     else:
#         dataset_transform_dir = os.path.join(transform_dir, active_dataset_name, input_transform_key+'-'+target_transform_key)
#         os.makedirs(dataset_transform_dir, exist_ok=True)
#         input_transform_path = os.path.join(dataset_transform_dir, "input.pickle")
#         target_transform_path = os.path.join(dataset_transform_dir, "target.pickle")
#         lock_path = os.path.join(transform_dir, ".lock")
#         from filelock import FileLock
#         lock = FileLock(lock_path, timeout=3600)
#         with lock:
#             if os.path.exists(input_transform_path):
#                 input_transform = load_transform(input_transform_path)
#             else:
#                 input_transform = _build_transform(
#                     filename,
#                     variables,
#                     active_dataset_name,
#                     model_src_dataset_name,
#                     input_transform_key,
#                     build_input_transform,
#                 )
#                 save_transform(input_transform, input_transform_path)
#             if os.path.exists(target_transform_path):
#                 target_transform = load_transform(target_transform_path)
#             else:
#                 if evaluation:
#                     raise RuntimeError("Target transform should only be fitted during training")
#                 target_transform = _build_transform(
#                     filename,
#                     target_variables,
#                     active_dataset_name,
#                     model_src_dataset_name,
#                     target_transform_key,
#                     build_target_transform,
#                 )
#                 save_transform(target_transform, target_transform_path)

#     gc.collect()
#     return input_transform, target_transform

# # -----------------------
# # registration utilities
# # -----------------------
# _XFMS = {}

# def register_transform(cls=None, *, name=None):
#     def _register(cls):
#         local_name = cls.__name__ if name is None else name
#         if local_name in _XFMS:
#             raise ValueError(f"Already registered transform with name: {local_name}")
#         _XFMS[local_name] = cls
#         return cls
#     if cls is None:
#         return _register
#     return _register(cls)

# def get_transform(name: str):
#     return _XFMS[name]

# # -----------------------
# # helpers: convert datasets to numpy dicts and axis mapping
# # -----------------------
# def _ensure_numpy_dict(ds: Any, variables: Iterable[str] = None) -> Dict[str, np.ndarray]:
#     out = {}
#     if isinstance(ds, dict):
#         for k, v in ds.items():
#             if variables is None or k in set(variables):
#                 out[k] = np.asarray(v)
#         return out
#     vars_to_read = variables
#     if vars_to_read is None:
#         try:
#             vars_to_read = list(ds.data_vars)
#         except Exception:
#             try:
#                 vars_to_read = list(ds.keys())
#             except Exception:
#                 raise RuntimeError("Could not determine variable names from dataset. Provide 'variables' list.")
#     for var in vars_to_read:
#         val = None
#         try:
#             val = ds[var].values
#         except Exception:
#             try:
#                 val = np.asarray(ds[var])
#             except Exception:
#                 raise RuntimeError(f"Cannot convert variable {var} to numpy array.")
#         out[var] = np.asarray(val)
#     return out

# def _close_dataset_if_possible(ds: Any):
#     try:
#         if hasattr(ds, "close"):
#             ds.close()
#     except Exception:
#         pass

# def _dim_index_map_for_ndim(ndim: int) -> Dict[str, int]:
#     if ndim == 4:
#         return {"time": 0, "ensemble": 1, "ensemble_member": 1, "lat": 2, "latitude": 2, "lon": 3, "longitude": 3}
#     if ndim == 3:
#         return {"time": 0, "lat": 1, "latitude": 1, "lon": 2, "longitude": 2}
#     if ndim == 2:
#         return {"lat": 0, "latitude": 0, "lon": 1, "longitude": 1}
#     return {"time": 0}

# def _axes_for_dims(arr: np.ndarray, dims: Iterable[str]) -> List[int]:
#     mapping = _dim_index_map_for_ndim(arr.ndim)
#     axes = []
#     for d in dims:
#         if d in mapping:
#             axes.append(mapping[d])
#     axes = sorted(list(set(axes)))
#     return axes

# def _maybe_reduce(arr: np.ndarray, dims: Iterable[str]):
#     axes = _axes_for_dims(arr, dims)
#     if len(axes) == 0:
#         return arr
#     axes_tuple = tuple(axes) if len(axes) > 1 else axes[0]
#     return axes_tuple, axes

# # -----------------------
# # Transform classes (NumPy-based)
# # -----------------------
# class CropT:
#     def __init__(self, size: int):
#         self.size = size

#     def fit(self, target_ds, model_src_ds):
#         return self

#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             out = {}
#             for var, a in _ensure_numpy_dict(arr).items():
#                 if a.ndim >= 2:
#                     out[var] = a[..., : self.size, : self.size]
#                 else:
#                     out[var] = a
#             return out
#         # ndarray path
#         arr_cf, had_batch, _ = _array_channel_info(arr, ["dummy"])  # variable count only for detection; not used
#         out = arr_cf[..., : self.size, : self.size]
#         return out if had_batch else out[0]

# @register_transform(name="stan")
# class Standardize:
#     def __init__(self, variables: Iterable[str]):
#         self.variables = list(variables)

#     def fit(self, target_ds, model_src_ds):
#         t = _ensure_numpy_dict(target_ds, self.variables)
#         self.means = {v: np.mean(t[v]) for v in self.variables}
#         self.stds = {v: np.std(t[v]) for v in self.variables}
#         for v in self.variables:
#             if self.stds[v] == 0:
#                 self.stds[v] = 1.0
#         return self

#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {}
#             for v in self.variables:
#                 a = dsn[v]
#                 out[v] = (a - self.means[v]) / self.stds[v]
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         means_stack = np.array([self.means[v] for v in self.variables])
#         stds_stack = np.array([self.stds[v] for v in self.variables])
#         means_b = _param_broadcast_for_arr(means_stack, arr_cf)
#         stds_b = _param_broadcast_for_arr(stds_stack, arr_cf)
#         out = (arr_cf - means_b) / stds_b
#         return out if had_batch else out[0]

#     def invert(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {}
#             for v in self.variables:
#                 out[v] = dsn[v] * self.stds[v] + self.means[v]
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         means_stack = np.array([self.means[v] for v in self.variables])
#         stds_stack = np.array([self.stds[v] for v in self.variables])
#         means_b = _param_broadcast_for_arr(means_stack, arr_cf)
#         stds_b = _param_broadcast_for_arr(stds_stack, arr_cf)
#         out = arr_cf * stds_b + means_b
#         return out if had_batch else out[0]

# @register_transform(name="pixelstan")
# class PixelStandardize:
#     def __init__(self, variables: Iterable[str]):
#         self.variables = list(variables)

#     def fit(self, target_ds, model_src_ds):
#         t = _ensure_numpy_dict(target_ds, self.variables)
#         self.means = {}
#         self.stds = {}
#         for v in self.variables:
#             a = t[v]
#             axes = _axes_for_dims(a, ["time"])
#             if len(axes) == 0:
#                 self.means[v] = np.asarray(a)
#                 self.stds[v] = np.asarray(a)
#             else:
#                 axes_tuple = tuple(axes) if len(axes) > 1 else axes[0]
#                 self.means[v] = np.mean(a, axis=axes_tuple)
#                 self.stds[v] = np.std(a, axis=axes_tuple)
#                 self.stds[v] = np.where(self.stds[v] == 0, 1.0, self.stds[v])
#         return self

#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {}
#             for v in self.variables:
#                 a = dsn[v]
#                 out[v] = (a - self.means[v]) / self.stds[v]
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         means_stack = _stack_param_dict_to_array(self.means, self.variables)
#         stds_stack = _stack_param_dict_to_array(self.stds, self.variables)
#         means_b = _param_broadcast_for_arr(means_stack, arr_cf)
#         stds_b = _param_broadcast_for_arr(stds_stack, arr_cf)
#         out = (arr_cf - means_b) / stds_b
#         return out if had_batch else out[0]

# @register_transform(name="noop")
# class NoopT:
#     def fit(self, target_ds, model_src_ds):
#         return self

#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             return _ensure_numpy_dict(arr)
#         return arr

#     def invert(self, arr, times=None):
#         if isinstance(arr, dict):
#             return _ensure_numpy_dict(arr)
#         return arr

# @register_transform(name="pixelmmsstan")
# class PixelMatchModelSrcStandardize:
#     def __init__(self, variables: Iterable[str]):
#         self.variables = list(variables)

#     def fit(self, target_ds, model_src_ds):
#         t = _ensure_numpy_dict(target_ds, self.variables)
#         m = _ensure_numpy_dict(model_src_ds, self.variables)
#         self.pixel_target_means = {}
#         self.pixel_target_stds = {}
#         self.pixel_model_src_means = {}
#         self.pixel_model_src_stds = {}
#         self.global_model_src_means = {}
#         self.global_model_src_stds = {}
#         for v in self.variables:
#             arr_t = t[v]
#             arr_m = m[v]
#             axes_t = _axes_for_dims(arr_t, ["time", "ensemble", "ensemble_member"])
#             axes_m = _axes_for_dims(arr_m, ["time", "ensemble", "ensemble_member"])
#             if len(axes_t) == 0:
#                 self.pixel_target_means[v] = arr_t
#                 self.pixel_target_stds[v] = arr_t
#             else:
#                 axes_tuple_t = tuple(axes_t) if len(axes_t) > 1 else axes_t[0]
#                 self.pixel_target_means[v] = np.mean(arr_t, axis=axes_tuple_t)
#                 self.pixel_target_stds[v] = np.std(arr_t, axis=axes_tuple_t)
#                 self.pixel_target_stds[v] = np.where(self.pixel_target_stds[v] == 0, 1.0, self.pixel_target_stds[v])
#             if len(axes_m) == 0:
#                 self.pixel_model_src_means[v] = arr_m
#                 self.pixel_model_src_stds[v] = arr_m
#             else:
#                 axes_tuple_m = tuple(axes_m) if len(axes_m) > 1 else axes_m[0]
#                 self.pixel_model_src_means[v] = np.mean(arr_m, axis=axes_tuple_m)
#                 self.pixel_model_src_stds[v] = np.std(arr_m, axis=axes_tuple_m)
#                 self.pixel_model_src_stds[v] = np.where(self.pixel_model_src_stds[v] == 0, 1.0, self.pixel_model_src_stds[v])
#             self.global_model_src_means[v] = np.mean(arr_m)
#             self.global_model_src_stds[v] = np.std(arr_m)
#             if self.global_model_src_stds[v] == 0:
#                 self.global_model_src_stds[v] = 1.0
#         return self

#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {}
#             for v in self.variables:
#                 a = dsn[v]
#                 da_pixel_stan = (a - self.pixel_target_means[v]) / self.pixel_target_stds[v]
#                 da_pixel_like_model_src = da_pixel_stan * self.pixel_model_src_stds[v] + self.pixel_model_src_means[v]
#                 da_global_stan_like_model_src = (da_pixel_like_model_src - self.global_model_src_means[v]) / self.global_model_src_stds[v]
#                 out[v] = da_global_stan_like_model_src
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         ptm = _stack_param_dict_to_array(self.pixel_target_means, self.variables)
#         pts = _stack_param_dict_to_array(self.pixel_target_stds, self.variables)
#         pmm = _stack_param_dict_to_array(self.pixel_model_src_means, self.variables)
#         pms = _stack_param_dict_to_array(self.pixel_model_src_stds, self.variables)
#         gmean = np.array([self.global_model_src_means[v] for v in self.variables])
#         gstd = np.array([self.global_model_src_stds[v] for v in self.variables])
#         ptm_b = _param_broadcast_for_arr(ptm, arr_cf)
#         pts_b = _param_broadcast_for_arr(pts, arr_cf)
#         pmm_b = _param_broadcast_for_arr(pmm, arr_cf)
#         pms_b = _param_broadcast_for_arr(pms, arr_cf)
#         gmean_b = _param_broadcast_for_arr(gmean, arr_cf)
#         gstd_b = _param_broadcast_for_arr(gstd, arr_cf)
#         da_pixel_stan = (arr_cf - ptm_b) / pts_b
#         da_pixel_like_model_src = da_pixel_stan * pms_b + pmm_b
#         da_global_stan_like_model_src = (da_pixel_like_model_src - gmean_b) / gstd_b
#         out = da_global_stan_like_model_src
#         return out if had_batch else out[0]

# @register_transform(name="mm")
# class MinMax:
#     def __init__(self, variables: Iterable[str]):
#         self.variables = list(variables)
#     def fit(self, target_ds, model_src_ds):
#         t = _ensure_numpy_dict(target_ds, self.variables)
#         self.maxs = {v: np.max(t[v]) for v in self.variables}
#         self.mins = {v: np.min(t[v]) for v in self.variables}
#         for v in self.variables:
#             if self.maxs[v] == self.mins[v]:
#                 self.maxs[v] = self.mins[v] + 1.0
#         return self
#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {}
#             for v in self.variables:
#                 out[v] = (dsn[v] - self.mins[v]) / (self.maxs[v] - self.mins[v])
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         maxs_stack = np.array([self.maxs[v] for v in self.variables])
#         mins_stack = np.array([self.mins[v] for v in self.variables])
#         maxs_b = _param_broadcast_for_arr(maxs_stack, arr_cf)
#         mins_b = _param_broadcast_for_arr(mins_stack, arr_cf)
#         out = (arr_cf - mins_b) / (maxs_b - mins_b)
#         return out if had_batch else out[0]
#     def invert(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {}
#             for v in self.variables:
#                 out[v] = dsn[v] * (self.maxs[v] - self.mins[v]) + self.mins[v]
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         maxs_stack = np.array([self.maxs[v] for v in self.variables])
#         mins_stack = np.array([self.mins[v] for v in self.variables])
#         maxs_b = _param_broadcast_for_arr(maxs_stack, arr_cf)
#         mins_b = _param_broadcast_for_arr(mins_stack, arr_cf)
#         out = arr_cf * (maxs_b - mins_b) + mins_b
#         return out if had_batch else out[0]

# @register_transform(name="ur")
# class UnitRangeT:
#     def __init__(self, variables: Iterable[str]):
#         self.variables = list(variables)
#     def fit(self, target_ds, model_src_ds):
#         t = _ensure_numpy_dict(target_ds, self.variables)
#         self.maxs = {v: np.max(t[v]) for v in self.variables}
#         for v in self.variables:
#             if self.maxs[v] == 0:
#                 self.maxs[v] = 1.0
#         return self
#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {}
#             for v in self.variables:
#                 out[v] = dsn[v] / self.maxs[v]
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         maxs_stack = np.array([self.maxs[v] for v in self.variables])
#         maxs_b = _param_broadcast_for_arr(maxs_stack, arr_cf)
#         out = arr_cf / maxs_b
#         return out if had_batch else out[0]
#     def invert(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {}
#             for v in self.variables:
#                 out[v] = dsn[v] * self.maxs[v]
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         maxs_stack = np.array([self.maxs[v] for v in self.variables])
#         maxs_b = _param_broadcast_for_arr(maxs_stack, arr_cf)
#         out = arr_cf * maxs_b
#         return out if had_batch else out[0]

# @register_transform(name="clip")
# class ClipT:
#     def __init__(self, variables: Iterable[str]):
#         self.variables = list(variables)
#     def fit(self, target_ds, model_src_ds):
#         return self
#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             return _ensure_numpy_dict(arr)
#         return arr
#     def invert(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {}
#             for v in self.variables:
#                 min_val = 0.0
#                 nclipped = int(np.sum(dsn[v] < min_val))
#                 logger.debug(f"Clipping {v} to {min_val}: {nclipped}")
#                 out[v] = np.clip(dsn[v], a_min=min_val, a_max=None)
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         out = np.clip(arr_cf, a_min=0.0, a_max=None)
#         return out if had_batch else out[0]

# @register_transform(name="pc")
# class PercentToPropT:
#     def __init__(self, variables: Iterable[str]):
#         self.variables = list(variables)
#     def fit(self, target_ds, _model_src_ds):
#         return self
#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: (dsn[v] / 100.0) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         out = arr_cf / 100.0
#         return out if had_batch else out[0]
#     def invert(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: (dsn[v] * 100.0) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         out = arr_cf * 100.0
#         return out if had_batch else out[0]

# @register_transform(name="recen")
# class RecentreT:
#     def __init__(self, variables: Iterable[str]):
#         self.variables = list(variables)
#     def fit(self, target_ds, model_src_ds):
#         return self
#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: (dsn[v] * 2.0 - 1.0) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         out = arr_cf * 2.0 - 1.0
#         return out if had_batch else out[0]
#     def invert(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: ((dsn[v] + 1.0) / 2.0) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         out = (arr_cf + 1.0) / 2.0
#         return out if had_batch else out[0]

# @register_transform(name="sqrt")
# class SqrtT:
#     def __init__(self, variables: Iterable[str]):
#         self.variables = list(variables)
#     def fit(self, target_ds, model_src_ds):
#         return self
#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: np.power(dsn[v], 0.5) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         out = np.power(arr_cf, 0.5)
#         return out if had_batch else out[0]
#     def invert(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: np.power(dsn[v], 2.0) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         out = np.power(arr_cf, 2.0)
#         return out if had_batch else out[0]

# @register_transform(name="root")
# class RootT:
#     def __init__(self, variables: Iterable[str], root_base: float):
#         self.variables = list(variables)
#         self.root_base = root_base
#     def fit(self, target_ds, model_src_ds):
#         return self
#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: np.power(dsn[v], 1.0 / self.root_base) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         out = np.power(arr_cf, 1.0 / self.root_base)
#         return out if had_batch else out[0]
#     def invert(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: np.power(dsn[v], self.root_base) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         out = np.power(arr_cf, self.root_base)
#         return out if had_batch else out[0]

# @register_transform(name="rm")
# class RawMomentT:
#     def __init__(self, variables: Iterable[str], root_base: float):
#         self.variables = list(variables)
#         self.root_base = root_base
#     def fit(self, target_ds, model_src_ds):
#         t = _ensure_numpy_dict(target_ds, self.variables)
#         self.raw_moments = {
#             var: np.power(np.mean(np.power(t[var], self.root_base)), 1.0 / self.root_base)
#             for var in self.variables
#         }
#         for v in self.variables:
#             if self.raw_moments[v] == 0:
#                 self.raw_moments[v] = 1.0
#         return self
#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: (dsn[v] / self.raw_moments[v]) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         rm = np.array([self.raw_moments[v] for v in self.variables])
#         rm_b = _param_broadcast_for_arr(rm, arr_cf)
#         out = arr_cf / rm_b
#         return out if had_batch else out[0]
#     def invert(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: (dsn[v] * self.raw_moments[v]) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         rm = np.array([self.raw_moments[v] for v in self.variables])
#         rm_b = _param_broadcast_for_arr(rm, arr_cf)
#         out = arr_cf * rm_b
#         return out if had_batch else out[0]

# @register_transform(name="log")
# class LogT:
#     def __init__(self, variables: Iterable[str]):
#         self.variables = list(variables)
#     def fit(self, target_ds, model_src_ds):
#         return self
#     def transform(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: np.log1p(dsn[v]) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         out = np.log1p(arr_cf)
#         return out if had_batch else out[0]
#     def invert(self, arr, times=None):
#         if isinstance(arr, dict):
#             dsn = _ensure_numpy_dict(arr, self.variables)
#             out = {v: np.expm1(dsn[v]) for v in self.variables}
#             return {**dsn, **out}
#         arr_cf, had_batch, _ = _array_channel_info(arr, self.variables)
#         out = np.expm1(arr_cf)
#         return out if had_batch else out[0]

# @register_transform(name="compose")
# class ComposeT:
#     def __init__(self, transforms: Iterable[Any]):
#         self.transforms = list(transforms)
#     def fit(self, target_ds, model_src_ds=None):
#         current_target = target_ds
#         for t in self.transforms:
#             t.fit(current_target, model_src_ds)
#             current_target = t.transform(current_target)
#         return self
#     def transform(self, arr, times=None):
#         current = arr
#         for t in self.transforms:
#             try:
#                 current = t.transform(current, times=times)
#             except TypeError:
#                 current = t.transform(current)
#         return current
#     def invert(self, arr, times=None):
#         current = arr
#         for t in reversed(self.transforms):
#             if hasattr(t, "invert"):
#                 try:
#                     current = t.invert(current, times=times)
#                 except TypeError:
#                     current = t.invert(current)
#             else:
#                 raise RuntimeError(f"Transform {t} has no invert method")
#         return current

# # -----------------------
# # builders (same API)
# # -----------------------
# def build_input_transform(variables, key="v1"):
#     if key == "v1":
#         return ComposeT([Standardize(variables), UnitRangeT(variables)])
#     if key == "none":
#         return NoopT()
#     if key in ["standardize", "stan"]:
#         return ComposeT([Standardize(variables)])
#     if key == "stanur":
#         return ComposeT([Standardize(variables), UnitRangeT(variables)])
#     if key == "stanurrecen":
#         return ComposeT([Standardize(variables), UnitRangeT(variables), RecentreT(variables)])
#     if key == "pixelstan":
#         return ComposeT([PixelStandardize(variables)])
#     if key == "pixelmmsstan":
#         return ComposeT([PixelMatchModelSrcStandardize(variables)])
#     if key == "pixelmmsstanur":
#         return ComposeT([PixelMatchModelSrcStandardize(variables), UnitRangeT(variables)])
#     xfms = [get_transform(name)(variables) for name in key.split(";")]
#     return ComposeT(xfms)

# def build_target_transform(target_variables, keys):
#     return ComposeT([_build_target_transform(tvar, keys[tvar]) for tvar in target_variables])

# def _build_target_transform(target_variable, key):
#     if key == "v1":
#         return ComposeT([SqrtT([target_variable]), ClipT([target_variable]), UnitRangeT([target_variable])])
#     if key == "none":
#         return NoopT()
#     if key == "sqrt":
#         return ComposeT([RootT([target_variable], 2), ClipT([target_variable])])
#     if key == "sqrtur":
#         return ComposeT([RootT([target_variable], 2), ClipT([target_variable]), UnitRangeT([target_variable])])
#     if key == "sqrturrecen":
#         return ComposeT([RootT([target_variable], 2), ClipT([target_variable]), UnitRangeT([target_variable]), RecentreT([target_variable])])
#     if key == "sqrtrm":
#         return ComposeT([RootT([target_variable], 2), RawMomentT([target_variable], 2), ClipT([target_variable])])
#     if key == "cbrt":
#         return ComposeT([RootT([target_variable], 3), ClipT([target_variable])])
#     if key == "cbrtur":
#         return ComposeT([RootT([target_variable], 3), ClipT([target_variable]), UnitRangeT([target_variable])])
#     if key == "qdrt":
#         return ComposeT([RootT([target_variable], 4), ClipT([target_variable])])
#     if key == "log":
#         return ComposeT([LogT([target_variable]), ClipT([target_variable])])
#     if key == "logurrecen":
#         return ComposeT([ClipT([target_variable]), LogT([target_variable]), UnitRangeT([target_variable]), RecentreT([target_variable])])
#     if key == "stanurrecen":
#         return ComposeT([Standardize([target_variable]), UnitRangeT([target_variable]), RecentreT([target_variable])])
#     if key == "stanmmrecen":
#         return ComposeT([Standardize([target_variable]), MinMax([target_variable]), RecentreT([target_variable])])
#     if key == "urrecen":
#         return ComposeT([UnitRangeT([target_variable]), RecentreT([target_variable])])
#     if key == "mmrecen":
#         return ComposeT([MinMax([target_variable]), RecentreT([target_variable])])
#     if key == "pcrecen":
#         return ComposeT([PercentToPropT([target_variable]), RecentreT([target_variable])])
#     if key == "recen":
#         return ComposeT([RecentreT([target_variable])])
#     xfms = [get_transform(name)([target_variable]) for name in key.split(";")]
#     return ComposeT(xfms)