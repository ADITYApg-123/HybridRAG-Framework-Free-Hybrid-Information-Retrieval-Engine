"""Generates a comprehensive corpus of PyTorch technical HTML documentation in data/raw/."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("hybrid_rag.ingestion.corpus_builder")


def _wrap_html_doc(module: str, symbol: str, title: str, sections: List[Dict[str, str]]) -> str:
    """Wrap structured section dicts in realistic Sphinx-style PyTorch HTML markup."""
    section_html_parts: List[str] = []
    for sec in sections:
        section_html_parts.append(
            f'<section id="{sec["id"]}" data-module="{module}" data-symbol="{symbol}">\n'
            f'  <h2>{sec["heading"]}<a class="headerlink" href="#{sec["id"]}">¶</a></h2>\n'
            f'  {sec["body"]}\n'
            f'</section>'
        )

    sections_joined = "\n".join(section_html_parts)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        f"  <meta charset=\"utf-8\" />\n"
        f"  <meta name=\"module\" content=\"{module}\" />\n"
        f"  <meta name=\"symbol\" content=\"{symbol}\" />\n"
        f"  <title>{title} — PyTorch 2.6 Documentation</title>\n"
        "  <script>console.log('analytics boilerplate');</script>\n"
        "  <style>.navbar { display: flex; }</style>\n"
        "</head>\n<body>\n"
        "  <nav class=\"navbar\">PyTorch Navigation Header | Docs | Tutorials</nav>\n"
        "  <aside class=\"sidebar\">Table of Contents Sidebar</aside>\n"
        f"  <main class=\"document\">\n{sections_joined}\n  </main>\n"
        "  <footer class=\"footer\">Copyright PyTorch Contributors.</footer>\n"
        "</body>\n</html>\n"
    )


PYTORCH_HTML_CORPUS: Dict[str, Dict] = {
    "torch_nn_conv2d.html": {
        "module": "torch.nn",
        "symbol": "Conv2d",
        "title": "torch.nn.Conv2d",
        "sections": [
            {
                "id": "signature_and_parameters",
                "heading": "torch.nn.Conv2d Signature and Parameters",
                "body": (
                    "<p><code>class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, "
                    "padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')</code></p>"
                    "<p>Applies a 2D convolution over an input signal composed of several input planes. "
                    "<code>in_channels</code> specifies channels in the input image; <code>out_channels</code> "
                    "specifies channels produced by the convolution filters. <code>padding='same'</code> pads "
                    "the input so the output has the same height and width as the input, but "
                    "<code>padding='same'</code> cannot be used with any stride value other than <code>stride=1</code>. "
                    "Supported <code>padding_mode</code> values are <code>'zeros'</code>, <code>'reflect'</code>, "
                    "<code>'replicate'</code>, and <code>'circular'</code>.</p>"
                ),
            },
            {
                "id": "shape_and_depthwise_math",
                "heading": "Conv2d Output Spatial Shape Formula and Depthwise Convolution",
                "body": (
                    "<p>For an input tensor of shape <code>(N, C_in, H_in, W_in)</code>, the spatial output height "
                    "and width of <code>torch.nn.Conv2d</code> are computed using the floor equation: "
                    "<code>H_out = floor((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)</code>.</p>"
                    "<p>When <code>groups == in_channels</code> and <code>out_channels == K * in_channels</code>, "
                    "this configuration performs a <strong>depthwise separable convolution</strong> where each input "
                    "channel is convolved with its own separate set of K filters.</p>"
                    "<pre><code>depthwise = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)\n"
                    "pointwise = nn.Conv2d(32, 64, kernel_size=1)</code></pre>"
                ),
            },
        ],
    },
    "torch_nn_multihead_attention.html": {
        "module": "torch.nn",
        "symbol": "MultiheadAttention",
        "title": "torch.nn.MultiheadAttention",
        "sections": [
            {
                "id": "scaled_dot_product_overview",
                "heading": "torch.nn.MultiheadAttention Overview and Scaled Dot-Product Math",
                "body": (
                    "<p><code>class torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, "
                    "add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False)</code></p>"
                    "<p>Computes multi-head scaled dot-product attention: "
                    "<code>Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k)) @ V</code>. "
                    "<code>embed_dim</code> must be divisible by <code>num_heads</code> so each head has dimension "
                    "<code>head_dim = embed_dim // num_heads</code>; otherwise an <code>AssertionError</code> is raised. "
                    "Setting <code>batch_first=True</code> changes tensor layout from <code>(seq_len, batch, embed_dim)</code> "
                    "to <code>(batch, seq_len, embed_dim)</code>.</p>"
                ),
            },
            {
                "id": "masks_and_fastpath",
                "heading": "MultiheadAttention Mask Semantics: key_padding_mask vs attn_mask",
                "body": (
                    "<p>In <code>MultiheadAttention.forward</code>, <code>key_padding_mask</code> of shape <code>(N, S)</code> "
                    "masks padding tokens across the source sequence length S, where a boolean <code>True</code> value "
                    "indicates the corresponding key position is ignored. <code>attn_mask</code> of shape <code>(L, S)</code> "
                    "prevents attention to specific positions such as future tokens in causal autoregressive decoding.</p>"
                    "<p>When <code>need_weights=False</code>, execution dispatches to the fused "
                    "<code>torch.nn.functional.scaled_dot_product_attention</code> FlashAttention fast-path kernel.</p>"
                ),
            },
        ],
    },
}
PYTORCH_HTML_CORPUS.update({
    "torch_nn_normalization.html": {
        "module": "torch.nn",
        "symbol": "BatchNorm2d_LayerNorm",
        "title": "torch.nn.BatchNorm2d and torch.nn.LayerNorm",
        "sections": [
            {
                "id": "batchnorm2d_running_stats",
                "heading": "torch.nn.BatchNorm2d Running Statistics and Momentum",
                "body": (
                    "<p><code>class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)</code></p>"
                    "<p>Normalizes 4D inputs <code>(N, C, H, W)</code> per channel across batch and spatial dimensions with default "
                    "stability constant <code>eps=1e-05</code>. During <code>model.train()</code>, buffers <code>running_mean</code> "
                    "and <code>running_var</code> update via exponential moving average with <code>momentum=0.1</code>: "
                    "<code>running_stat_new = (1 - momentum) * running_stat_old + momentum * batch_stat_observed</code>. "
                    "During <code>model.eval()</code>, running statistics are frozen.</p>"
                ),
            },
            {
                "id": "layernorm_vs_batchnorm",
                "heading": "torch.nn.LayerNorm Per-Sample Normalization for Transformers",
                "body": (
                    "<p><code>class torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True)</code></p>"
                    "<p>Unlike BatchNorm, <code>LayerNorm</code> computes mean and variance independently per sample across the last "
                    "D dimensions given by <code>normalized_shape</code>. It maintains no running statistics, behaves identically in "
                    "<code>train()</code> and <code>eval()</code> modes, and is invariant to batch size for Transformer sequences.</p>"
                ),
            },
        ],
    },
    "torch_nn_recurrent_lstm.html": {
        "module": "torch.nn",
        "symbol": "LSTM",
        "title": "torch.nn.LSTM",
        "sections": [
            {
                "id": "lstm_architecture_and_gates",
                "heading": "torch.nn.LSTM Gate Equations and Hidden States",
                "body": (
                    "<p><code>class torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False)</code></p>"
                    "<p>Applies a multi-layer Long Short-Term Memory (LSTM) recurrent neural network. At each timestep t, the cell maintains "
                    "hidden state <code>h_t</code> and cell state <code>c_t</code> regulated by input gate <code>i_t</code>, forget gate <code>f_t</code>, "
                    "cell candidate <code>g_t</code>, and output gate <code>o_t</code>. When <code>bidirectional=True</code>, output hidden dimension doubles "
                    "to <code>2 * hidden_size</code>. Note: <code>dropout</code> applies between stacked layers except the last layer, so setting "
                    "<code>dropout &gt; 0</code> with <code>num_layers=1</code> triggers a UserWarning and has no effect.</p>"
                ),
            },
            {
                "id": "pack_padded_sequence_handling",
                "heading": "Handling Variable-Length Sequences with pack_padded_sequence",
                "body": (
                    "<p>To prevent recurrent layers like <code>nn.LSTM</code> from wasting computation over trailing padding tokens, wrap padded batches "
                    "using <code>torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)</code> before the LSTM, "
                    "and unpack outputs back to padded tensors using <code>torch.nn.utils.rnn.pad_packed_sequence</code>.</p>"
                ),
            },
        ],
    },
    "torch_nn_losses.html": {
        "module": "torch.nn",
        "symbol": "CrossEntropyLoss_BCEWithLogitsLoss",
        "title": "torch.nn.CrossEntropyLoss and BCEWithLogitsLoss",
        "sections": [
            {
                "id": "crossentropyloss_logits",
                "heading": "torch.nn.CrossEntropyLoss Unnormalized Logits and ignore_index",
                "body": (
                    "<p><code>class torch.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)</code></p>"
                    "<p>Combines <code>nn.LogSoftmax</code> and <code>nn.NLLLoss</code> in a single numerically stable log-sum-exp pass. "
                    "Do not apply Softmax before passing inputs to <code>CrossEntropyLoss</code>; pass raw unnormalized logits directly. "
                    "The parameter <code>ignore_index=-100</code> masks out padding target tokens so they do not contribute to the gradient.</p>"
                ),
            },
            {
                "id": "bcewithlogitsloss_pos_weight",
                "heading": "torch.nn.BCEWithLogitsLoss Numerical Stability and Class Imbalance",
                "body": (
                    "<p><code>class torch.nn.BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None)</code></p>"
                    "<p>Fuses a <code>Sigmoid</code> layer and binary cross entropy (<code>BCELoss</code>) using the log-sum-exp trick for numerical stability. "
                    "To handle positive/negative class imbalance in binary classification, set <code>pos_weight</code> to "
                    "<code>num_negative_samples / num_positive_samples</code>.</p>"
                ),
            },
        ],
    },
})
PYTORCH_HTML_CORPUS.update({
    "torch_optim_adamw.html": {
        "module": "torch.optim",
        "symbol": "AdamW",
        "title": "torch.optim.AdamW and Decoupled Weight Decay",
        "sections": [
            {
                "id": "decoupled_weight_decay_math",
                "heading": "torch.optim.AdamW Decoupled Weight Decay vs L2 Regularization",
                "body": (
                    "<p><code>class torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, fused=None)</code></p>"
                    "<p>Implements AdamW decoupled weight decay regularization. Unlike standard <code>torch.optim.Adam</code> where L2 penalty is added to the "
                    "gradient before computing adaptive moments <code>m_t</code> and <code>v_t</code>, <code>AdamW</code> subtracts "
                    "<code>lr * weight_decay * theta_t</code> directly from weights simultaneously with the moment step. "
                    "Setting <code>fused=True</code> on CUDA launches a single fused multi-tensor kernel update.</p>"
                ),
            },
            {
                "id": "parameter_groups_no_decay",
                "heading": "Excluding Bias and LayerNorm Parameters from Weight Decay",
                "body": (
                    "<p>When training Transformers with <code>AdamW</code>, 1D parameters such as bias vectors and <code>LayerNorm</code> / <code>BatchNorm</code> "
                    "affine scale weights should be excluded from weight decay (<code>weight_decay=0.0</code>) using optimizer parameter group dicts.</p>"
                    "<pre><code>optim_groups = [\n"
                    "    {'params': [p for p in model.parameters() if p.dim() >= 2], 'weight_decay': 0.01},\n"
                    "    {'params': [p for p in model.parameters() if p.dim() < 2], 'weight_decay': 0.0},\n"
                    "]\noptimizer = torch.optim.AdamW(optim_groups, lr=3e-4, fused=True)</code></pre>"
                ),
            },
        ],
    },
    "torch_optim_lr_schedulers.html": {
        "module": "torch.optim.lr_scheduler",
        "symbol": "OneCycleLR_CosineAnnealingLR",
        "title": "Learning Rate Schedulers: OneCycleLR and CosineAnnealingWarmRestarts",
        "sections": [
            {
                "id": "onecyclelr_super_convergence",
                "heading": "torch.optim.lr_scheduler.OneCycleLR Super-Convergence Schedule",
                "body": (
                    "<p><code>class torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=None, pct_start=0.3, anneal_strategy='cos', div_factor=25.0, final_div_factor=10000.0)</code></p>"
                    "<p>Warms up the learning rate from <code>initial_lr = max_lr / div_factor</code> to <code>max_lr</code> over the first <code>pct_start</code> "
                    "(default 0.3) fraction of total steps, then anneals down to <code>initial_lr / final_div_factor</code>. "
                    "Unlike epoch-level schedulers, <code>OneCycleLR.step()</code> must be called after every single training batch iteration.</p>"
                ),
            },
            {
                "id": "cosine_annealing_warm_restarts",
                "heading": "CosineAnnealingWarmRestarts (SGDR) Periodic Restarts",
                "body": (
                    "<p><code>class torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0)</code></p>"
                    "<p>Implements Stochastic Gradient Descent with Warm Restarts (SGDR), decaying the learning rate along a cosine curve down to "
                    "<code>eta_min</code> over <code>T_0</code> iterations before restarting at the maximum learning rate and multiplying the period by <code>T_mult</code>.</p>"
                ),
            },
        ],
    },
    "torch_utils_data_dataloader.html": {
        "module": "torch.utils.data",
        "symbol": "DataLoader",
        "title": "torch.utils.data.DataLoader",
        "sections": [
            {
                "id": "multiprocess_workers_and_memory",
                "heading": "DataLoader Multiprocessing, pin_memory, and persistent_workers",
                "body": (
                    "<p><code>class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, prefetch_factor=None, persistent_workers=False)</code></p>"
                    "<p>Setting <code>pin_memory=True</code> allocates fetched host tensors in page-locked (pinned) RAM for fast asynchronous DMA transfers to "
                    "CUDA GPUs via <code>tensor.to(device, non_blocking=True)</code>. Setting <code>persistent_workers=True</code> keeps worker subprocesses "
                    "alive across epochs instead of shutting down and respawning workers after every epoch. Default <code>prefetch_factor</code> is 2 when <code>num_workers &gt; 0</code>.</p>"
                ),
            },
            {
                "id": "custom_collate_fn_and_samplers",
                "heading": "Custom collate_fn and DistributedSampler Usage",
                "body": (
                    "<p>The <code>collate_fn</code> callable merges a list of dataset samples into a mini-batch and handles variable-length sequence padding. "
                    "In multi-GPU training with <code>DistributedSampler</code>, set <code>shuffle=False</code> on the DataLoader and call "
                    "<code>sampler.set_epoch(epoch)</code> at the start of every epoch to guarantee deterministic random shuffling across epochs.</p>"
                ),
            },
        ],
    },
})
PYTORCH_HTML_CORPUS.update({
    "torch_autograd_mechanics.html": {
        "module": "torch.autograd",
        "symbol": "Autograd_GradMode",
        "title": "torch.autograd Mechanics: inference_mode, no_grad, and Checkpointing",
        "sections": [
            {
                "id": "inference_mode_vs_no_grad",
                "heading": "torch.inference_mode vs torch.no_grad Context Managers",
                "body": (
                    "<p>Both <code>torch.no_grad()</code> and <code>torch.inference_mode()</code> disable gradient tracking during forward evaluation. "
                    "However, <code>torch.inference_mode()</code> is faster because it additionally disables <strong>view tracking and version counter bumps</strong> "
                    "on tensors. Tensors created inside an <code>inference_mode</code> block cannot be modified in-place outside the block or used in autograd backward passes.</p>"
                ),
            },
            {
                "id": "detach_and_backward_hooks",
                "heading": "tensor.detach(), register_full_backward_hook, and Activation Checkpointing",
                "body": (
                    "<p><code>tensor.detach()</code> returns a new tensor view sharing storage with the original tensor but detached from the autograd graph "
                    "(<code>requires_grad=False</code>). To inspect gradients flowing through a module during backpropagation, use "
                    "<code>module.register_full_backward_hook(hook_fn)</code>. To reduce GPU activation memory by recomputing forward activations during backward, "
                    "use <code>torch.utils.checkpoint.checkpoint(block, *inputs, use_reentrant=False)</code>.</p>"
                ),
            },
        ],
    },
    "torch_nn_grad_clipping.html": {
        "module": "torch.nn.utils",
        "symbol": "clip_grad_norm_",
        "title": "Gradient Clipping: clip_grad_norm_ and clip_grad_value_",
        "sections": [
            {
                "id": "clip_grad_norm_mechanics",
                "heading": "torch.nn.utils.clip_grad_norm_ for Preventing Exploding Gradients",
                "body": (
                    "<p><code>torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False)</code></p>"
                    "<p>Clips the total gradient norm of an iterable of parameters in-place to mitigate exploding gradients in RNNs, LSTMs, and Transformers. "
                    "If the total gradient norm exceeds <code>max_norm</code>, all gradients are scaled uniformly by <code>max_norm / (total_norm + 1e-6)</code>. "
                    "Call <code>clip_grad_norm_</code> after <code>loss.backward()</code> (and after <code>scaler.unscale_(optimizer)</code>) and before <code>optimizer.step()</code>.</p>"
                ),
            },
        ],
    },
})
PYTORCH_HTML_CORPUS.update({
    "torch_amp_mixed_precision.html": {
        "module": "torch.amp",
        "symbol": "autocast_GradScaler",
        "title": "Automatic Mixed Precision (AMP): torch.amp.autocast and GradScaler",
        "sections": [
            {
                "id": "autocast_fp16_bf16",
                "heading": "torch.amp.autocast Precision Casting Rules (FP16 vs BF16)",
                "body": (
                    "<p><code>class torch.amp.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)</code></p>"
                    "<p>Runs eligible operations (<code>matmul</code>, <code>conv2d</code>, <code>linear</code>, <code>bmm</code>) in <code>float16</code> or "
                    "<code>bfloat16</code> to reduce GPU VRAM usage and accelerate Tensor Core execution, while keeping reductions and normalization "
                    "(<code>softmax</code>, <code>LayerNorm</code>, <code>cross_entropy</code>) in <code>float32</code>. Only wrap the forward pass and loss calculation "
                    "inside <code>with torch.amp.autocast('cuda'):</code>; never wrap <code>loss.backward()</code> inside <code>autocast</code>.</p>"
                ),
            },
            {
                "id": "gradscaler_underflow_prevention",
                "heading": "torch.amp.GradScaler Dynamic Loss Scaling and unscale_",
                "body": (
                    "<p><code>class torch.amp.GradScaler(device='cuda', init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)</code></p>"
                    "<p>Prevents small FP16 gradients from underflowing to zero by multiplying the loss by a dynamic scale factor before <code>backward()</code>. "
                    "Before calling <code>clip_grad_norm_</code>, you must call <code>scaler.unscale_(optimizer)</code> so gradients are clipped at true unscaled magnitudes. "
                    "If <code>Inf</code> or <code>NaN</code> gradients appear during <code>scaler.step(optimizer)</code>, the optimizer update is skipped and the scale multiplies by <code>backoff_factor</code>.</p>"
                ),
            },
        ],
    },
    "torch_distributed_ddp.html": {
        "module": "torch.distributed",
        "symbol": "DistributedDataParallel",
        "title": "torch.nn.parallel.DistributedDataParallel (DDP)",
        "sections": [
            {
                "id": "ddp_vs_dataparallel",
                "heading": "DistributedDataParallel Architecture and All-Reduce Gradient Bucketing",
                "body": (
                    "<p><code>class torch.nn.parallel.DistributedDataParallel(module, device_ids=None, broadcast_buffers=True, find_unused_parameters=False, gradient_as_bucket_view=False, static_graph=False)</code></p>"
                    "<p>Implements multi-process data parallelism and is strongly preferred over single-process multi-threaded <code>torch.nn.DataParallel</code> "
                    "(which suffers from Python GIL contention). DDP spawns one process per GPU via <code>torchrun</code>, initializes with "
                    "<code>torch.distributed.init_process_group(backend='nccl')</code>, and overlaps backward pass computation with ring All-Reduce communication "
                    "across gradient buckets (<code>bucket_cap_mb=25</code>). Set <code>TORCH_NCCL_ASYNC_ERROR_HANDLING=1</code> to abort hung NCCL collectives.</p>"
                ),
            },
            {
                "id": "ddp_no_sync_gradient_accumulation",
                "heading": "Accelerating Gradient Accumulation in DDP with model.no_sync()",
                "body": (
                    "<p>When accumulating gradients over K micro-batches in <code>DistributedDataParallel</code>, wrap the first K-1 micro-batch forward and backward passes "
                    "inside <code>with ddp_model.no_sync():</code> to suppress redundant cross-GPU All-Reduce communications until the final synchronizing step.</p>"
                ),
            },
        ],
    },
})
PYTORCH_HTML_CORPUS.update({
    "torch_compile_dynamo.html": {
        "module": "torch",
        "symbol": "compile",
        "title": "torch.compile, TorchDynamo, and TorchInductor",
        "sections": [
            {
                "id": "torch_compile_modes_and_backends",
                "heading": "torch.compile Compilation Pipeline and Optimization Modes",
                "body": (
                    "<p><code>torch.compile(model=None, *, fullgraph=False, dynamic=None, backend='inductor', mode=None)</code></p>"
                    "<p>Captures Python bytecode into FX graphs via <strong>TorchDynamo</strong>, traces backward passes with <strong>AOTAutograd</strong>, "
                    "and generates fused Triton (GPU) or C++/OpenMP (CPU) kernels via <strong>TorchInductor</strong>. "
                    "<code>mode='reduce-overhead'</code> uses CUDA Graphs to minimize launch overhead for small batches; "
                    "<code>mode='max-autotune'</code> benchmarks Triton matrix multiplication templates and CUDA graphs for peak throughput.</p>"
                ),
            },
            {
                "id": "graph_breaks_and_fullgraph",
                "heading": "Graph Breaks, Data-Dependent Control Flow, and fullgraph=True",
                "body": (
                    "<p>When TorchDynamo encounters data-dependent control flow (<code>if tensor.sum() &gt; 0:</code>) or <code>.item()</code> calls, "
                    "it inserts a <strong>graph break</strong>, falling back to Python between compiled subgraphs. Passing <code>fullgraph=True</code> raises an error "
                    "if any graph break occurs. Use <code>torch._dynamo.explain(model)(*inputs)</code> to diagnose graph breaks.</p>"
                ),
            },
        ],
    },
    "torch_serialization_checkpoints.html": {
        "module": "torch",
        "symbol": "save_load_state_dict",
        "title": "Serialization and Checkpointing: torch.save, torch.load, and state_dict",
        "sections": [
            {
                "id": "weights_only_security_and_map_location",
                "heading": "Secure Checkpoint Loading with weights_only=True and map_location",
                "body": (
                    "<p><code>torch.load(f, map_location=None, *, weights_only=True, mmap=None)</code></p>"
                    "<p>Passing <code>weights_only=True</code> prevents arbitrary pickle code execution by restricting deserialization to tensors and primitive dicts. "
                    "Use <code>map_location=torch.device('cpu')</code> when loading GPU checkpoints onto CPU or remapping across distributed ranks, and "
                    "<code>mmap=True</code> to memory-map large checkpoint files without spiking host RAM.</p>"
                ),
            },
            {
                "id": "state_dict_ddp_prefix",
                "heading": "Handling state_dict strict=False and DDP 'module.' Key Prefix",
                "body": (
                    "<p><code>model.load_state_dict(state_dict, strict=False)</code> returns <code>IncompatibleKeys(missing_keys, unexpected_keys)</code> "
                    "without raising an error when fine-tuning modified heads. Because <code>DistributedDataParallel</code> prepends the prefix "
                    "<code>'module.'</code> to all parameter keys, save <code>ddp_model.module.state_dict()</code> for portable checkpoints.</p>"
                ),
            },
        ],
    },
    "torch_cuda_memory_management.html": {
        "module": "torch.cuda",
        "symbol": "CUDA_CachingAllocator",
        "title": "CUDA Memory Management, Caching Allocator, and Reproducibility",
        "sections": [
            {
                "id": "caching_allocator_and_empty_cache",
                "heading": "CUDA Caching Allocator, memory_allocated vs memory_reserved, and empty_cache()",
                "body": (
                    "<p>PyTorch manages GPU memory via an asynchronous caching allocator. <code>torch.cuda.memory_allocated()</code> returns bytes occupied by active "
                    "live tensors, whereas <code>torch.cuda.memory_reserved()</code> returns total bytes held by the allocator pool (active plus cached free blocks). "
                    "<code>torch.cuda.empty_cache()</code> releases unused cached blocks back to the CUDA driver for other processes, but does not free live tensors. "
                    "Set <code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code> to mitigate fragmentation OOMs.</p>"
                ),
            },
            {
                "id": "cudnn_benchmark_and_determinism",
                "heading": "Reproducibility: cudnn.benchmark vs torch.use_deterministic_algorithms",
                "body": (
                    "<p>Setting <code>torch.backends.cudnn.benchmark = True</code> benchmarks and selects the fastest cuDNN convolution kernel for fixed input shapes, "
                    "but degrades speed when input tensor shapes vary dynamically each step. For strict bit-wise reproducibility, set "
                    "<code>torch.backends.cudnn.benchmark = False</code>, <code>torch.backends.cudnn.deterministic = True</code>, and call "
                    "<code>torch.use_deterministic_algorithms(True)</code> with <code>CUBLAS_WORKSPACE_CONFIG=:4096:8</code>.</p>"
                ),
            },
        ],
    },
})
PYTORCH_HTML_CORPUS.update({
    "torch_tensor_views_contiguity.html": {
        "module": "torch",
        "symbol": "Tensor_Views_Contiguity",
        "title": "Tensor Views, Memory Strides, view() vs reshape(), and register_buffer",
        "sections": [
            {
                "id": "view_vs_reshape_contiguous",
                "heading": "Memory Strides, Contiguity, and tensor.view() vs tensor.reshape()",
                "body": (
                    "<p>Operations like <code>transpose()</code> and <code>permute()</code> return a view sharing storage with modified strides, "
                    "making the tensor non-contiguous (<code>tensor.is_contiguous() == False</code>). Calling <code>tensor.view(*shape)</code> on a "
                    "non-contiguous tensor raises a <code>RuntimeError</code> unless <code>tensor.contiguous().view(*shape)</code> is called first. "
                    "<code>tensor.reshape(*shape)</code> returns a view when contiguous and copies data automatically when non-contiguous.</p>"
                ),
            },
            {
                "id": "register_buffer_vs_parameter",
                "heading": "nn.Module.register_buffer vs nn.Parameter for Non-Trainable State",
                "body": (
                    "<p><code>Module.register_buffer(name, tensor, persistent=True)</code> registers a non-trainable tensor state that is not updated "
                    "by optimizers and is excluded from <code>model.parameters()</code>, yet moves automatically with <code>model.to(device)</code> and is saved "
                    "in <code>model.state_dict()</code> (unless <code>persistent=False</code>).</p>"
                ),
            },
        ],
    },
    "torch_quantization_ao.html": {
        "module": "torch.ao.quantization",
        "symbol": "Quantization_PTQ_QAT",
        "title": "Model Quantization: Dynamic Quantization, Static PTQ, and QAT",
        "sections": [
            {
                "id": "dynamic_vs_static_ptq_qat",
                "heading": "Dynamic Quantization vs Static Post-Training Quantization (PTQ) vs QAT",
                "body": (
                    "<p>PyTorch offers three INT8 quantization modes: (1) <strong>Dynamic Quantization</strong> (<code>quantize_dynamic</code>) quantizes weights "
                    "to <code>torch.qint8</code> ahead of time while computing activation scales dynamically at runtime (ideal for LSTMs and Linear layers); "
                    "(2) <strong>Static PTQ</strong> calibrates activation observers on representative data before <code>convert_fx</code>; "
                    "(3) <strong>Quantization-Aware Training (QAT)</strong> inserts <code>FakeQuantize</code> modules during training with straight-through estimators (STE).</p>"
                ),
            },
        ],
    },
    "torch_linalg_svd_einops.html": {
        "module": "torch.linalg",
        "symbol": "linalg_svd_einsum",
        "title": "Linear Algebra Operations: torch.linalg.svd and torch.einsum",
        "sections": [
            {
                "id": "linalg_svd_and_einsum",
                "heading": "torch.linalg.svd Singular Value Decomposition and torch.einsum",
                "body": (
                    "<p><code>torch.linalg.svd(A, full_matrices=True)</code> computes the Singular Value Decomposition <code>(U, S, Vh)</code> where "
                    "<code>A = U @ diag(S) @ Vh</code>. Setting <code>full_matrices=False</code> computes the reduced economy SVD of dimension <code>k = min(m, n)</code>. "
                    "<code>torch.einsum(equation, *operands)</code> evaluates multilinear Einstein summation contractions such as "
                    "<code>torch.einsum('bhsd,bhtd-&gt;bhst', Q, K)</code>.</p>"
                ),
            },
        ],
    },
})


def build_raw_corpus(output_dir: str | Path) -> List[Path]:
    """Generate the PyTorch HTML technical documentation files inside output_dir."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written_files: List[Path] = []

    for filename, doc_spec in sorted(PYTORCH_HTML_CORPUS.items()):
        html_str = _wrap_html_doc(
            module=doc_spec["module"],
            symbol=doc_spec["symbol"],
            title=doc_spec["title"],
            sections=doc_spec["sections"],
        )
        file_path = out_dir / filename
        file_path.write_text(html_str, encoding="utf-8")
        written_files.append(file_path)

    logger.info("Generated %d raw HTML documentation files in %s", len(written_files), out_dir)
    return written_files


if __name__ == "__main__":
    from src.config import load_config, resolve_project_path, setup_logging

    cfg = load_config()
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    raw_dir = resolve_project_path(cfg["paths"]["raw_data_dir"])
    build_raw_corpus(raw_dir)







