"""edgeopt: optimising AI models for edge inference on resource-constrained devices.

Experiment code for the paper of the same name. Modules:

    models           model families (DS-CNN, residual CNN, compact transformer) with width/resolution scaling
    data             CIFAR-10 and Speech Commands (12-class) loaders
    train            training loop with fixed budget, knowledge distillation (Eq. 16), pruning hooks
    prune            structured pruning (real channel removal) and gradual unstructured magnitude pruning
    quantize         post-training INT8 (static for CNNs, dynamic for transformers) and FP16
    profile          parameters, MACs, peak activation memory, serialised size
    benchmark        latency protocol of Section 7.3 (warm-up, median/p95/p99, threads, cold start)
    stats            bootstrap confidence intervals (Eq. 17), Pareto front, rank utilities
    run_experiments  the staged campaign of Section 7.7
    analyse          Pareto plots and the hypothesis checks H1, H3, H4, H5 of Table 22
"""
