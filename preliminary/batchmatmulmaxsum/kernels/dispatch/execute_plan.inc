inline void ExecutePlan(
        const Plan& plan, GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        int32_t inputDtype, aclrtStream stream,
        bool transposeX1, bool transposeX2) {
    switch (plan.path) {
        case KernelPath::REFERENCE:
            ExecuteReferencePlan(plan, x1, x2, y, inputDtype, stream,
                                 transposeX1, transposeX2);
            return;
        case KernelPath::AUTO_FUSED:
            switch (plan.key) {
                case TilingKey::AUTO_MATMUL_FUSED:
                    ExecuteAutoFusedPlan<false, false, 1, 64, 128, 64>(
                        plan, x1, x2, y, inputDtype, stream,
                        transposeX1, transposeX2);
                    return;
                case TilingKey::AUTO_MATMUL_FUSED_VECTOR:
                    ExecuteAutoFusedPlan<true, false, 1, 64, 128, 64>(
                        plan, x1, x2, y, inputDtype, stream,
                        transposeX1, transposeX2);
                    return;
                case TilingKey::AUTO_MATMUL_FUSED_ASYNC:
                    ExecuteAutoFusedPlan<true, true, 1, 64, 128, 64>(
                        plan, x1, x2, y, inputDtype, stream,
                        transposeX1, transposeX2);
                    return;
                case TilingKey::AUTO_MATMUL_FUSED_ASYNC_DB:
                    ExecuteAutoFusedPlan<true, true, 2, 64, 128, 64>(
                        plan, x1, x2, y, inputDtype, stream,
                        transposeX1, transposeX2);
                    return;
                case TilingKey::AUTO_MATMUL_FUSED_SPLIT_N:
                    if (plan.tiling.splitN > 1) {
                        ExecuteAutoFusedSplitNPlan<64, 128, 64>(
                            plan, x1, x2, y, inputDtype, stream,
                            transposeX1, transposeX2);
                    } else {
                        ExecuteAutoFusedPlan<true, true, 2, 64, 128, 64>(
                            plan, x1, x2, y, inputDtype, stream,
                            transposeX1, transposeX2);
                    }
                    return;
                default:
                    throw std::invalid_argument(
                        "Selected automatic fused key is not implemented");
            }
    }
    throw std::invalid_argument("Selected kernel path is invalid");
}

}  // namespace bmms
