inline void ExecuteReferencePlan(
        const Plan& plan, GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        int32_t inputDtype, aclrtStream stream,
        bool transposeX1, bool transposeX2) {
    const CachedExecutionResources resources =
        GetReferenceExecutionResources(
            plan, stream, inputDtype, transposeX1, transposeX2);
    GM_ADDR atomicOutput = resources.workspace;
    const Shape shape{plan.tiling.b, plan.tiling.m,
                      plan.tiling.n, plan.tiling.k};
    CheckAcl(aclrtMemsetAsync(
                 atomicOutput,
                 static_cast<size_t>(resources.workspaceBytes), 0,
                 static_cast<size_t>(resources.workspaceBytes), stream),
             "aclrtMemsetAsync(reference atomic output)");

    if (inputDtype == 1) {
        LaunchRowMax<half>(x1, x2, atomicOutput, shape,
                           plan.tiling.launchBlocks, stream,
                           transposeX1, transposeX2);
    } else {
        LaunchRowMax<bfloat16_t>(x1, x2, atomicOutput, shape,
                                  plan.tiling.launchBlocks, stream,
                                  transposeX1, transposeX2);
    }
    const size_t outputBytes =
        static_cast<size_t>(shape.b) * sizeof(float);
    CheckAcl(aclrtMemcpyAsync(
                 y, outputBytes, atomicOutput, outputBytes,
                 ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
             "aclrtMemcpyAsync(reference atomic output to y)");
}
