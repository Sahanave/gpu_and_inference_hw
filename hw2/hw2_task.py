import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)
from torch.profiler import profile as torch_profile
from torch.profiler import ProfilerActivity, record_function

def optimized_loop(model, input_ids, n_steps):
    # TODO: fix the performance issues you found — changes may include
    # both `optimized_loop` and `generate_optimized`
    generated_tokens = []
    
    # ── prefill: process the full prompt ONCE, build the cache
    outputs = model(input_ids=input_ids, use_cache=True)
    past = outputs.past_key_values
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)   # (1,)
    generated_tokens.append(next_token_id)

    for _ in range(n_steps - 1):
        cur = next_token_id.unsqueeze(0)                              # (1,1)
        outputs = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token_id)
    return torch.cat(generated_tokens).tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    # TODO: wrap loop_fn(model, input_ids, PROFILE_STEPS) with torch.profiler,
    # print the summary table, and export a Chrome trace to RESULTS_DIR / trace_name
    with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        trace = trace_name.split('.')[0]
        with record_function(trace):
            loop_fn(model,input_ids,PROFILE_STEPS)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(str(RESULTS_DIR / f"{trace_name}"))



def generate_optimized(optimized_trace_name: str) -> float:
    # TODO: load the model (consider dtype and other loading options),
    # then call profile() and time_generation() on optimized_loop.
    # Return the elapsed time from time_generation so main() can print a speedup.
    model = torch.compile(build_model(torch.bfloat16), dynamic=True)
    input_ids = get_input_ids()
    optimized_loop(model, input_ids, PROFILE_STEPS)   # warmup: compile BEFORE profiling
    torch.cuda.synchronize()
    profile(optimized_loop, model, input_ids, optimized_trace_name)   # now a clean trace
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    del model
    torch.cuda.empty_cache()
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Numbers below were measured on an H100 (development GPU). The graded run is on
# an L40S; ratios are expected to be at least as high there (see note at the end).
#
# Changes made and speedup per fix:
#
# (0) Baseline V0: slow_loop, fp32, no KV cache. ~0.95s for 128 tokens.
#
# (1) Sync once, not per step.
#     The baseline calls next_token_id.item() every step. In the trace that
#     shows up as aten::item -> aten::_local_scalar_dense -> cudaStreamSynchronize:
#     the CPU blocks until the GPU drains all queued work, copies back one token,
#     and only then launches the next step. That serializes CPU and GPU.
#     Fix: keep token ids on the GPU, collect them, and do ONE
#     torch.cat(tokens).tolist() after the loop.
#     Speedup contributed: ~7 ms only. Small
#
# (2) KV cache (the big one).
#     The baseline re-runs the full forward over the whole growing sequence every
#     step -- record_shapes shows the matmul/attention input seq dim growing
#     1024 -> 1035 -> ..., i.e. O(n^2) work, recomputing past tokens' K/V that
#     never change. Fix: prefill the prompt once with use_cache=True, then feed
#     only the 1 new token each step carrying past_key_values; the growing
#     torch.cat on the sequence is removed entirely.
#     Speedup contributed: ~0.95s -> ~0.28s == ~3.4x. This is essentially the
#     whole speedup.
#
# (3) bf16 weights (build_model(torch.bfloat16)).
#     Confirmed active in the trace (nvjet tensor-core GEMMs + flash attention vs
#     fp32 xmma/cutlass in V0). Speedup contributed on H100: ~0 (276ms -> 280ms).
#     Reason as : after the KV cache the loop is CPU-dispatch-bound
#     (Self CUDA ~4.7ms vs Self CPU ~164ms over the profile), so halving the
#     already-tiny GPU compute does nothing to wall-clock. Kept because it's free
#     and should help more on the bandwidth-poor L40S (the prefill is GPU-bound).
#
# (4) torch.compile(model, dynamic=True) -- the strong second win.
#     After the KV cache the loop is CPU-dispatch-bound (Self CUDA ~4.7ms vs Self
#     CPU ~164ms): the cost is launching ~hundreds of tiny ops per step, not the
#     GPU math. torch.compile (Inductor) FUSES many ops into single Triton kernels
#     -- the trace shows kernels like triton_poi_fused__to_copy__unsafe_view_add_
#     bmm_cat_... and Torch-Compiled Region / CompiledFunction entries. Fewer
#     kernels => fewer launches => the dispatch bottleneck shrinks. dynamic=True
#     is essential: the KV cache length changes every step, so without it compile
#     would re-specialize (recompile) on every new length; dynamic=True compiles
#     one shape-generic graph instead.
#     Speedup contributed: ~0.28s -> ~0.17s == ~3.36x -> ~5.59x.
#
# Final: ~5.59x on H100 (KV cache + sync-once + bf16 + torch.compile).
#
# Biggest impact and why:
#   The KV cache, by far. It turns each decode step from a full O(seq) forward
#   over the entire growing sequence (recomputing all past tokens' K/V) into an
#   O(1) single-token forward that reuses cached K/V -- collapsing the loop's
#   total work from O(n^2) to O(n) (~0.95s -> ~0.28s). That fix also CHANGED the
#   bottleneck: with the GPU math now tiny, the loop became CPU-dispatch-bound,
#   which is exactly why torch.compile's op-fusion then mattered as the second
#   win (~0.28s -> ~0.17s). The recurring lesson:
#   each fix only pays for the bottleneck that is currently dominating.
