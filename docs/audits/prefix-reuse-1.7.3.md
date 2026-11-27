# Prefix reuse in Splash 1.7.3: what skips GPU prefill, and what only saves memory

A read of the source at tag 1.7.3 (`d40c250f`). Nothing here was built or run. Each claim cites the file and the lines it rests on, and the experiments at the end are meant to confirm or refute those claims. The question this audit answers: when several requests share a token prefix, which part of the prefill is actually skipped on the GPU, and which part is computed again and only deduplicated in memory afterwards?

## Summary

- **Compute is skipped only from a cached state.** A request resumes from the deepest state its prefix reaches. That state is the GDN/recurrent state plus the draft context. The model is given `restoredTokens = resumeBoundary` (`runtime/engine/Engine.cpp:1035`).
- **Matching KV beyond that state is recomputed.** If the request's KV matches further than its state, those rows still go through the full forward pass. Their pages are then swapped for the cached ones (`Cache::publishCommittedBlocks`, `runtime/engine/Cache.cpp:418-482`) and counted as `deduplicated_kv_tokens`. That saves memory, not compute. On a hybrid model it cannot be otherwise: GDN rows need every layer.
- **States sit only at planned boundaries, all at 32-token pages.** A finished prompt keeps four states:
  - the rolling checkpoint below the head;
  - the head itself;
  - the last checkpoint below the replay point;
  - the replay point.

  Every other rolling checkpoint (each 4096 tokens) is retired as the lane moves on (`Engine.cpp:1509-1510`, `1566-1567`; `StateCache::retireCheckpointState`, `StateCache.cpp:222-228`).
- **In-flight sharing exists.** A request that shares a prefix with a prompt still in prefill waits (`waiting_prefix`), holding no lane or pages. The producer's command is cut at a boundary the waiter can resume from (`Engine::pendingSharedPrefill`, `Engine.cpp:925-956`; cut at `1731-1741`). So simultaneous identical cold prompts are computed once.

## Request path

| Stage | Where |
|---|---|
| Render and tokenize | `frontend._render_prompt` (`server/frontend.py:719-808`). `_generation_prompt` and `_head_tokens` (`frontend.py:99-127`) mark where the generation prompt starts and where the stable head ends. |
| Salt and submit | `request_fields.cache_salt` (`server/request_fields.py:111-123`); `backend._generation_request` and `submit` (`server/backend.py:900-929`). On the wire as lo/hi (`server/protocol.py:985-1006`). |
| Admission | `Engine::submit` (`Engine.cpp:95-148`), then `admitQueued` (`673-869`), then `Cache::probe`/`refresh` (`Cache.cpp:276-297`), then `pendingSharedPrefill`, which either waits or admits (`979-1145`). |
| Lookup and restore | `Cache::lookup` (`Cache.cpp:299-336`), then `Cache::restoreRequest` (`2580-2694`), which takes the chain up to the state block only. Then `configureDraftStatePlan` (`Engine.cpp:1281-1310`) and `Runtime::beginRestore` (`runtime/model/Runtime.cpp:3605`). An SSD restore goes through `pollRestores` (`Engine.cpp:1221-1264`). |
| Dispatch | `Scheduler::next` (`Scheduler.cpp:346-373`), then `Engine::prepare`, then `Runtime::prefillAsync` / `encodePrefillGraph` (`Runtime.cpp:3688`, `2256`). |
| Commit | `Cache::publishCommittedBlocks` (`Cache.cpp:418-482`, with `KvCache::insert` at `KvCache.cpp:107-205`), then `publishStateBoundary` (`Engine.cpp:1409-1575`), then write-behind. |
| Eviction | `Cache::walkFor` / `nextUnit` (`Cache.cpp:727-874`), then `StateCache::reclaim` (`StateCache.cpp:370-392`) or `reclaimKvLeaf` (`Cache.cpp:1615-1643`). |
| SSD restart | `Cache::adopt` (`Cache.cpp:71-185`) at startup; `startRestore` (`Cache.cpp:2703`) per request. |

## Matching

- **Key.** A block is keyed by its parent block, its exact 32 token ids and a `BlockIdentity` (`KvCache::find`, `KvCache.cpp:86-105`). The chained hash only filters candidates. Images overlapping the block are part of the identity. The salt is folded into the root block only (`KvCache.cpp:58-71`).
- **Granularity.** Pages are 32 tokens (`runtime/metal/abi/ExecutionGeometry.h:50`). Only whole pages match, at most `(len−1)/32` of them (`Cache.cpp:235-255`). A partial page never matches.
- **Scope.** A cache is global to the process, partitioned by salt and images. The persistent tier's namespace also hashes the model manifest, the KV/GDN/draft layouts and rope (`RuntimeResources.mm:534-579`).
- **Salt.** It comes from the request field `cache_salt`; a missing one is 0. The value is hashed with blake2b-128 (person `splash-salt`). It isolates blocks, states and in-flight sharing (`Engine.cpp:896`). Encoded image rows are shared regardless (`docs/api.md:70-75`).

## Where states are taken, and what survives

- **Planned** (`Engine.cpp:1281-1310`):
  - rolling checkpoints at multiples of `kPrefillCheckpointTokens` = 2 × `draftContextTokens` (4096). Only those at least one 2048-token chunk past the resume point and one chunk below the replay point are planned (`plannedCheckpoints`, `Engine.hpp:94-105`);
  - a junction where a branch with a state meets the request, when it gains at least 2048 tokens (`Cache.cpp:323-324`);
  - the head, at floor32(head tokens), when it is 2048 or more past the resume point;
  - the replay point, at floor32(prompt − generation prompt) (`Engine.cpp:879-882`);
  - boundaries shared with peers that have not started yet (`Engine.cpp:1338-1352`);
  - reply states captured during decode (`EngineMemory.cpp:790`).
- **Retained.** A new checkpoint or junction retires the lane's previous checkpoint. The head and the replay point keep the checkpoint just below them (`keepsCheckpoint`, `Engine.cpp:888-891`).
- **Consequence.** A later request that shares L tokens resumes at the deepest kept state at or below L, and recomputes the rest. The distance is not bounded by 4096. Example: a 30K prompt with a 10K head keeps states at 8192, 10K, 24576 and its replay point. A request sharing the first 20K resumes at 10K and recomputes 10K tokens. Only then does a lazy junction give later requests 20K.

## In-flight sharing

- **What the TTFT log's "claim" is.** In the per-request breakdown ("own prefill, claim wait, decode, other prefill, unit wait", written at `NativeRuntime.cpp:447-448`), "claim" is the request's memory claim in the governor's ledger (`MemoryLedger.hpp:372-389`), not a prefix claim.
  - A request waiting on a prefix has its claim cleared (`deferWait`, `EngineMemory.cpp:550-567`), so its wait shows up as "other prefill".
  - "own prefill" is the wall time of the commands that carried the request's rows, and is shared with its batch-mates; it is not per-request GPU time.
- **When B waits for A.** B, not yet started, waits for a producer A when all of these hold (`Engine.cpp:925-956`):
  - A holds a lane and is prefilling or restoring;
  - A has the same or a more urgent priority;
  - A still has a planned boundary b with B's resume point < b ≤ shared(A, B).

  shared(A, B) = floor32(min(common token prefix, both replay boundaries)). It is cut at the first differing image, and is 0 across salts (`Engine.cpp:893-923`). While waiting, B holds no lane or pages. A's command is cut to end at b (`capture_cuts`), then B is admitted and resumes there.
- **Partial overlap.** A request already queued when A is admitted gets a boundary at its divergence point, with no minimum. A request that arrives later gets one only after A's next command reaches a boundary, and only 2048 past A's progress (`Engine.cpp:1331-1333`, `1451-1457`). If A has already passed the divergence, B does not wait: it resumes from A's last published state at or below the divergence and recomputes the rest beside A.
- **Producer cancellation.** `settle` sets `pendingEnd` (`Engine.cpp:2143-2158`) and the waiter is released at the next admission pass. A's in-flight command publishes nothing (`1935-1942`); states from its completed commands stay (`vacateLane`, `2242-2269`). No prefix-wait timeout exists beyond the request's own deadline.
- **Who never waits.**
  - A more urgent B never waits (`Engine.cpp:931`).
  - With every lane busy, B is counted as `waiting_resources`, not `waiting_prefix` (`755-758`).

## Scheduling

- `--decode-share` (default 0.5, finite ≥ 0; `server/serve_options.py:285-292`, `Engine.hpp:62`). While same-or-more-urgent requests decode, each prefill unit adds share × its wall time to a decode debt, and decode commands pay it off. A share of 0 means strict alternation (`Scheduler.cpp:346-373`, `780-790`).
- Not on the CLI:
  - the 2048-token prefill chunk (`ExecutionGeometry.h:48`);
  - the 512-row floor (`Scheduler.cpp:16`);
  - the 500 ms decoder-landing wait (`Engine.hpp:67`);
  - the 30 s resource wait (`Engine.hpp:37`);
  - background requests ageing to normal after 10 s (`Engine.cpp:31`).
- `--max-concurrent-requests` sets the lanes (default 4, at most 8).

## Eviction and the SSD tier

- **Eviction order** (`Cache.cpp:782-874`). It is ordered by class, then least recently used, and is not cost-aware:
  1. checkpoints;
  2. KV leaves no state restores through;
  3. ordinary states and leaves, by least recent use;
  4. in-use units.
- **What goes to disk.** An evicted state is written when the tier is writable (`StateCache.cpp:379-391`). A KV leaf is written only if a state needs it (`Cache.cpp:1631-1642`).
- **Persistence.** The tier is temporary unless `--persistent-cache` is given (`serve_options.py:655-701`). Write-behind copies heads and replay points that are 2048 or more tokens deep, after 10 s (`WriteBehind.hpp:55-56`). `adopt` takes back only the states and the KV chains they need.

## What the status reports

| Kind | Fields |
|---|---|
| Work skipped or done | Per request: `usage.prompt_tokens_details.cached_tokens` (the resume boundary, `backend.py:1100-1104`). Globally: `metrics.prefill_input_tokens` (committed rows), `scheduler.prefill_rows`, `cache.reused_tokens`, `cache.hits`, `cache.cold_misses` (`Engine.cpp:1211-1216`), `model_timing.prefill.total_gpu_ms` (includes warm-up). |
| Matched, not skipped | `prefix_tokens`, `cache.kv_hit_tokens` (`Cache.cpp:339`), `cache.deduplicated_kv_tokens` (rows computed and then swapped). |
| Not available | per-request GPU milliseconds; a counter or duration of prefix waits (there is only the `scheduler.waiting_prefix` gauge); a counter of rows recomputed past a state. |

## Audit matrix

| Case | Native mechanism | Expected from the source | Remaining redundant compute | Where a fix would belong |
|---|---|---|---|---|
| A then B, identical prompt | replay point | B's cached = floor32(P − generation prompt) | the last partial page + the generation prompt (#588) | none (upstream #588 tracks it) |
| Simultaneous identical cold prompts | boundary planned at A's admission; B in `waiting_prefix` | B resumes at the replay point; its TTFT is at least A's | none, unless B is more urgent (then everything) | none |
| B arrives during A's prefill | wait for A's deepest boundary ≤ shared | identical prompt: the replay point | when the shared point is behind A's progress: from A's last checkpoint, beside A | runtime |
| Early divergence | none | cached ≈ 0 | the whole prompt (causal, inherent) | client prompt order |
| Late divergence, sequential | the deepest surviving state, then a lazy junction | resume at the head or the last kept checkpoint | up to L − head, because rolling checkpoints were retired; the first divergent request pays | runtime (retention), client |
| Short head (< 2048 tokens) | none sequentially; concurrent boundaries have no minimum | cached 0 for a new conversation | the whole head (#874) | runtime |
| Producer cancelled or failed | `pendingEnd` releases waiters | resume at A's last published state | A's in-flight command | runtime (#866) |
| Memory budget full | reply-state capture refused (#818) | the next turn re-prefills from the last kept state | the reply and everything after the last state | runtime (#818) |
| SSD eviction or restart | write-behind, offload, adopt | `state_source` = disk | everything, without `--persistent-cache`; SSD read time otherwise | server configuration |
| Thinking or effort change | the replay point sits before the generation prompt | no effect if the template renders effort only there | everything after the change when the template puts effort text at the top of the system prompt | client / template |
| Tools added or changed | head state, lazy junction | resume at the checkpoint below the change | everything after the change (#880) | client (stable, append-only tools), #880 |

## Experiments that would confirm or refute this

Each run snapshots `/status` before and after and diffs these fields:

- globally: `metrics.prefill_input_tokens`, `cache.reused_tokens`, `cache.deduplicated_kv_tokens`, `model_timing.prefill.total_gpu_ms`;
- per request: `cached_tokens` and `prefix_tokens`.

Each run uses a cache salt of its own, so it neither reads nor leaves blocks for other traffic.

1. **Identical pair, sequential and then simultaneous.** `prefill_input_tokens` grows by about P, not 2P. With different salts it grows by 2P.
2. **Late arrival.** Start a 40K prompt, and send B when about 10K is processed:
   - B identical: `capture_cuts` rises and B's cached = A's replay point;
   - B diverging at 20K: cached = 20480;
   - B diverging at 8K: cached is a checkpoint at or below 8K, and `deduplicated_kv_tokens` > 0.
3. **Late divergence, sequential.** A: 30K with a 10K head. B shares the first 20K: cached ≈ 10K, `prefix_tokens` ≈ 20K. C shares the same 20K: cached ≈ 20K.
4. **Cancellation.** Cancel A at about 15K while B waits. `waiting_prefix` returns to 0 and B's cached = 12288.
5. **Short head.** A 1000-token shared head gives cached 0 sequentially and 992 when the requests arrive together.
6. **Restart.** With `--persistent-cache`, wait past 10 s, restart, resend: `state_source` = disk.
7. **Effort and tools.** Vary effort through the API field and through system text; reorder tools; add one tool. Compare cached with the position of the first difference.

## Upstream issues this touches

All of these are open at the time of writing:

- **incoai/splash#588:** the replay point is page-aligned, so a repeat re-reads 5-36 tokens.
- **incoai/splash#818:** on hybrid models, a reply's state capture is refused once cached KV fills the budget.
- **incoai/splash#866:** a time-bounded prefill pass, which would bound cancel latency.
- **incoai/splash#874:** no head state when system prompt + tools is under 2048 tokens.
- **incoai/splash#877:** normal and background requests starve under steady foreground traffic (1.7.1).
- **incoai/splash#880:** tools declared mid-conversation without busting the prefix cache.
