# Agentic Reconstructor — Architecture

Two Mermaid diagrams that together describe the agentic scene-reconstruction
loop in `src/isaacsim_bench/agents/`.

- **Figure 1 — System level.** Inputs the reconstructor consumes, the three
  internal components, and what comes out. Use this to set up the mental
  model in 30 seconds.
- **Figure 2 — Inside the loop.** What happens turn-by-turn inside the VLM
  driver: which tool categories the agent can call, what state they mutate,
  which gates block which transitions. Use this to discuss the
  inventory / render / submit gates — the load-bearing pieces of the
  current design.

Both render natively on GitHub. To export to PNG / SVG for slides:

```bash
npm i -g @mermaid-js/mermaid-cli
mmdc -i docs/agentic_architecture.md -o agentic_architecture.svg -t neutral
```

---

## Figure 1 — System level

```mermaid
flowchart LR
    subgraph IN[Inputs]
        REF[Reference images<br/>scene.json camera<br/>opt]
        TAX[Asset taxonomy<br/>6321 variants]
        POOL[Retrieval pool]
        ANCH[Anchor registry<br/>asset_anchors.json]
        EXT[Asset extents<br/>bbox per asset]
        THUMB[Thumbnails for<br/>CLIP catalog]
    end

    subgraph CORE[AgenticReconstructor]
        CLIP[CLIP catalog<br/>top-k visual shortlist]
        LOOP[Agentic loop<br/>VLM driver + tools]
        REND[Isaac render session<br/>opt: --enable-render-tool]
    end

    subgraph OUT[Outputs]
        PRED[PredictionJSON]
        USD[scene.usd<br/>ComposerAgent.export_usd]
        TRACE[agentic_trace.json<br/>tool call log]
        EVAL[Evaluator report<br/>component / placement / relation]
    end

    REF --> CLIP --> LOOP
    TAX --> LOOP
    POOL --> LOOP
    ANCH --> LOOP
    EXT --> LOOP
    THUMB --> CLIP
    LOOP <--> REND
    LOOP --> PRED --> EVAL
    PRED --> USD
    LOOP --> TRACE
```

**Source files**

| Block | File |
|---|---|
| AgenticReconstructor / loop driver | `src/isaacsim_bench/agents/agentic.py` |
| Tool handlers + AgentState | `src/isaacsim_bench/agents/agentic_tools.py` |
| Render tool (Isaac Sim bound) | `src/isaacsim_bench/agents/agentic_render.py` |
| CLIP catalog | `src/isaacsim_bench/agents/clip_catalog.py` |
| VLM provider abstraction | `src/isaacsim_bench/agents/vlm.py` |
| Anchor registry | `src/isaacsim_bench/schemas/anchors.py` |
| Asset extents | `data/asset_extents.json` |
| Evaluator | `src/isaacsim_bench/evaluator/` |

---

## Figure 2 — Inside the loop

```mermaid
flowchart TD
    START([Start: refs + pool + tools]) --> SYS[System prompt<br/>build_system_prompt]
    SYS --> MSG0[Initial user msg:<br/>refs + pool summary]
    MSG0 --> TURN{turn N ≤ max_turns?}

    TURN -- yes --> BUDGET{Token / render<br/>budget left?}
    BUDGET -- no --> EXIT_BUDGET[stop_reason = budget]
    BUDGET -- yes --> VLM[VLM.run_turn<br/>system + messages + tools]

    VLM --> RESP{Tool calls?}
    RESP -- none --> NUDGE[Inject 'use a tool' nudge<br/>force tool_choice = any]
    NUDGE --> TURN
    RESP -- one or more --> DISPATCH[Dispatch each tool call]

    DISPATCH --> CAT{Tool category}

    CAT -- inventory --> INV[set_inventory<br/>mark_matched<br/>unmark_matched<br/>add_inventory_item<br/>update_inventory_item]
    CAT -- search --> SRCH[get_catalog_hits<br/>list_families<br/>list_assets_in_family<br/>get_asset_info]
    CAT -- edit --> EDIT[add_component<br/>add_aligned_component<br/>align_components<br/>modify_component<br/>remove_component]
    CAT -- inspect --> INSP[list_components<br/>get_component_anchors<br/>plot_top_down<br/>view_reference]
    CAT -- render --> RENDER[Isaac render<br/>+ scene_hash cache]
    CAT -- submit --> SUB[submit_prediction]

    INV --> STATE[(AgentState<br/>prediction +<br/>inventory +<br/>scene_modified)]
    SRCH --> STATE
    EDIT -- pre-check --> G1{Inventory<br/>locked?}
    G1 -- no --> EDIT_BLOCK[Reject:<br/>call set_inventory first]
    EDIT_BLOCK --> STATE
    G1 -- yes --> G2{Render gate:<br/>scene modified<br/>since last render?}
    G2 -- yes --> EDIT_BLOCK
    G2 -- no --> STATE
    INSP --> STATE
    RENDER --> CLEAR[scene_modified = False]
    CLEAR --> STATE

    SUB --> SG{All submit gates pass?}
    SG -- "no: inventory unmatched<br/>/ extras / un-rendered" --> SUB_BLOCK[Reject with hint]
    SUB_BLOCK --> STATE
    SG -- yes --> DONE[stop_reason = submitted]

    STATE --> APPEND[Append tool results<br/>to messages]
    APPEND --> TURN

    TURN -- no --> EXIT_TURN[stop_reason = turn_budget]
    DONE --> RET([Return PredictionJSON + AgenticRunResult])
    EXIT_BUDGET --> RET
    EXIT_TURN --> RET
```

### Gate hierarchy (the load-bearing rules)

1. **Inventory lock gate** — `add_component` / `modify_component` /
   `align_components` / `add_aligned_component` refuse to run until
   `set_inventory` has been called. Forces the agent to commit a
   structured count before any placement.
2. **Render-after-edit gate** — every scene edit must be followed by a
   `render` before the next edit, so the agent can attribute each change
   to a visible delta. Skipped when the render tool isn't wired in.
3. **Submit gates** (`submit_prediction`):
   - inventory locked + non-empty
   - render-after-edit clear (when render tool wired)
   - every inventory item has ≥1 matched component *or* is listed in
     `acknowledge_unmatched`
   - every placed component is matched to some inventory item *or* is
     listed in `acknowledge_extras` — catches under-counting where the
     agent treats a modular assembly as one item

### Why this shape

The fixed-pipeline reconstructor in `orchestrator.py` runs
`perception → retrieval → spatial → composer → critic` as a deterministic
chain. The agentic loop replaces that with a single VLM driver that
decides which tool to call next, iterating until it submits or hits a
budget. The structured inventory + gates exist because the unguarded
loop was happy to submit the first plausible scene and exit; the gates
force the agent to record an explicit count, place against it, and
account for every component before the loop can terminate.

---

## Suggested usages

| Audience / context | Use |
|---|---|
| 30-second elevator pitch | Figure 1 only |
| Progress meeting / writeup | Both figures |
| Slide deck | Figure 1 + a simplified Figure 2 (drop search / inspect branches) |
| Discussing a specific gate | Figure 2 + the gate hierarchy bullet list |
