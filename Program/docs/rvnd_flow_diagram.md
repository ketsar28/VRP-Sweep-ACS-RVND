# RVND Algorithm Flow Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RVND MAIN CONTROLLER                     │
│                                                              │
│  Input: Initial Solution from ACS                           │
│  Output: Improved Feasible Solution                         │
│                                                              │
│  State:                                                      │
│    • iter_inter = 1  (never resets)                        │
│    • iter_intra = 1  (never resets)                        │
│    • current_solution                                        │
│    • current_metrics                                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   INTER-ROUTE RVND PHASE          │
        │   (Multi-route scenarios only)    │
        │                                   │
        │   Max Iterations: 50              │
        │   Neighborhoods: 6                │
        │     • shift(1,0)                  │
        │     • shift(2,0)                  │
        │     • swap(1,1)                   │
        │     • swap(2,1)                   │
        │     • swap(2,2)                   │
        │     • cross                       │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   INTRA-ROUTE RVND PHASE          │
        │   (Local search within route)     │
        │                                   │
        │   Max Iterations: 100             │
        │   Neighborhoods: 4                │
        │     • 2-opt                       │
        │     • or-opt                      │
        │     • reinsertion                 │
        │     • exchange                    │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   RETURN IMPROVED SOLUTION        │
        │                                   │
        │   Guarantees:                     │
        │     ✓ Feasible                   │
        │     ✓ Non-worsening              │
        │     ✓ All constraints satisfied   │
        └───────────────────────────────────┘
```

## Detailed Intra-Route RVND Flow

```
START rvnd_intra
    │
    ├─ current_solution = initial_sequence
    ├─ current_metrics = evaluate(current_solution)
    ├─ iter_count = 0
    │
    ▼
┌─────────────────────────────────────────────┐
│ OUTER LOOP: while iter_count < MAX_INTRA   │
└─────────────────────────────────────────────┘
    │
    ├─ NL_intra = ["2-opt", "or-opt", "reinsertion", "exchange"]
    │
    ▼
┌─────────────────────────────────────────────┐
│ INNER LOOP: while NL_intra not empty       │
└─────────────────────────────────────────────┘
    │
    ├─ neighborhood = random_choice(NL_intra)
    ├─ new_solution = apply_neighborhood(current_solution)
    │
    ▼
┌─────────────────────────────────────────────┐
│ new_solution == None?                       │
└─────────────────────────────────────────────┘
    │
    ├─ YES ───► Remove neighborhood from NL_intra
    │           │
    │           └──► Continue to next neighborhood
    │
    ├─ NO
    │   │
    │   ▼
    │ new_metrics = evaluate(new_solution)
    │
    ▼
┌─────────────────────────────────────────────┐
│ is_solution_better(new, current)?          │
│   • Check feasibility                       │
│   • Check objective improvement             │
└─────────────────────────────────────────────┘
    │
    ├─ YES ───► ACCEPT IMPROVEMENT
    │           │
    │           ├─ current_solution = new_solution
    │           ├─ current_metrics = new_metrics
    │           │
    │           └──► RESET NL_intra (FULL LIST)
    │                │
    │                └──► BREAK (restart inner loop)
    │
    ├─ NO ────► REJECT
    │           │
    │           └──► Remove neighborhood from NL_intra
    │                │
    │                └──► Continue to next neighborhood
    │
    ▼
┌─────────────────────────────────────────────┐
│ NL_intra empty?                             │
└─────────────────────────────────────────────┘
    │
    ├─ YES ───► EXIT inner loop
    │           │
    │           └──► BREAK outer loop (no more improvements)
    │
    ├─ NO ────► Continue inner loop
    │
    ▼
┌─────────────────────────────────────────────┐
│ iter_count += 1                             │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ iter_count >= MAX_INTRA?                    │
└─────────────────────────────────────────────┘
    │
    ├─ YES ───► EXIT (early stopping)
    │
    ├─ NO ────► Continue outer loop
    │
    ▼
RETURN current_metrics
```

## Neighborhood Selection Process

```
┌─────────────────────────────────────────────┐
│          NEIGHBORHOOD LIST (NL)             │
│                                             │
│  Initial State: [N1, N2, N3, N4]           │
│  (All neighborhoods available)              │
└─────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Random Selection      │
        │ neighborhood = N2     │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Apply N2              │
        │ Evaluate Result       │
        └───────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ IMPROVEMENT   │       │ NO IMPROVEMENT│
│               │       │               │
│ Accept        │       │ Reject        │
│               │       │               │
│ NL = [N1,N2  │       │ NL = [N1,N3  │
│      N3,N4]  │       │      N4]      │
│ (RESET!)     │       │ (REMOVE N2)   │
└───────────────┘       └───────────────┘
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ Restart with  │       │ Continue with │
│ full NL       │       │ reduced NL    │
└───────────────┘       └───────────────┘
```

## Feasibility Checking Flow

```
┌─────────────────────────────────────────────┐
│         Candidate Solution Evaluation       │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────────┐   ┌──────────────────┐
│ CAPACITY CHECK   │   │ TIME WINDOW CHECK│
│                  │   │                  │
│ total_demand ≤   │   │ arrival_time ≤   │
│ vehicle_capacity │   │ due_time         │
└──────────────────┘   └──────────────────┘
        │                       │
        ▼                       ▼
   ┌─────────┐            ┌─────────┐
   │ Pass/Fail│            │ Pass/Fail│
   └─────────┘            └─────────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Both constraints OK?  │
        └───────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
    ┌────────┐            ┌────────┐
    │ FEASIBLE│            │INFEASIBLE│
    │         │            │         │
    │ Check   │            │ REJECT  │
    │objective│            │         │
    └────────┘            └────────┘
        │
        ▼
┌─────────────────┐
│ objective_new < │
│ objective_curr? │
└─────────────────┘
        │
    ┌───┴───┐
    │       │
    ▼       ▼
┌────────┐ ┌────────┐
│ ACCEPT │ │ REJECT │
└────────┘ └────────┘
```

## Operator Application Examples

### 2-opt Example

```
Before: [0, 1, 2, 3, 4, 5, 0]
                ╭─────╮
Apply 2-opt(2,4)
                ╰─────╯
After:  [0, 1, 4, 3, 2, 5, 0]
             └─reverse─┘
```

### Or-opt Example

```
Before: [0, 1, 2, 3, 4, 5, 0]
             ╭───╮
Apply or-opt(2, length=2, j=5)
             │   │
             ╰───┼─────────╮
                 │         │
After:  [0, 1, 4, 5, 2, 3, 0]
                   └──┴──┘
```

### Reinsertion Example

```
Before: [0, 1, 2, 3, 4, 5, 0]
             │
Apply reinsertion(2, 5)
             │
             ╰───────────╮
                         │
After:  [0, 1, 3, 4, 5, 2, 0]
                      └─┘
```

### Exchange Example

```
Before: [0, 1, 2, 3, 4, 5, 0]
             │       │
Apply exchange(2, 4) 
             ╰───┬───╯
                 │
After:  [0, 1, 4, 3, 2, 5, 0]
             └─swap─┘
```

## State Transition Diagram

```
┌──────────────┐
│   INITIAL    │
│   SOLUTION   │
└──────┬───────┘
       │
       ▼
┌──────────────┐     iter_inter++
│  INTER-ROUTE │◄────────────┐
│     RVND     │             │
└──────┬───────┘             │
       │                     │
       │ NL_inter empty      │
       │ OR                  │
       │ iter_inter > MAX    │
       │                     │
       ▼                     │
┌──────────────┐             │
│  INTRA-ROUTE │             │
│     RVND     │             │
└──────┬───────┘             │
       │                     │
       │ NL_intra empty      │
       │ OR                  │
       │ iter_intra > MAX    │
       │                     │
       ├─────────────────────┘
       │ (loop back if needed)
       │
       │ No more improvements
       │
       ▼
┌──────────────┐
│    FINAL     │
│   SOLUTION   │
│              │
│  ✓ Feasible  │
│  ✓ Improved  │
└──────────────┘
```

## Iteration Counter Behavior

```
Time ───────────────────────────────────────────►

iter_inter:  1  2  3  4  5  6  7  8  9 ... 50
             │  │  │  │  │  │  │  │  │     │
             │  │  │  │  │  │  │  │  │     └─ MAX_INTER reached
             │  │  │  │  │  │  │  │  │
             ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
        Inter→Intra→Intra→Intra→...→Intra
                                    (cycles)

iter_intra:  1  2  3  4  5  6 ... 100
             │  │  │  │  │  │      │
             │  │  │  │  │  │      └─ MAX_INTRA reached
             │  │  │  │  │  │
             ▼  ▼  ▼  ▼  ▼  ▼
        Continuous increment (NO RESET)
```

## Decision Tree

```
                    ┌─────────────┐
                    │  New Move   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Feasible?  │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
           YES│                         │NO
              ▼                         ▼
    ┌──────────────────┐         ┌──────────┐
    │  Better          │         │  REJECT  │
    │  Objective?      │         └──────────┘
    └──────┬───────────┘
           │
      ┌────┴────┐
      │         │
   YES│         │NO
      ▼         ▼
┌─────────┐  ┌──────────┐
│ ACCEPT  │  │  REJECT  │
│         │  │          │
│ Reset NL│  │ Remove N │
└─────────┘  └──────────┘
```

---

**Legend:**
- `│`, `─`, `┌`, `└`, `┐`, `┘` = Flow connectors
- `▼`, `►` = Direction arrows
- `✓` = Requirement satisfied
- `NL` = Neighborhood List
- `N1, N2, ...` = Individual neighborhoods
- `MAX_INTER`, `MAX_INTRA` = Iteration limits

**Diagram Version**: 1.0
**Created**: January 22, 2026
**Author**: Claude Sonnet 4.5
