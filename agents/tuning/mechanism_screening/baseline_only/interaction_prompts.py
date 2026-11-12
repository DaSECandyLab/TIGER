"""Stage-specific prompts for iterative mechanism screening.

Keep the shared contract separate from each stage's job: preliminary comparisons
and interaction proposals never authorize deletion; final scope confidence does.
"""

COMMON = '''
# Goal and inputs
Identify clearly poor mechanism regions and promising regions for later tuning, not the fastest
fixed configuration. Use only supplied workload facts, mechanism knowledge, legal configurations,
interaction knowledge, and historical observations. Cite exact IDs. Explain judgments in Chinese.

# Evidence and attribution
Compare ALL historical runs that help explain mechanism effects, not only the latest run against
baseline. A=baseline 100s, B=CacheBlend 60s, C=CacheBlend+ngram 80s supports a CacheBlend benefit
and an ngram penalty in that background; C beating A does not establish an ngram benefit.
Inspect actual changed fields. Forced companion changes belong to a mechanism bundle; its loss
does not prove each companion independently harmful. Failed runs are not objective measurements.
Conversely, necessary companion changes do NOT invalidate a comparison of the complete mechanism
bundle. Evaluate the operational on bundle against the operational off configuration, including
both their costs and benefits. You do not need to isolate each companion's independent effect to
judge the bundle. This applies symmetrically when the on bundle wins or loses. Do not reframe a
bundle comparison as a demand to prove that prefix, chunked prefill, or quantization alone is bad.
Timeout bounds, SQL completion and request progress are evidence, not exact full-workload times.
For censored runs, use the program-provided censoring.throughput_time_estimate to assess the
approximate performance gap: total expected requests divided by measured average successful-request
throughput. Compare estimated_slowdown_vs_baseline with plausible interaction/tuning improvements.
This is a conditional forecast assuming future average throughput stays unchanged, NOT a measured
runtime, confidence interval or lower bound. Keep it separate from lower_bound_s and SQL completion;
do not overwrite objective_value or treat a projection as a completed run. Evaluate possible request
mix/cache/scheduling changes instead of ignoring the estimate just because it is a projection.

# Runtime noise policy (user-defined screening tolerance)
For comparable completed runs, compute relative runtime change as
(candidate.wall_time_s - control.wall_time_s) / control.wall_time_s, using the relevant historical
control, not necessarily baseline. An absolute change <= 0.02 (inclusive, in either direction) is
within the noise band: treat that timing contrast as inconclusive, not as evidence of a meaningful
speedup or slowdown, and do not justify deletion from that contrast alone. Use unrounded values.
This is a user-defined practical tolerance, not a statistically estimated confidence interval.
A change > 2% in absolute value is outside this band. Do not dismiss it as ordinary run-to-run
noise merely because it is small, measurements are not repeated, or prefix hit rates differ.
If concrete evidence shows a confound or measurement problem, explain its causal relevance rather
than widening the noise band implicitly. Outside-band changes still require attribution, joint
interaction, relative tuning-potential and deletion-confidence assessment; they do not automatically
establish a clearly good/bad region. For example, a 4.1% slowdown is outside the noise band, but
whether to delete its region remains a separate judgment. This timing rule does not constrain
feature-only initial screening or independent quality/legality evidence. Do not treat censored
bounds or throughput projections as exact completed runtimes to apply a +/-2% equality test.

# Scope and interactions
Hard legality is enforced by the program. Assess both hard-linked performance effects and obvious
performance interactions, jointly when several coexist. Unrelated parameters take ANY legal value.
Transfer only the effect supported by evidence. An on loss does not prove off+quantization is bad.
Missing benefits, missing measurements and numerical ordering alone do not establish clear harm.
Conversely, an interaction label or an unmeasured background is NOT sufficient to dismiss an
already observed loss. Assess evidence transfer, not whether every combination has been measured.
For each plausible reversal, explain the causal path, its direction, workload activation, and
whether its plausible magnitude could overturn the observed gap after RELATIVE tuning on both
sides. Distinguish weakening the loser's benefit from rescuing it: faster prefill that reduces
CacheBlend's avoidable work does not by itself reverse a measured CacheBlend loss. Quantization
may also change recomputation/kernel costs; evaluate that separate path if applicable. Never
assume those effects are large enough, or too small, without an evidence-based argument.
No direct measurement does not imply deletion_confidence=0. Combine the strength of measured
history with concrete transfer uncertainty. If an activated reversal path has unresolved material
magnitude, retain with an explanation of that uncertainty; do not force deletion or a high score.

# Match reversal explanations to the size of the loss
A many-fold slowdown requires a many-fold recovery explanation, not the same generic uncertainty
used for a small regression. Use throughput_time_estimate.required_speedup_to_match_baseline and
required_runtime_reduction_fraction when available. These describe the recovery needed IF the
forecast holds; they are not measured lower bounds or an automatic deletion threshold.
For example, roughly 5 minutes versus a projected 2 hours is a different screening question from
5 minutes versus 6 minutes. A hypothetical 2x or 4x improvement does not close a roughly 24x gap.
Do not treat "quantization may help", "the interaction is unmeasured", "context dependent", or
"there are tuning knobs" as sufficient reversal explanations for such a gap.
For an interaction group sharing the inferior mechanism bundle, evaluate BOTH possible rescues:
(1) The throughput forecast is misleading: specify the workload/cache/request-mix transition that
could raise future throughput enough to close the gap, using available progress and workload facts.
(2) The interaction/tuning changes performance: identify the affected dominant cost and explain
whether removing/reducing it could plausibly meet the required recovery, including improvements
available to the winning side. Do not assume an unknown cost is dominant or invent a speedup.
Weigh speculative rescue paths against the measured slowdown, low completion progress and workload
mechanism evidence. Lack of an exact cost breakdown is not an automatic veto on screening-level
extrapolation. If no workload-supported rescue plausibly approaches the required magnitude and the
loss attribution transfers, a high-confidence exclusion of the interaction group is allowed without
measuring that group. If rescue remains materially plausible, identify why and retain uncertainty.
The numerical gap alone is insufficient: preserve evidence, attribution and confidence checks.
Consider counterexamples and concrete relative tuning potential on both sides (including topology,
batching and memory when applicable). Unknown tuning ranges do not mean those directions are absent.
CacheBlend applicability: long input, limited useful native prefix reuse, high non-prefix segment
reuse opportunity. Speculation applicability: long output / substantial decode work. These are
priors, not guarantees or automatic exclusions; a mechanism with low benefit may still have large cost.

# Deletion confidence and execution
deletion_confidence in [0,1] means confidence that the ENTIRE stated region is clearly poor enough
to exclude after accounting for attribution, all history, joint interactions and downstream tuning.
It is not confidence that one measured point was slower, not a confidence of being fastest, and
not a statistically calibrated probability. confidence_reason must explain evidence strength,
transfer uncertainty and the strongest remaining reversal possibility. Do not aim for the threshold,
round up to obtain deletion, average away uncertain members, or invent numeric speedups.
Only a supported drop with deletion_confidence >= policy.deletion_confidence_threshold executes.
Below-threshold drops become unresolved and remain eligible. High confidence NEVER bypasses evidence,
attribution, legality or counterexample checks. Conflicting members need a relevant narrower scope
or an unresolved group. Deletions are final for this screening run: never reopen or reassess removed
regions. Later rounds use all historical observations but update ONLY the surviving candidates.
Initial feature screening and final all-history scope decisions are the only performance deletion
gates. Local good/bad signals and interaction proposals are provisional.

# Output
Return only JSON matching output_schema; preserve IDs and field names. No deletion quota.
'''

STAGES = {
    'performance_interaction_initial': '''
# Task: simple workload-feature initial screening (no new runs)
This stage is a lightweight feature-based heuristic, not a performance-cost proof. Use
screening.feature_prescreen_policy. Do not apply the later measured-gap analysis as a prerequisite.
1. For each of the seven mechanisms, identify clear relevant workload signals and the mechanism
   direction they support. Give a short feature -> mechanism effect -> scope judgment explanation.
   Use only supplied facts/knowledge and legal values. No synthetic timing or speedup estimates.
2. Use broad conditional scopes, disjoint and exhaustive within each mechanism. Review the listed
   hard/performance links briefly. Split only concrete relevant backgrounds; unrelated parameters
   remain ANY legal value. Separate forced companions from independently chosen settings.
3. Prefix-cache rule: when measured useful prefix reuse is clearly present (high actual cache hits
   with reusable full prefix blocks), favor prefix=on and propose dropping prefix=off AND
   CacheBlend=off. Losing that reuse is sufficient feature-level cost under the user's heuristic.
   This rule requires substantial reuse, not merely nonzero reuse. A low hit rate, the existence
   of full prefix blocks, and prefill dominance do not by themselves establish a clearly poor
   prefix-off region. When the high-reuse condition is absent, keep that scope unresolved unless
   independent evidence supports exclusion; do not apply the rule and then demand a counterexample.
   Do not demand an on/off run or a quantified hashing/bookkeeping cost comparison. Short inputs
   alone, unmeasured overhead, and generic future tuning are NOT exceptions to this feature rule.
   Preserve or split exceptions only when supplied facts/knowledge identify an active counter-path,
   such as material cache pressure, or an alternative reuse mechanism. CacheBlend-on is separate.
4. Give promising/unfavorable/unresolved and deletion_confidence with concise confidence_reason.
   Clear direction and no evidenced active exception can justify a high-confidence feature exclusion.
   Missing/contradictory signals remain unresolved. Do not mechanically set all unresolved scores to 0.
   Other mechanisms also use relevant feature signals; lack of opportunity alone is not proof of harm.
5. Put exclusions in rules and cover them with unfavorable scopes. Existing rule fields may be short:
   benefit_limitation_evidence = why that branch lacks the indicated benefit;
   cost_evidence = lost useful work or an applicable mechanism cost;
   scope_justification = relevant condition and actual exceptions;
   optimization_review = whether a KNOWN tuning/interaction path changes the feature direction.
   Do not invent hypothetical exceptions or request detailed cost breakdowns to fill these fields.
The confidence gate and independent scope audit still apply. Feature-based exclusions are heuristic,
not measured speed claims. Empty rules are valid when no clear directional feature signal exists.
''',
    'performance_interaction_initial_audit': '''
# Task: lightweight feature-exclusion audit
Review each proposed_initial_rule once by rule_id. Check exact feature citations, whether the
feature-to-mechanism direction follows screening.feature_prescreen_policy, correct scope, and any
concrete active exception supported by the supplied facts/knowledge. Give supported/defer plus
independent deletion_confidence and confidence_reason. Do not copy a proposed confidence.
For useful measured prefix reuse, loss of that reuse is sufficient feature-level cost for
prefix=off AND CacheBlend=off under the user's heuristic. Do not veto solely for missing on/off
runs, unknown bookkeeping costs, short inputs, or unspecified future tuning. An actual active
counter-path can justify defer or a narrower scope. Do not convert hit rate into runtime savings.
The program uses min(proposal, audit) and executes only supported scores at or above the threshold.
''',
    'performance_interaction_decision': '''
# Task: choose the next screening question and representative
1. Use accumulated history, current scope judgments/confidence and unresolved interaction questions.
   Refresh questions from the surviving scopes every round. Refinement is evidence-driven, not
   round-driven: split only when an activated interaction or historical contrast suggests different
   screening decisions within a scope. Keep a broad question when members still share one causal
   uncertainty; move to another mechanism when the previous question is settled. Several old
   questions may be replaced by one common question if new evidence reveals a shared cause.
   Do not progressively freeze unrelated parameters or enumerate tuples just because rounds elapsed.
   Each new question must identify the unresolved premise and how good/bad outcomes could resolve it.
2. Compare useful questions by meaningful performance impact, ability to change retention/deletion,
   interpretability against historical runs, and importance after downstream tuning. A likely bad
   configuration can be worth running to confirm an important exclusion. Raw member counts,
   probability/expected-deletion scores, runtime-cost scores and fastest-point search are not criteria.
3. Choose one runnable, unattempted representative and an existing complete operational control.
   The predicate defines the question's full region: omitted dimensions mean ANY legal value,
   even if question prose describes a fixed representative. varying_dimensions must explain
   EVERY mechanism that actually varies among predicate members, not just the experimental change.
   For example, {cacheblend_mode:[off]} leaves prefix, chunked, quantization, dtype, KV dtype and
   speculation varying; account for all of them. If a question truly requires a fixed background,
   encode those constraints explicitly in predicate. Do not freeze unrelated dimensions just
   to bypass this requirement; explain why they share the causal uncertainty instead.
   Prefer changing the questioned mechanism plus necessary companions. The selected control does
   not limit the historical comparisons or scope decisions after the run. Do not cripple a background.
4. Explain exactly good/bad forecast branches, the strongest alternative question and what each
   result could clarify; forecasts cannot pre-authorize deletion. Stop if only tuning-level details
   remain or there is no consequential, interpretable screening question.
   Low expected BENEFIT is not low VERIFICATION VALUE. A weak-benefit mechanism with applicable
   overhead can be a valuable exclusion test. Missing acceptance rates, kernel effects or other
   causal evidence may define the next question, not justify stopping. Settling one mechanism does
   not settle others. Distinguish testing whether a mechanism family should survive from optimizing
   a numerical setting within an already retained family. A remaining single-run budget is usable,
   not by itself a stop reason. Stops undergo an independent review of surviving mechanism questions.
predicate describes the represented region. varying_dimensions lists ONLY dimensions varying
within that region; changed_parameters lists actual treatment/control changes. action=run needs
decision and null stop_reason; action=stop needs null decision and a substantive stop_reason.
''',
    'performance_interaction_stop_audit': '''
# Task: independently audit whether screening can stop
Review EVERY required_mechanism_reviews entry exactly once using ALL supplied historical evidence,
remaining candidates, workload features and actual experiment_options. Do not review deleted scopes.
For each mechanism state the best remaining screening_question and the screening effect of a
clearly good AND clearly bad result. Set can_change_screening if either could materially change
retention/deletion or resolve a promising/unresolved family; set interpretable_test_exists based on
an available unattempted representative and a complete historical control. Explain both judgments.
Low potential acceleration does NOT imply low test value: confirming a weak-benefit mechanism's
overhead can eliminate its enabled family. Lack of acceptance-rate/kernel-cost evidence may be
exactly the premise to test. Do not dismiss such family-level questions as tuning details just
because their benefit is uncertain, the workload is short, memory pressure is low, or another
mechanism has been settled. Numerical optimum search within a retained family is tuning; deciding
whether the family survives is screening. Assess both possible outcomes without assuming either.
Do not require every mechanism to be measured: already resolved questions, evidence-supported
negligible effects in both directions, or absence of an interpretable feasible contrast may justify
stopping. Explain the concrete history/feature basis; no generic "remaining knobs are tuning".
Return continue_screening iff at least one review has both can_change_screening=true and
interpretable_test_exists=true. Otherwise return stop_supported. Do not prefer any named mechanism,
use runtime cost/coverage scores, or invent available cached results. Cite supplied evidence IDs.
''',
    'performance_propagation_signal': '''
# Task: explain the new local comparison, not final scope quality
Identify the focal changed mechanism bundle and label its effect good/bad/inconclusive relative
to the nominated control, checking ALL history for confounding or counterexamples. Cite both runs.
Mechanically required companions may change; unrelated simultaneous changes prevent attribution.
Numerical ordering alone is not a clear signal. A censored slowdown bound can support bad, not good.
This is a provisional attribution used to construct candidate transfer regions. It neither assigns
final deletion confidence nor determines the global screening scope. The all-history stage does that.
''',
    'performance_linked_groups': '''
# Task: propose joint interaction-group judgments
Evaluate each supplied group_id exactly once. The supplied groups are disjoint; their joint tuples
define the actual member backgrounds. Do not invent groups, predicates or external configurations.
Consider ALL listed relation_ids jointly, including companion-to-companion links; explain combined
effect, reversal potential and plausible magnitude after downstream tuning. Cite applicable history,
facts and knowledge. Hard compatibility alone does not answer the performance question.
For each interaction, distinguish effects that preserve/worsen the measured loss from effects
that could reverse it. Compare plausible reversal scale with the observed loss and workload
progress evidence, including the supplied conditional throughput-time estimate. Merely listing tunable
knobs, saying context_dependent, or noting no group measurement is not an adequate assessment.
magnitude_assessment must state the approximate recovery needed and whether the described joint
interaction/tuning or forecast-error path could approach it; distinguish evidence from assumption.
Return promising/drop/unresolved plus deletion_confidence and confidence_reason for the whole group.
If its members cannot share a conclusion, use unresolved. These are provisional proposals: both
these groups and direct-transfer proposals receive a final all-history confidence review.
''',
    'performance_history_scope': '''
# Task: final all-history scope and confidence decision, every iteration
1. Read ALL observations together, including earlier useful comparisons and contrary evidence.
   Each proposed region includes measured_members computed from exact combination membership.
   Review and cite those observations. If nonempty, NEVER call the region entirely unmeasured;
   distinguish its measured anchor from its unmeasured members. Baseline is a real measured member.
   Use the full observation details for quality, timing bounds and configuration differences.
   measured_history_contrasts enumerates completed feasible run pairs, actual changed fields,
   objective gaps and activated hard links. Start from those observed net effects, then identify
   which changed fields form the questioned mechanism plus required companions and which are
   independent changes. Independent changes limit attribution; required companions are part of
   the operational mechanism being evaluated. Link relevant contrasts to each region's measured
   anchors and inferior mechanism signature before assessing transfer. Do not confuse absence of
   an isolated-field ablation with absence of a measured operational contrast.
   Evaluate every proposed region_id exactly once. Local labels and prior deletion labels are
   proposals, not facts. Only surviving candidates are in scope; previously deleted regions are
   not reconsidered. Low-confidence deletion proposals become unresolved/retain.
2. Explain historical_reasoning, joint interaction_assessment,
   tuning_assessment, untested_conditions, next_evidence, deletion_confidence and confidence_reason.
   Historical evidence strength must govern confidence, not the proposed label or threshold.
   Assess the evidence and score BEFORE deciding promising/drop/unresolved. An unresolved verdict
   is not a zero deletion score: it may contain substantial evidence of harm with unresolved
   transfer uncertainty. Do not encode verdicts as 0 for retained and a high number for dropped.
   confidence_reason must distinguish (a) evidence supporting harm across this group,
   (b) evidence/unknowns limiting transfer, and (c) why their balance supports this score.
   If you assign 0 despite an applicable measured loss, explain why that loss provides no support
   for this entire scope; absence of a direct group measurement is not that explanation.
   For many-fold measured/projected losses, explicitly apply the shared magnitude test in
   interaction_assessment and tuning_assessment. A generic unmeasured interaction does not justify
   disregarding the gap; any retention explanation must address the recovery needed or a concrete
   reason the forecast substantially overstates the gap. Independently reassess weak provisional
   retention arguments instead of copying their verdict or score.
   Independently challenge both proposed deletion AND proposed retention. For retention despite
   a transferable measured loss, identify the actual reversal path or precise missing premise
   that makes its magnitude materially uncertain. Reject directionally inconsistent reversal
   stories. An unmeasured interaction alone cannot erase the historical negative evidence.
   关键区别：若已观测 CacheBlend-on 捆绑明显差，评价 CacheBlend-on + quantization
   时，目标仍是判断 CacheBlend-on 捆绑的劣势能否迁移，并不是证明 quantization 有害。
   “不能把损失归因于量化”与本问题无关，不能作为保留理由。先承认已有捆绑负证据，
   再具体评估量化是否会改变该劣势。若量化只是加快可被复用的 prefill，这会削弱
   CacheBlend 的节省价值；若量化还能减少实际重算或其他关键成本，必须指出该不同
   成本路径，并评估其是否在当前 workload 激活、是否可能抵消实测差距。也要考虑
   CacheBlend-off 一侧可获得的相同量化和调优收益。不要把“尚未测量”当成结论，
   不要求额外实跑才能进行这种带置信度的机制外推，也不得为了删除而忽略真实不确定性。
3. When the CacheBlend-on bundle wins, its required prefix-off/chunked-off/eager settings are
   already included in the measured net benefit. The off-side prefix/chunked benefits are already
   included in the slower baseline; merely listing them does not explain a reversal. For example,
   266s off versus 152s on is measured bundle evidence, not an unmeasured off-side hypothesis.
   Evaluate whether the off-side's interactions and RELATIVE downstream tuning could close that
   gap. No individual-companion ablation is required. Do not infer that every off member loses:
   transfer must still match the supported loss signature and assess concrete reversal paths.
   If retention is warranted, state the actual missing transfer premise, not 'no direct run'.
   For each measured-history drop, supporting_comparisons identifies whichever historical better
   and worse observations support the loss. Neither must be the latest run or baseline. Explain
   the combined historical case and relevant counterexamples, not merely a pairwise numeric rank.
4. loss_signature is a conjunction of actual inferior changed MECHANISM VALUES in that comparison.
   It describes the measured losing configuration, NOT all values in the target region. Do not
   copy target-background values into it if the losing observation had different values. When a
   focal mechanism's hard links explain companion changes, the signature may contain only that
   focal mechanism. Evaluate changed target companion backgrounds in transfer_reason and the
   interaction assessment, rather than inventing measured companion values. This works in BOTH
   directions (on loses or off loses); the same attribution and transfer checks still apply.
   Every dropped member must match a supported signature. It must explain all jointly changed
   mechanisms except mandatory companions entailed by the signature's hard links. You cannot
   attribute a bundle loss to an isolated companion. Give an explicit transfer_reason addressing
   the members' interaction backgrounds and relative tuning potential.
   Example: baseline vs CacheBlend-on with required prefix/chunked off supports CacheBlend-on.
   It cannot delete CacheBlend-off, or isolate prefix-off as the cause. Other history may provide
   a separate justified signature. B vs C in the A/B/C example can support ngram-on in its context.
5. Failed results cannot be worse timing witnesses; censored results are lower bounds. Preserve
   competitive measured members. A group with unsupported members is unresolved, not an averaged
   high-confidence deletion. Missing positive evidence is not negative evidence.
6. Do not create new feature-only exclusions in this post-run stage. Deleted candidates stay
   deleted, while their historical measurements remain available for interpreting surviving scopes.
The program records your proposed verdict and score, then separately applies the confidence gate.
''',
    'performance_interaction_assess': '''
# Task: contextual regional proposal (compatibility path)
Assess the outcome and all history using conditional scopes. Preserve complete region coverage,
joint hard/performance interaction explanations, tuning possibilities and exact evidence references.
These labels are proposals, never authority to bypass the final all-history confidence gate.
''',
}


def register_interaction_prompts(tasks):
    tasks.update({name: body + COMMON for name, body in STAGES.items()})
