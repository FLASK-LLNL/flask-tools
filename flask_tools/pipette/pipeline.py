from __future__ import annotations
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import inspect
import traceback

from .config import PipetteConfig
from .verifiers import (
    BasicSmilesValidationChecker,
    ChargeConservationChecker,
    ExactMatchChecker,
    MassConservationChecker,
    ReactionChecker,
    ReactionEnergyChecker,
)
from .verifiers.base import Speed, CacheableReactionChecker
from .judges import AsyncLLMJudge
from .constants import ReactionGrade, ToolResult, ToolStatus
from .reaction_fixer import AsyncLLMReactionFixer, ReactionFix
from .rules import apply_exact_rules
from .smiles import canonicalize_reaction_smiles
from .llm_query import _run_coroutine_sync

CheckerFactory = Callable[[PipetteConfig], ReactionChecker]
LLMJudge = AsyncLLMJudge
LLMReactionFixer = AsyncLLMReactionFixer


@dataclass
class PendingReaction:
    """Reaction for which some tools have been run, but not all"""

    rxn_smiles: str
    prefix_results: list[tuple[str, str, ToolResult]] = field(default_factory=list)


def resolve_tool_list(
    tool_list: str | list[str], available_tools: list[str]
) -> list[str]:
    if tool_list == "all":
        return list(available_tools)

    if not isinstance(tool_list, list):
        raise ValueError(
            "PipetteConfig.tool_list must be 'all' or a list of tool names."
        )
    if len(set(tool_list)) != len(tool_list):
        raise ValueError(
            "PipetteConfig.tool_list must not contain duplicate tool names."
        )

    unknown = set(tool_list) - set(available_tools)
    if unknown:
        raise ValueError(
            f"Unknown tool names in PipetteConfig.tool_list: {', '.join(sorted(unknown))}"
        )

    return list(tool_list)


class GradingPipeline:
    def __init__(
        self,
        checkers: list[ReactionChecker],
        config: PipetteConfig | None = None,
        judge: AsyncLLMJudge | None = None,
        reaction_fixer: AsyncLLMReactionFixer | None = None,
    ) -> None:
        """judge: If provided, overrides the judge specified by the config"""
        self.checkers = checkers
        self.config = config or PipetteConfig()
        self.judge = judge
        self.reaction_fixer = reaction_fixer
        self._validate_configuration()
        if self.config.mode == "ai" and self.judge is None:
            self.judge = AsyncLLMJudge.from_config(self.config)
        if self.reaction_fixer is None:
            self.reaction_fixer = AsyncLLMReactionFixer.from_config(self.config)

    def _validate_configuration(self) -> None:
        checker_names = [checker.name for checker in self.checkers]
        if len(set(checker_names)) != len(checker_names):
            raise ValueError("Checker names must be unique within a grading pipeline.")

        if self.config.mode not in {"exact", "ai"}:
            raise ValueError(f"Unsupported pipeline mode: {self.config.mode!r}")

        if self.config.mode == "ai":
            unknown = sorted(set(self.config.llm_judge.allow_fail) - set(checker_names))
            if unknown:
                raise ValueError(
                    "Unknown tool names in PipetteConfig.llm_judge.allow_fail: "
                    + ", ".join(unknown)
                )

    def _should_skip_remaining(
        self, checker: ReactionChecker, result: ToolResult
    ) -> bool:
        if (
            not self.config.rules.stop_on_hard_fail
            or not checker.stops_on_fail
            or (result.status not in {ToolStatus.FAIL, ToolStatus.ERROR})
            or (
                self.config.mode == "ai"
                and self.config.llm_judge.allow_fail == "all"
                or (checker.name in self.config.llm_judge.allow_fail)
            )
        ):
            return False
        return True

    @staticmethod
    def _should_try_llm_fix(context: dict[str, ToolResult]) -> bool:
        basic = context.get("basic_smiles_validation")
        exact_match = context.get("exact_match")
        charge = context.get("charge_conservation")
        mass = context.get("mass_conservation")
        if basic is None or exact_match is None or charge is None or mass is None:
            return False
        return (
            basic.status is ToolStatus.PASS
            and exact_match.status is ToolStatus.UNKNOWN
            and charge.status is ToolStatus.PASS
            and mass.status is ToolStatus.FAIL
        )

    @staticmethod
    def _with_prefix_results(
        result: ReactionGrade | PendingReaction,
        prefix_results: list[tuple[str, str, ToolResult]],
    ) -> ReactionGrade | PendingReaction:
        """Combine the tool results from the prefix_results with the results"""
        if not prefix_results:
            return result
        if isinstance(result, PendingReaction):
            return PendingReaction(
                rxn_smiles=result.rxn_smiles,
                prefix_results=[*prefix_results, *result.prefix_results],
            )
        return ReactionGrade(
            final_grade=result.final_grade,
            short_reason=result.short_reason,
            results=[tool_result for _, _, tool_result in prefix_results],
            comment=result.comment,
        )

    @staticmethod
    def _build_fix_result(rxn_smiles: str, fix: ReactionFix) -> ToolResult:
        return ToolResult(
            name="llm_reaction_fix",
            status=ToolStatus.PASS,
            data={
                "original_reaction_smiles": rxn_smiles,
                "fixed_reaction_smiles": fix.fixed_reaction_smiles,
                "removed_agents": fix.removed_agents,
                "added_reactants": fix.added_reactants,
                "added_products": fix.added_products,
            },
            comment=fix.reasoning_summary
            or "LLM proposed a corrected reaction and the pipeline was rerun.",
        )

    async def _attempt_llm_fix_async(
        self,
        rxn_smiles: str,
        context: dict[str, ToolResult],
    ) -> tuple[ToolResult, str | None] | None:
        if self.reaction_fixer is None:
            return None

        try:
            fix_result = self.reaction_fixer.fix(
                rxn_smiles,
                list(context.values()),
            )
            fix = await fix_result if inspect.isawaitable(fix_result) else fix_result
        except Exception as exc:
            return (
                ToolResult(
                    name="llm_reaction_fix",
                    status=ToolStatus.ERROR,
                    comment=f"LLM reaction fixer failed: {exc}",
                ),
                None,
            )

        if fix.fixed_reaction_smiles == canonicalize_reaction_smiles(
            rxn_smiles, include_agents=True
        ):
            return (
                ToolResult(
                    name="llm_reaction_fix",
                    status=ToolStatus.UNKNOWN,
                    data={
                        "original_reaction_smiles": rxn_smiles,
                        "fixed_reaction_smiles": fix.fixed_reaction_smiles,
                    },
                    comment="LLM reaction fixer did not propose a changed reaction.",
                ),
                None,
            )

        return self._build_fix_result(rxn_smiles, fix), fix.fixed_reaction_smiles

    async def _finalize_grade_async(
        self,
        rxn_smiles: str,
        context: dict[str, ToolResult],
        prefix_results: list[tuple[str, str, ToolResult]],
    ) -> ReactionGrade:
        """Exact grading happens without the prefix_results, namely the results before llm-fix of rxn balance.
        AI grading does use it, because it should be smart enough to deal with it.
        """
        if self.config.mode == "exact":
            res = apply_exact_rules(list(context.values()))
            assert res is not None
            res = self._with_prefix_results(res, prefix_results)
            assert isinstance(res, ReactionGrade)
            return res

        if self.config.mode == "ai":
            if self.judge is None:
                raise ValueError("AI mode requires an LLM judge implementation.")

            results = [pr[-1] for pr in prefix_results] + list(context.values())
            judge_result = self.judge.judge(
                rxn_smiles,
                results,
            )
            res = (
                await judge_result
                if inspect.isawaitable(judge_result)
                else judge_result
            )
            assert res is not None
            res = self._with_prefix_results(res, prefix_results)
            assert isinstance(res, ReactionGrade)
            return res

        raise ValueError(f"Unsupported pipeline mode: {self.config.mode!r}")

    async def grade_one_async(
        self,
        rxn_smiles: str,
        tiered_run: bool = False,
        *,
        fix_attempted: bool = False,
        prefix_results: list[tuple[str, str, ToolResult]] | None = None,
    ) -> ReactionGrade | PendingReaction:
        """
        tiered_run: Will return None if there are long-running checks in the pipeline without cached result
        prefix_results: grade_one can be called twice for one reaction if tiered_run is True. In that case, the tool
            results from the first call will be passed in here.
        """
        prefix = list(prefix_results or [])
        prefix_smiles_checkers = {
            (smi, checker_name): res for smi, checker_name, res in prefix
        }
        context: dict[str, ToolResult] = {}
        should_skip_remaining = False
        skip_reason = ""

        for checker in self.checkers:
            if (rxn_smiles, checker.name) in prefix_smiles_checkers:
                context[checker.name] = prefix_smiles_checkers[
                    (rxn_smiles, checker.name)
                ]
                continue
            if should_skip_remaining:
                result = checker.skipped(skip_reason)
            else:
                if checker.speed == Speed.CACHEABLE_SLOW:
                    assert isinstance(checker, CacheableReactionChecker)
                try:
                    if tiered_run:
                        result = None
                        has_cache = isinstance(checker, CacheableReactionChecker)
                        is_slow = checker.speed in (Speed.SLOW, Speed.CACHEABLE_SLOW)
                        if has_cache or is_slow:
                            if has_cache:
                                result = checker.check_cache(rxn_smiles)
                            if result is None:
                                # This returns without the current context. But that should only contain fast results, so it's ok if they are repeated
                                # But it would be a nice improvement if it did.
                                return PendingReaction(
                                    rxn_smiles=rxn_smiles, prefix_results=prefix
                                )
                        else:
                            result = checker.run(rxn_smiles, context)
                    else:
                        result = checker.run(rxn_smiles, context)
                except Exception as exc:
                    result = checker.errored(
                        f"{checker.name} raised an unexpected error: {exc}",
                        traceback_text=traceback.format_exc(),
                    )

                if self._should_skip_remaining(checker, result):
                    should_skip_remaining = True
                    skip_reason = f"Skipped after hard failure in {checker.name}."

            prefix.append((rxn_smiles, checker.name, result))
            context[checker.name] = result

            # Reaction fixing / infilling of byproducts
            # If LLM fixing returned a value, call new grade_one with new rxn and
            # it's main tool result list will start from the new rxn
            if (
                checker.name == "mass_conservation"
                and not fix_attempted
                and self._should_try_llm_fix(context)
            ):
                if (rxn_smiles, "llm_reaction_fix") in prefix_smiles_checkers:
                    raise RuntimeError("llm_reaction_fix already ran")
                fix_attempt = await self._attempt_llm_fix_async(
                    rxn_smiles,
                    context,
                )
                if fix_attempt is not None:
                    fix_result, fixed_rxn_smiles = fix_attempt
                    if fixed_rxn_smiles is None:
                        context[fix_result.name] = fix_result
                    else:
                        return await self.grade_one_async(
                            fixed_rxn_smiles,
                            tiered_run=tiered_run,
                            fix_attempted=True,
                            prefix_results=[
                                *prefix,
                                (rxn_smiles, "llm_reaction_fix", fix_result),
                            ],
                        )

        return await self._finalize_grade_async(rxn_smiles, context, prefix)

    def grade_one(
        self,
        rxn_smiles: str,
        tiered_run: bool = False,
        *,
        fix_attempted: bool = False,
        prefix_results: list[tuple[str, str, ToolResult]] | None = None,
    ) -> ReactionGrade | PendingReaction:
        return _run_coroutine_sync(
            self.grade_one_async(
                rxn_smiles,
                tiered_run=tiered_run,
                fix_attempted=fix_attempted,
                prefix_results=prefix_results,
            )
        )

    def grade(
        self, rxn_smiles_list: list[str], tiered_run=True
    ) -> Iterator[ReactionGrade]:
        """tiered_run: Simple two tiered/two phase scheduling. Will first yield answers for molecules that can avoid
        long-running checks, like DFT. On the second pass all checks will run for remaining molecules.  If False,
        run all checks in one phase.
        """
        run_later: list[PendingReaction] = []
        for rxn_smiles in rxn_smiles_list:
            res = self.grade_one(rxn_smiles, tiered_run=tiered_run)
            if isinstance(res, PendingReaction):
                run_later.append(res)
            else:
                yield res
        if tiered_run:
            while run_later:
                pending = run_later.pop()
                fix_results = [
                    result
                    for _, _, result in pending.prefix_results
                    if result.name == "llm_reaction_fix"
                ]
                res = self.grade_one(
                    pending.rxn_smiles,
                    tiered_run=False,
                    fix_attempted=bool(fix_results),
                    prefix_results=pending.prefix_results,
                )
                assert not isinstance(res, PendingReaction)
                yield res

    # def grade_new(self, rxn_smiles_list: list[str]) -> Iterator[ReactionGrade]:
    #     """WIP"""
    #     results: list[list[ToolResult]] = [[] * len(rxn_smiles_list)]
    #     context: list[dict[str, ToolResult]] = [{} * len(rxn_smiles_list)]
    #
    #     found_slow = False
    #     fast_checkers = []
    #     slow_checkers = []
    #     for checker in self.checkers:
    #         if checker.speed in (Speed.SLOW, Speed.CACHEABLE_SLOW):
    #             slow_checkers.append(checker)
    #             if found_slow:
    #                 raise ValueError("Slow verifiers must be at the end of the pipeline.")  # Or, could just let the slow things happen in the middle w/ caveat that it isn't optimized for
    #             found_slow = True
    #         else:
    #             fast_checkers.append(checker)
    #
    #     # Go through fast verifiers
    #     for rxn_i, rxn_smiles in enumerate(rxn_smiles_list):
    #         should_skip_remaining = False
    #         skip_reason = ""
    #         for checker in fast_checkers:
    #             if should_skip_remaining:
    #                 result = checker.skipped(skip_reason)
    #             else:
    #                 try:
    #                     result = checker.run(rxn_smiles, context)
    #                 except Exception as exc:
    #                     result = checker.errored(
    #                         f"{checker.name} raised an unexpected error: {exc}",
    #                         traceback_text=traceback.format_exc(),
    #                     )
    #
    #                 if self._should_skip_remaining(checker, result):
    #                     should_skip_remaining = True
    #                     skip_reason = f"Skipped after hard failure in {checker.name}."
    #             if should_skip_remaining:
    #                 result._done = True
    #             else:
    #                 result._done = False
    #             results[rxn_i].append(result)
    #             context[rxn_i][checker.name] = result
    #
    #     # For those that are done (early fail or pass), yield
    #     for rxn_i, rxn_smiles in enumerate(rxn_smiles_list):
    #         tool_results: list = list(context[rxn_i].values())
    #         if tool_results[-1]._done:
    #             yield tool_results
    #
    #     # Continue with the slow verifiers
    #     # Returned the cacheable ones first
    #     deferred_inds: set[int] = set()
    #     for rxn_i, rxn_smiles in enumerate(rxn_smiles_list):
    #         if list(context[rxn_i].values())[-1]._done:
    #             continue
    #         should_skip_remaining = False
    #         skip_reason = ""
    #         for checker in slow_checkers:
    #             if checker.name in context[rxn_i]:
    #                 # Skip if checker has been run already, like when a cached result was found before
    #                 continue
    #             if should_skip_remaining:
    #                 result = checker.skipped(skip_reason)
    #             else:
    #                 try:
    #                     result = None
    #                     if isinstance(checker, CacheableReactionChecker) or checker.speed == Speed.CACHEABLE_SLOW:
    #                         result = checker.check_cache(rxn_smiles)
    #                         if result is None:
    #                             deferred_inds.add(rxn_i)
    #                             break
    #                     else:
    #                         deferred_inds.add(rxn_i)
    #                         break
    #                 except Exception as exc:
    #                     result = checker.errored(
    #                         f"{checker.name} raised an unexpected error: {exc}",
    #                         traceback_text=traceback.format_exc(),
    #                     )
    #
    #                 if self._should_skip_remaining(checker, result):
    #                     should_skip_remaining = True
    #                     skip_reason = f"Skipped after hard failure in {checker.name}."
    #             if should_skip_remaining:
    #                 result._done = True
    #             else:
    #                 result._done = False
    #             context[rxn_i][checker.name] = result
    #
    #     # Run the not cached ones
    #     for rxn_i, rxn_smiles in enumerate(rxn_smiles_list):
    #         if list(context[rxn_i].values())[-1]._done or rxn_i in deferred_inds:
    #             continue
    #         should_skip_remaining = False
    #         skip_reason = ""
    #         for checker in slow_checkers:
    #             if checker.name in context[rxn_i]:
    #                 # Skip if checker has been run already, like when a cached result was found before
    #                 continue
    #             if should_skip_remaining:
    #                 result = checker.skipped(skip_reason)
    #             else:
    #                 try:
    #                     result = checker.run(rxn_smiles, context)
    #                 except Exception as exc:
    #                     result = checker.errored(
    #                         f"{checker.name} raised an unexpected error: {exc}",
    #                         traceback_text=traceback.format_exc(),
    #                     )
    #
    #                 if self._should_skip_remaining(checker, result):
    #                     should_skip_remaining = True
    #                     skip_reason = f"Skipped after hard failure in {checker.name}."
    #             if should_skip_remaining:
    #                 result._done = True
    #             else:
    #                 result._done = False
    #             context[rxn_i][checker.name] = result
    #
    #     if self.config.mode == "exact":
    #         return apply_exact_rules(results)
    #
    #     if self.config.mode == "ai":
    #         if self.judge is None:
    #             raise ValueError("AI mode requires an LLM judge implementation.")
    #         return self.judge.judge(rxn_smiles, results, self.config)
    #
    #     raise ValueError(f"Unsupported pipeline mode: {self.config.mode!r}")


def build_default_pipeline(
    config: PipetteConfig | None = None,
    *,
    judge: AsyncLLMJudge | None = None,
    reaction_fixer: AsyncLLMReactionFixer | None = None,
    checker_factories: dict[str, CheckerFactory] | None = None,
) -> GradingPipeline:
    resolved = config or PipetteConfig()
    factories = checker_factories or {
        "basic_smiles_validation": lambda _: BasicSmilesValidationChecker(),
        "exact_match": lambda _: ExactMatchChecker(),
        "charge_conservation": lambda _: ChargeConservationChecker(),
        "mass_conservation": lambda current: MassConservationChecker(current),
        "reaction_energy": lambda current: ReactionEnergyChecker(
            current,
            database=(
                str(current.tools_setting.reaction_energy.database)
                if current.tools_setting.reaction_energy.database is not None
                else None
            ),
        ),
    }
    selected_names = resolve_tool_list(resolved.tool_list, list(factories))
    checkers = [factories[name](resolved) for name in selected_names]
    return GradingPipeline(
        checkers=checkers,
        config=resolved,
        judge=judge,
        reaction_fixer=reaction_fixer,
    )
