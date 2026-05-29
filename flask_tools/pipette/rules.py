from __future__ import annotations

from .constants import FinalGrade, ReactionGrade, ToolResult, ToolStatus


def apply_exact_rules(results: list[ToolResult]) -> ReactionGrade:
    named_results = {result.name: result for result in results}

    must_pass = ["charge_conservation"]
    for result in [named_results[name] for name in must_pass]:
        if result.status in {
            ToolStatus.FAIL,
        }:
            return ReactionGrade(
                final_grade=FinalGrade.IMPOSSIBLE,
                short_reason="exact.hard_fail",
                results=results,
                comment=f"Hard failure from {result.name}.",
            )

    # Check for errors, return UNCERTAIN if any
    for result in results:
        if result.status is ToolStatus.ERROR:
            return ReactionGrade(
                final_grade=FinalGrade.UNCERTAIN,
                short_reason="exact.tool_error",
                results=results,
                comment=f"Error from {result.name}.",
            )

    exact_match = named_results.get("exact_match")
    mass = named_results.get("mass_conservation")
    energy = named_results.get("reaction_energy")

    if exact_match is not None and exact_match.status is ToolStatus.PASS:
        return ReactionGrade(
            final_grade=FinalGrade.LIKELY,
            confidence=0.95,
            short_reason="exact.database_match",
            results=results,
            comment="Exact database match found.",
        )

    if mass is not None and energy is not None:
        if mass.status is ToolStatus.PASS and energy.status is ToolStatus.PASS:
            return ReactionGrade(
                final_grade=FinalGrade.LIKELY,
                short_reason="exact.mass_and_energy_pass",
                results=results,
                comment="Mass conservation and reaction energy both passed.",
            )

        if mass.status is ToolStatus.POTENTIAL and energy.status is ToolStatus.PASS:
            print(
                f'Warning the rule for "mass.status is ToolStatus.POTENTIAL and energy.status is ToolStatus.PASS:" is somewhat arbitrary'
            )
            missing_product_confidence = mass.data.get("missing_product_confidence")
            mapping = {
                "high": FinalGrade.POSSIBLE,
                "med": FinalGrade.UNCERTAIN,
                "medium": FinalGrade.UNCERTAIN,
                "low": FinalGrade.UNLIKELY,
            }
            final_grade = mapping.get(missing_product_confidence, FinalGrade.UNCERTAIN)
            return ReactionGrade(
                final_grade=final_grade,
                short_reason=f"exact.mass_potential-{missing_product_confidence or 'unknown'}.energy_pass",
                results=results,
                comment="Reaction depends on a possible omitted species to satisfy mass balance, reaction energy passed.",
            )

    if exact_match is not None and exact_match.status is ToolStatus.POTENTIAL:
        return ReactionGrade(
            final_grade=FinalGrade.POSSIBLE,
            short_reason="exact.database_match_without_agents",
            results=results,
            comment="Reaction matched a database record after agents were ignored.",
        )

    return ReactionGrade(
        final_grade=FinalGrade.UNCERTAIN,
        short_reason="exact.insufficient_signal",
        results=results,
        comment="No hard failures were found, but the available signals are incomplete.",
    )
